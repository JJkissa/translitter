"""Offline ASR → translation → speech. No capture, playback or turn logging."""
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np

from .core import MAX_TURN_SAMPLES, MAX_TURN_SECONDS, Route
from .languages import VOICE_FILES, missing_assets, translation_steps


@dataclass(frozen=True)
class TurnResult:
    generation: int
    speaker: int
    listener: int
    source_language: str
    target_language: str
    original: str
    translation: str
    audio: np.ndarray
    rate: int
    elapsed: float
    capture_id: int | None = None


class ModelEngine:
    """Lazy local models; call on one worker thread, never the GUI thread."""

    def __init__(self, root: Path, model_name="whisper-turbo", device="cuda"):
        if model_name not in ("whisper-turbo", "whisper-small") or device not in ("cuda", "cpu"):
            raise ValueError("Select an installed Whisper model and cuda or cpu device")
        self.root = Path(root)
        self.model_name = model_name
        self.device = device
        self._loaded = False
        self.translation_backend = "marian"
        self._quality_runtime = None
        self._quality_client = None

    def set_translation_backend(self, backend, cancelled=lambda: False, progress=lambda message: None):
        """Worker-thread-only selection; failure never silently changes translators."""
        if backend not in ("marian", "translategemma"):
            raise ValueError("Select Marian or local TranslateGemma")
        if cancelled():
            return False
        if backend == "marian":
            self.close()
            return True
        if self._quality_client is None:
            # A previous failed start may still own a child. Never overwrite it.
            self.close()
            from .quality_runtime import QualityRuntime
            runtime = QualityRuntime(self.root)
            self._quality_runtime = runtime
            try:
                progress("Loading local TranslateGemma 4B")
                client = runtime.start(cancelled)
                stopped = client is None or cancelled()
            except Exception:
                self.close()
                raise
            if stopped:
                self.close()
                return False
            self._quality_client = client
        self.translation_backend = backend
        return True

    def close(self):
        """Release only the optional runtime this engine owns; safe repeatedly."""
        runtime = self._quality_runtime
        if runtime is not None:
            runtime.close()
        self._quality_runtime = self._quality_client = None
        self.translation_backend = "marian"

    def load(self, progress=lambda message: None):
        if self._loaded:
            return
        models = self.root / "models"
        for name in (self.model_name, "translate-fi-en", "translate-en-fi", "voice-fi", "voice-en"):
            if not (models / name).is_dir():
                raise FileNotFoundError(f"Required local model asset missing: {name}")
        import torch
        from faster_whisper import WhisperModel
        from transformers import MarianMTModel, MarianTokenizer
        from piper import PiperVoice

        self._torch = torch
        torch.set_num_threads(4)
        models = self.root / "models"
        progress("Loading speech recognition")
        self._whisper = WhisperModel(
            str(models / self.model_name), device=self.device,
            compute_type="int8_float16" if self.device == "cuda" else "int8",
            cpu_threads=4, local_files_only=True,
        )
        self._translations = {}
        for source, target in (("fi", "en"), ("en", "fi")):
            progress(f"Loading {source} → {target} translation")
            path = models / f"translate-{source}-{target}"
            tokenizer = MarianTokenizer.from_pretrained(path, local_files_only=True)
            model = MarianMTModel.from_pretrained(
                path, local_files_only=True, trust_remote_code=False, weights_only=True,
            ).to("cpu").eval()
            self._translations[source, target] = tokenizer, model
        self._voices = {}
        for language, filename in (
            ("fi", "fi/fi_FI/harri/medium/fi_FI-harri-medium.onnx"),
            ("en", "en/en_US/lessac/medium/en_US-lessac-medium.onnx"),
        ):
            progress(f"Loading {language} voice")
            self._voices[language] = PiperVoice.load(
                models / f"voice-{language}" / filename, use_cuda=False,
            )
        self._loaded = True

    def _load_optional(self, steps, language, progress, cancelled):
        """Extend the bounded model cache only for this turn's translation path."""
        from transformers import MarianMTModel, MarianTokenizer
        from piper import PiperVoice

        models = self.root / "models"
        for source, target in steps:
            if cancelled():
                return False
            if (source, target) not in self._translations:
                progress(f"Loading {source} → {target} translation")
                if cancelled():
                    return False
                path = models / f"translate-{source}-{target}"
                tokenizer = MarianTokenizer.from_pretrained(path, local_files_only=True)
                if cancelled():
                    return False
                model = MarianMTModel.from_pretrained(
                    path, local_files_only=True, trust_remote_code=False, weights_only=True,
                ).to("cpu").eval()
                self._translations[source, target] = tokenizer, model
                if cancelled():
                    return False
        if cancelled():
            return False
        if language not in self._voices:
            progress(f"Loading {language} voice")
            if cancelled():
                return False
            self._voices[language] = PiperVoice.load(
                models / f"voice-{language}" / VOICE_FILES[language], use_cuda=False,
            )
        return not cancelled()

    @staticmethod
    def _text(value):
        """Reject absent/garbled/special-token-only output; never guess a repair."""
        if not isinstance(value, str) or "\ufffd" in value:
            return ""
        if any(ord(char) < 32 and char not in "\n\r\t" for char in value):
            return ""
        import re
        text = value.strip()
        if not any(char.isalnum() for char in re.sub(r"<[^>]*>", "", text)):
            return ""
        return text

    def synthesize_text(self, text: str, language: str):
        return self._synthesize(text, language, lambda: False)

    def _synthesize(self, text, language, cancelled):
        chunks = []
        rate = None
        iterator = iter(self._voices[language].synthesize(text))
        while not cancelled():
            try:
                chunk = next(iterator)
            except StopIteration:
                break
            if cancelled():
                return None
            audio = np.asarray(chunk.audio_float_array, dtype=np.float32)
            next_rate = int(chunk.sample_rate)
            if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all() or not 8000 <= next_rate <= 48000:
                raise ValueError("Voice produced invalid audio")
            if rate is not None and rate != next_rate:
                raise ValueError("Voice changed sample rate during a turn")
            chunks.append(audio)
            rate = next_rate
        if cancelled():
            return None
        if not chunks:
            raise ValueError("Voice produced no audio")
        return np.concatenate(chunks).astype(np.float32), rate

    def interpret(self, samples: np.ndarray, route: Route, generation: int,
                  cancelled=lambda: False, progress=lambda message: None):
        started = monotonic()
        if cancelled():
            return None
        if not self._loaded:
            raise RuntimeError("Load models before interpreting")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not 0 < len(samples) <= MAX_TURN_SAMPLES or not np.isfinite(samples).all():
            raise ValueError(f"Input must be finite mono 16 kHz PCM, at most {MAX_TURN_SECONDS} seconds")
        steps = translation_steps(route.source_language, route.target_language)
        missing = missing_assets(self.root, route.source_language, route.target_language,
                                 additional_only=True)
        if self.translation_backend == "translategemma":
            missing = tuple(path for path in missing if not path.startswith("translate-"))
        if missing:
            raise FileNotFoundError("Required local model asset missing: " + ", ".join(missing))
        if not self._load_optional(steps if self.translation_backend == "marian" else (), route.target_language, progress, cancelled):
            return None
        original = self.recognize(samples, route, cancelled, progress)
        if cancelled() or not original:
            return None
        return self.interpret_text(original, route, generation, cancelled, progress,
                                   started=started, prepared=True)

    def recognize(self, samples, route, cancelled=lambda: False,
                  progress=lambda message: None, *, words=False):
        """Recognize a complete bounded utterance, preserving neural VAD."""
        if cancelled():
            return None
        if not self._loaded:
            raise RuntimeError("Load models before recognizing")
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not 0 < len(samples) <= MAX_TURN_SAMPLES or not np.isfinite(samples).all():
            raise ValueError(f"Input must be finite mono 16 kHz PCM, at most {MAX_TURN_SECONDS} seconds")
        progress("Recognizing speech")
        if cancelled():
            return None
        options = {"word_timestamps": True} if words else {}
        segments, _ = self._whisper.transcribe(
            samples, language=route.source_language, beam_size=1,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False, **options,
        )
        parts = []
        timed = []
        iterator = iter(segments)
        while not cancelled():
            try:
                segment = next(iterator)
            except StopIteration:
                break
            if cancelled():
                return None
            parts.append(segment.text.strip())
            if words:
                timed.extend((word.word.strip(), float(word.start), float(word.end))
                             for word in (segment.words or []))
        if cancelled():
            return None
        text = self._text(" ".join(parts))
        return (timed if text else []) if words else text

    def interpret_text(self, original, route, generation, cancelled=lambda: False,
                       progress=lambda message: None, *, started=None, prepared=False,
                       capture_id=None):
        """Translate a released transcript or explicit text without running ASR again."""
        started = monotonic() if started is None else started
        original = self._text(original)
        if cancelled() or not original:
            return None
        steps = translation_steps(route.source_language, route.target_language)
        if not prepared:
            missing = missing_assets(self.root, route.source_language, route.target_language,
                                     additional_only=True)
            if self.translation_backend == "translategemma":
                missing = tuple(path for path in missing if not path.startswith("translate-"))
            if missing:
                raise FileNotFoundError("Required local model asset missing: " + ", ".join(missing))
            if not self._load_optional(steps if self.translation_backend == "marian" else (), route.target_language, progress, cancelled):
                return None
        translation = self.translate_text(original, route, cancelled, progress)
        if cancelled() or not translation:
            return None
        progress("Synthesizing speech")
        if cancelled():
            return None
        speech = self._synthesize(translation, route.target_language, cancelled)
        if cancelled() or speech is None:
            return None
        audio, rate = speech
        return TurnResult(generation, route.speaker, route.listener,
                          route.source_language, route.target_language, original,
                          translation, audio, rate, monotonic() - started, capture_id)

    def translate_text(self, original, route, cancelled=lambda: False,
                       progress=lambda message: None):
        """Text-translator substitution hook; models are prepared by interpret_text."""
        steps = translation_steps(route.source_language, route.target_language)
        progress("Translating")
        if cancelled():
            return None
        if self.translation_backend == "translategemma":
            return self._quality_client.translate(original, route.source_language,
                                                  route.target_language, cancelled)
        translation = original
        for edge in steps:
            if cancelled():
                return None
            tokenizer, model = self._translations[edge]
            inputs = tokenizer([translation], return_tensors="pt", padding=True)
            if cancelled():
                return None
            with self._torch.inference_mode():
                output = model.generate(**inputs, max_new_tokens=256, num_beams=1)
            if cancelled():
                return None
            translation = self._text(tokenizer.decode(output[0], skip_special_tokens=True))
            if cancelled() or not translation:
                return None
        return translation
