"""Offline ASR → translation → speech. No capture, playback or turn logging."""
from dataclasses import dataclass
from pathlib import Path
from time import monotonic

import numpy as np

from .core import Route


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


class ModelEngine:
    """Lazy local models; call on one worker thread, never the GUI thread."""

    def __init__(self, root: Path, model_name="whisper-turbo", device="cuda"):
        if model_name not in ("whisper-turbo", "whisper-small") or device not in ("cuda", "cpu"):
            raise ValueError("Select an installed Whisper model and cuda or cpu device")
        self.root = Path(root)
        self.model_name = model_name
        self.device = device
        self._loaded = False

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
        if samples.ndim != 1 or not 0 < len(samples) <= 128000 or not np.isfinite(samples).all():
            raise ValueError("Input must be finite mono 16 kHz PCM, at most eight seconds")
        progress("Recognizing speech")
        if cancelled():
            return None
        segments, _ = self._whisper.transcribe(
            samples, language=route.source_language, beam_size=1,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
            condition_on_previous_text=False,
        )
        parts = []
        iterator = iter(segments)
        while not cancelled():
            try:
                segment = next(iterator)
            except StopIteration:
                break
            if cancelled():
                return None
            parts.append(segment.text.strip())
        original = self._text(" ".join(parts))
        if cancelled() or not original:
            return None
        progress("Translating")
        if cancelled():
            return None
        tokenizer, model = self._translations[route.source_language, route.target_language]
        inputs = tokenizer([original], return_tensors="pt", padding=True)
        if cancelled():
            return None
        with self._torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=96, num_beams=1)
        if cancelled():
            return None
        translation = self._text(tokenizer.decode(output[0], skip_special_tokens=True))
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
                          translation, audio, rate, monotonic() - started)
