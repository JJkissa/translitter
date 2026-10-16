"""Opt-in, real offline model checks. No microphone capture or device playback.

Run with TRANSLITTER_MODEL_TESTS=1 ./run-python.sh -m pytest tests -q -s
"""
import gc
import hashlib
import json
import math
import os
import re
import time
import wave
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
KEYS = (
    "whisper-small", "whisper-turbo", "translate-en-fi", "translate-fi-en",
    "voice-fi", "voice-en",
)
pytestmark = pytest.mark.skipif(
    os.environ.get("TRANSLITTER_MODEL_TESTS") != "1",
    reason="Opt-in real model integration; set TRANSLITTER_MODEL_TESTS=1",
)


@pytest.fixture(scope="session")
def evidence():
    records = []
    yield records
    directory = ROOT / ".local"
    directory.mkdir(exist_ok=True)
    (directory / "model-integration.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2) + "\n"
    )


@pytest.mark.parametrize("key", KEYS)
def test_all_asset_files_match_the_pinned_manifest(key):
    manifest = json.loads((ROOT / "model-assets.lock.json").read_text())
    assert set(KEYS) <= set(manifest["assets"])
    optional = {"translate-ru-en", "translate-en-ru", "voice-ru",
                "translate-uk-en", "translate-en-uk", "voice-uk"}
    assert set(manifest["assets"]) <= set(KEYS) | optional
    record = manifest["assets"][key]
    assert re.fullmatch(r"[0-9a-f]{40}", record["revision"])
    for entry in record["files"]:
        path = ROOT / record["directory"] / entry["name"]
        assert path.stat().st_size == entry["bytes"]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                digest.update(block)
        assert digest.hexdigest() == entry["sha256"]
        if entry["publisher_sha256"] is not None:
            assert digest.hexdigest() == entry["publisher_sha256"]


@pytest.mark.parametrize("key,text,anchors", [
    ("translate-fi-en", "Hei, miten voit tänään?", {"how", "you", "today"}),
    ("translate-en-fi", "Hello, how are you today?", {"hei", "voit", "tänään"}),
])
def test_translation_models_generate_real_target_text(key, text, anchors, evidence):
    import torch
    from transformers import MarianMTModel, MarianTokenizer

    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    torch.set_num_threads(4)
    path = ROOT / "models" / key
    tokenizer = MarianTokenizer.from_pretrained(path, local_files_only=True)
    model = MarianMTModel.from_pretrained(
        path, local_files_only=True, trust_remote_code=False, weights_only=True,
    ).eval()
    inputs = tokenizer([text], return_tensors="pt", padding=True)
    started = time.monotonic()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=96, num_beams=1)
    result = tokenizer.decode(output[0], skip_special_tokens=True)
    record = {"operation": "translation", "model": key, "input": text,
              "output": result, "seconds": round(time.monotonic() - started, 3)}
    evidence.append(record)
    print(json.dumps(record, ensure_ascii=False))
    words = set(re.findall(r"\w+", result.casefold()))
    assert anchors <= words, result


@pytest.fixture(scope="session")
def speech_samples(tmp_path_factory, evidence):
    import numpy as np
    import soundfile as sf
    from piper import PiperVoice, SynthesisConfig
    from scipy.signal import resample_poly

    directory = tmp_path_factory.mktemp("synthesized-speech")
    samples = {}
    specs = (
        ("fi", "voice-fi", "fi/fi_FI/harri/medium/fi_FI-harri-medium.onnx", "Hei, miten voit tänään?"),
        ("en", "voice-en", "en/en_US/lessac/medium/en_US-lessac-medium.onnx", "Hello, how are you today?"),
    )
    for language, key, filename, text in specs:
        started = time.monotonic()
        voice = PiperVoice.load(ROOT / "models" / key / filename, use_cuda=False)
        path = directory / f"{language}.wav"
        with wave.open(str(path), "wb") as writer:
            # Fixed generator noise makes this a reproducible recognition fixture,
            # not a retry-until-green test of random TTS pronunciation.
            voice.synthesize_wav(text, writer, syn_config=SynthesisConfig(noise_scale=0, noise_w_scale=0))
        audio, rate = sf.read(path, dtype="float32")
        assert rate == 22050 and audio.ndim == 1
        assert len(audio) > rate and np.isfinite(audio).all()
        assert float(np.max(np.abs(audio))) > 0.001
        divisor = math.gcd(rate, 16000)
        normalized = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        # Give the decoder a short, explicit boundary around the real generated utterance.
        normalized = np.concatenate((np.zeros(3200, dtype=np.float32), normalized,
                                     np.zeros(8000, dtype=np.float32)))
        samples[language] = normalized
        record = {"operation": "tts", "model": key, "input": text,
                  "sample_rate": rate, "audio_seconds": round(len(audio) / rate, 3),
                  "seconds": round(time.monotonic() - started, 3)}
        evidence.append(record)
        print(json.dumps(record, ensure_ascii=False))
        del voice
        gc.collect()
    return samples


@pytest.mark.parametrize("language", ["fi", "en"])
def test_both_voice_models_generate_nonempty_audio(language, speech_samples):
    assert speech_samples[language].size > 16000


@pytest.fixture(scope="module", params=[
    ("whisper-small", "cuda", "int8_float16"),
    ("whisper-turbo", "cuda", "int8_float16"),
    ("whisper-small", "cpu", "int8"),
])
def asr_backend(request):
    from faster_whisper import WhisperModel

    key, device, precision = request.param
    model = WhisperModel(
        str(ROOT / "models" / key), device=device, compute_type=precision,
        cpu_threads=4, local_files_only=True,
    )
    yield key, device, precision, model
    del model
    gc.collect()


@pytest.mark.parametrize("language,anchors", [
    ("fi", {"hei", "voit", "tänään"}),
    ("en", {"hello", "how", "you", "today"}),
])
def test_whisper_recognizes_synthesized_speech(
    language, anchors, speech_samples, asr_backend, evidence,
):
    key, device, precision, model = asr_backend
    started = time.monotonic()
    segments, info = model.transcribe(
        speech_samples[language], language=language, beam_size=1,
        vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
        condition_on_previous_text=False,
    )
    transcript = " ".join(segment.text.strip() for segment in segments)
    record = {"operation": "asr", "model": key, "device": device,
              "precision": precision, "language": language, "output": transcript,
              "seconds": round(time.monotonic() - started, 3), "fixture": "Piper synthesized speech"}
    evidence.append(record)
    print(json.dumps(record, ensure_ascii=False))
    assert info.language == language
    words = set(re.findall(r"\w+", transcript.casefold()))
    assert anchors <= words, transcript
