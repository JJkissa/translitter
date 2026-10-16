"""Opt-in real Russian assets and speech pipelines; never hardware audio I/O.

TRANSLITTER_RUSSIAN_TESTS=1 ./run-python.sh -m pytest tests/test_russian_models.py -q -s
"""
import hashlib
import json
import math
import os
from pathlib import Path
import re

import pytest

ROOT = Path(__file__).resolve().parents[1]
KEYS = ("translate-ru-en", "translate-en-ru", "voice-ru")
PAIRS = (("ru", "en"), ("en", "ru"), ("ru", "fi"), ("fi", "ru"))
TEXT = {"ru": "Сегодня я хочу купить хлеб.", "en": "Today I want to buy bread.", "fi": "Haluan ostaa leipää tänään."}
ANCHORS = {"ru": {"сегодня", "хлеб"}, "en": {"today", "bread"}, "fi": {"tänään", "leipää"}}
pytestmark = pytest.mark.skipif(os.environ.get("TRANSLITTER_RUSSIAN_TESTS") != "1",
                              reason="Opt-in installed Russian model integration")


@pytest.mark.parametrize("key", KEYS)
def test_russian_files_match_pinned_publisher_checksums(key):
    manifest = json.loads((ROOT / "model-assets.lock.json").read_text())
    assert set(KEYS) <= set(manifest["assets"])
    asset = manifest["assets"][key]
    assert re.fullmatch(r"[0-9a-f]{40}", asset["revision"])
    for entry in asset["files"]:
        path = ROOT / asset["directory"] / entry["name"]
        assert path.stat().st_size == entry["bytes"]
        sha256 = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                sha256.update(block)
        assert sha256.hexdigest() == entry["sha256"]
        if entry["publisher_sha256"]:
            assert sha256.hexdigest() == entry["publisher_sha256"]


@pytest.mark.parametrize("source,target", PAIRS)
def test_installed_pairs_enable_native_start_without_real_device_io(qtbot, source, target):
    from test_ui import FakeAudio, FakeWorker
    from translitter.ui import Window
    audio, worker = FakeAudio(), FakeWorker()
    window = Window(ROOT, worker=worker, audio=audio, start_worker=False)
    qtbot.addWidget(window)
    for selector, language in zip(window.language_selectors, (source, target)):
        selector.setCurrentIndex(selector.findData(language))
    worker.ready.emit()
    assert window.start_button.isEnabled(), window.language_status.text()
    assert "not installed" not in window.language_status.text()
    window.start_session()
    assert window.active
    assert [(r.source_language, r.target_language) for r in worker.sessions[-1][1]] == [(source, target), (target, source)]
    window.stop_session()
    window.close()


@pytest.fixture(scope="module")
def engine():
    from translitter.engine import ModelEngine
    assert os.environ["HF_HUB_OFFLINE"] == "1"
    model = ModelEngine(ROOT)
    model.load()
    return model


@pytest.fixture(scope="module")
def evidence():
    records = []
    yield records
    (ROOT / ".local/russian-model-integration.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


@pytest.mark.parametrize("source,target", PAIRS)
def test_real_russian_text_translation_both_pairs(engine, source, target, evidence):
    from translitter.languages import missing_assets, translation_steps
    assert missing_assets(ROOT, source, target) == ()
    steps = translation_steps(source, target)
    assert engine._load_optional(steps, target, lambda message: None, lambda: False)
    result = TEXT[source]
    for edge in steps:
        tokenizer, model = engine._translations[edge]
        inputs = tokenizer([result], return_tensors="pt", padding=True)
        with engine._torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=96, num_beams=1)
        result = tokenizer.decode(output[0], skip_special_tokens=True)
    record = {"operation": "text", "source": source, "target": target, "input": TEXT[source], "output": result, "steps": steps}
    evidence.append(record)
    print(json.dumps(record, ensure_ascii=False))
    assert all(word in result.casefold() for word in ANCHORS[target]), result


@pytest.mark.xfail(strict=True, reason="Pinned EN→RU model omits 'today' in a conversational greeting, with both greedy and 4-beam decoding")
def test_known_greeting_temporal_omission_remains_visible(engine, evidence):
    assert engine._load_optional((("en", "ru"),), "ru", lambda message: None, lambda: False)
    tokenizer, model = engine._translations["en", "ru"]
    text = "Hello, how are you today?"
    with engine._torch.inference_mode():
        output = model.generate(**tokenizer([text], return_tensors="pt"), max_new_tokens=96, num_beams=1)
    translation = tokenizer.decode(output[0], skip_special_tokens=True)
    evidence.append({"operation": "known_quality_limitation", "input": text, "output": translation,
                     "limitation": "Temporal detail omitted; not a model-loading failure"})
    assert "сегодня" in translation.casefold(), translation


@pytest.fixture(scope="module")
def samples(engine):
    import numpy as np
    from piper import SynthesisConfig
    from scipy.signal import resample_poly
    inputs = {}
    for language, text in TEXT.items():
        assert engine._load_optional((), language, lambda message: None, lambda: False)
        chunks = list(engine._voices[language].synthesize(text, syn_config=SynthesisConfig(noise_scale=0, noise_w_scale=0)))
        assert chunks
        rate = chunks[0].sample_rate
        assert all(chunk.sample_rate == rate for chunk in chunks)
        audio = np.concatenate([chunk.audio_float_array for chunk in chunks]).astype(np.float32)
        assert audio.ndim == 1 and len(audio) > rate and np.isfinite(audio).all()
        assert float(np.max(np.abs(audio))) > 0.001
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        audio = np.pad(audio, (3200, 8000))
        assert len(audio) <= 128000
        inputs[language] = audio
    return inputs


@pytest.mark.parametrize("source,target", PAIRS)
def test_real_russian_speech_pipeline_both_pairs(engine, samples, source, target, evidence):
    import numpy as np
    from translitter.core import Route
    route = Route(0, 1, source, target, "fixture-mic", "fixture-output")
    result = engine.interpret(samples[source], route, 1)
    assert result is not None
    assert (result.source_language, result.target_language) == (source, target)
    assert (result.speaker, result.listener, result.generation) == (0, 1, 1)
    record = {"operation": "speech", "fixture": "deterministic Piper; no microphone capture or playback",
              "source": source, "target": target, "original": result.original,
              "translation": result.translation, "rate": result.rate, "samples": len(result.audio), "seconds": result.elapsed}
    evidence.append(record)
    print(json.dumps(record, ensure_ascii=False))
    assert all(word in result.original.casefold() for word in ANCHORS[source]), result.original
    assert all(word in result.translation.casefold() for word in ANCHORS[target]), result.translation
    assert result.audio.dtype == np.float32 and result.audio.ndim == 1
    assert np.isfinite(result.audio).all() and len(result.audio) > result.rate
