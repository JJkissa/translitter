"""All installed language pairs, opt-in real local models, no physical audio.

TRANSLITTER_MULTILINGUAL_TESTS=1 ./run-python.sh -m pytest tests/test_multilingual_models.py -q -s
"""
import hashlib
from itertools import permutations
import json
import math
import os
from pathlib import Path
import re

import pytest
from test_audio import explicit_audio

ROOT = Path(__file__).resolve().parents[1]
LANGUAGES = ("fi", "en", "ru", "uk")
PAIRS = tuple(permutations(LANGUAGES, 2))
KEYS = ("whisper-small", "whisper-turbo", "translate-fi-en", "translate-en-fi",
        "translate-ru-en", "translate-en-ru", "translate-uk-en", "translate-en-uk",
        "voice-fi", "voice-en", "voice-ru", "voice-uk")
TEXT = {"fi": "Haluan ostaa leipää tänään.", "en": "Today I want to buy bread.",
        "ru": "Сегодня я хочу купить хлеб.", "uk": "Сьогодні я хочу купити хліб."}
ANCHORS = {"fi": ("tänään", "leipää"), "en": ("today", "bread"),
           "ru": ("сегодня", "хлеб"), "uk": ("сьогодні", "хліб")}
pytestmark = pytest.mark.skipif(os.environ.get("TRANSLITTER_MULTILINGUAL_TESTS") != "1",
                              reason="Opt-in all-language real model integration")


@pytest.mark.parametrize("key", KEYS)
def test_complete_inventory_matches_local_and_publisher_digests(key):
    lock = json.loads((ROOT / "model-assets.lock.json").read_text())
    assert set(lock["assets"]) == set(KEYS)
    asset = lock["assets"][key]
    assert re.fullmatch(r"[0-9a-f]{40}", asset["revision"])
    for entry in asset["files"]:
        path = ROOT / asset["directory"] / entry["name"]
        assert path.stat().st_size == entry["bytes"]
        sha256, git_blob = hashlib.sha256(), hashlib.sha1()
        git_blob.update(f"blob {entry['bytes']}\0".encode())
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
                sha256.update(block)
                git_blob.update(block)
        assert sha256.hexdigest() == entry["sha256"]
        if entry["publisher_sha256"]:
            assert sha256.hexdigest() == entry["publisher_sha256"]
        elif entry.get("publisher_git_blob"):
            assert git_blob.hexdigest() == entry["publisher_git_blob"]


@pytest.fixture(scope="module")
def engine():
    from translitter.engine import ModelEngine
    assert os.environ.get("HF_HUB_OFFLINE") == "1"
    model = ModelEngine(ROOT)
    model.load()
    return model


@pytest.fixture(scope="module")
def evidence():
    records = []
    yield records
    (ROOT / ".local/multilingual-model-integration.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")


def assert_meaning(text, language):
    assert all(anchor in text.casefold() for anchor in ANCHORS[language]), text


@pytest.mark.parametrize("source,target", PAIRS)
def test_real_text_all_ordered_pairs(engine, evidence, source, target):
    from translitter.languages import missing_assets, translation_steps
    assert missing_assets(ROOT, source, target) == ()
    steps = translation_steps(source, target)
    assert engine._load_optional(steps, target, lambda message: None, lambda: False)
    translated = TEXT[source]
    for edge in steps:
        tokenizer, model = engine._translations[edge]
        with engine._torch.inference_mode():
            output = model.generate(**tokenizer([translated], return_tensors="pt", padding=True), max_new_tokens=96, num_beams=1)
        translated = tokenizer.decode(output[0], skip_special_tokens=True)
    record = {"operation": "text", "source": source, "target": target, "input": TEXT[source], "output": translated, "steps": steps}
    evidence.append(record)
    print(json.dumps(record, ensure_ascii=False))
    assert_meaning(translated, target)
    assert len(engine._translations) <= 6 and len(engine._voices) <= 4


@pytest.fixture(scope="module")
def samples(engine):
    import numpy as np
    from piper import SynthesisConfig
    from scipy.signal import resample_poly
    fixtures = {}
    for language in LANGUAGES:
        assert engine._load_optional((), language, lambda message: None, lambda: False)
        chunks = list(engine._voices[language].synthesize(TEXT[language], syn_config=SynthesisConfig(noise_scale=0, noise_w_scale=0)))
        assert chunks
        rate = chunks[0].sample_rate
        assert all(c.sample_rate == rate for c in chunks)
        audio = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
        assert len(audio) > rate and np.isfinite(audio).all() and float(np.max(np.abs(audio))) > 0.001
        divisor = math.gcd(rate, 16000)
        audio = resample_poly(audio, 16000 // divisor, rate // divisor).astype(np.float32)
        audio = np.pad(audio, (3200, 8000))
        assert audio.ndim == 1 and 0 < len(audio) <= 128000
        fixtures[language] = audio
    return fixtures


@pytest.mark.parametrize("source,target", PAIRS)
def test_real_speech_all_ordered_pairs_same_engine(engine, samples, evidence, source, target):
    import numpy as np
    from translitter.core import Route
    route = Route(0, 1, source, target, "generated-input", "no-hardware-output")
    result = engine.interpret(samples[source], route, 1)
    assert result is not None
    record = {"operation": "speech", "source": source, "target": target, "original": result.original,
              "translation": result.translation, "rate": result.rate, "samples": len(result.audio),
              "seconds": result.elapsed, "fixture": "deterministic Piper, no physical audio"}
    evidence.append(record)
    print(json.dumps(record, ensure_ascii=False))
    assert (result.source_language, result.target_language) == (source, target)
    assert_meaning(result.original, source)
    assert_meaning(result.translation, target)
    assert result.audio.dtype == np.float32 and result.audio.ndim == 1
    assert np.isfinite(result.audio).all() and len(result.audio) > result.rate
    assert len(engine._translations) <= 6 and len(engine._voices) <= 4


def test_native_worker_switches_all_pairs_and_dispatches_target_audio(explicit_audio, qtbot, samples, evidence):
    """One real worker/model session across all pairs; only physical Qt endpoints fake."""
    from translitter.ui import Window
    from translitter.worker import ModelWorker
    hub, pairs, opened = explicit_audio
    worker = ModelWorker(ROOT)
    window = Window(ROOT, worker=worker, audio=hub, start_worker=True)
    qtbot.addWidget(window)
    errors, results = [], []
    worker.error.connect(lambda generation, message: errors.append((generation, message)))
    worker.result.connect(results.append)
    try:
        qtbot.waitUntil(lambda: window.models_ready or bool(errors), timeout=45000)
        assert window.models_ready and not errors, errors
        for selector, pair in zip(window.selectors, pairs):
            selector.addItem(pair.label, pair)
            selector.setCurrentIndex(0)
        for index, (source, target) in enumerate(PAIRS):
            speaker = index % 2
            languages = (source, target) if speaker == 0 else (target, source)
            for selector, language in zip(window.language_selectors, languages):
                selector.setCurrentIndex(selector.findData(language))
            window.start_session()
            assert window.active and hub.running, window.status.text()
            card = window.cards[speaker]
            original_before = card.original.toPlainText()
            offset = len(opened)
            window._press(speaker)
            assert window._held[speaker]
            hub.samples.emit(speaker, samples[source])
            window._release(speaker)
            qtbot.waitUntil(lambda: card.original.toPlainText() != original_before or bool(errors), timeout=30000)
            assert not errors, errors
            qtbot.waitUntil(lambda: hub.played_bytes[1 - speaker] > 0, timeout=5000)
            assert_meaning(card.original.toPlainText(), source)
            assert_meaning(card.translated.toPlainText(), target)
            expected_output = pairs[1 - speaker].output_id
            assert [identifier for kind, identifier in opened[offset:] if kind == "playback"] == [expected_output]
            qtbot.waitUntil(lambda: len(results) > index or bool(errors), timeout=5000)
            assert not errors, errors
            result = results[index]
            assert result.generation == window.generation
            assert (result.source_language, result.target_language, result.speaker, result.listener) == (source, target, speaker, 1 - speaker)
            evidence.append({"operation": "native_worker_pair_switch", "source": source, "target": target,
                             "speaker": speaker, "listener": 1 - speaker, "original": result.original,
                             "translation": result.translation, "output_id": expected_output,
                             "fixture": "real worker/models and AudioHub; generated PTT input, fake physical endpoints"})
            window.stop_session()
            assert not hub.running and not hub.sources and not hub.sinks
        assert len(results) == len(PAIRS)
    finally:
        window.close()
        qtbot.waitUntil(lambda: not worker.isRunning(), timeout=45000)
        window.close()
