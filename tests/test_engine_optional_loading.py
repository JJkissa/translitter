"""Optional local asset cache failure/retry; no neural models or physical I/O."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from translitter.core import Route
from translitter.engine import ModelEngine
from translitter.languages import TRANSLATOR_FILES, VOICE_FILES


EDGES = (("ru", "en"), ("en", "uk"))
BOUNDARIES = [(edge, stage) for edge in EDGES
              for stage in ("tokenizer", "model", "to", "eval")] + [("uk", "voice")]


@pytest.fixture
def optional(monkeypatch, tmp_path):
    calls, spoken = [], []
    state = SimpleNamespace(hook=lambda key, stage: None)
    engine = ModelEngine(tmp_path)
    engine._loaded = True
    engine._torch = SimpleNamespace(inference_mode=nullcontext)
    baseline = {("fi", "en"): (object(), object()), ("en", "fi"): (object(), object())}
    voices = {"fi": object(), "en": object()}
    engine._translations = baseline.copy()
    engine._voices = voices.copy()

    def record(key, stage, **kwargs):
        calls.append((key, stage, kwargs))
        state.hook(key, stage)

    class Tokenizer:
        def __init__(self, edge):
            self.edge = edge

        def __call__(self, texts, **kwargs):
            record(self.edge, "tokenize", texts=texts, **kwargs)
            return {"input_ids": [[1]]}

        def decode(self, tokens, **kwargs):
            return "English pivot" if self.edge == EDGES[0] else "Final translation"

    class Model:
        def __init__(self, edge):
            self.edge = edge

        def to(self, device):
            record(self.edge, "to", device=device)
            return self

        def eval(self):
            record(self.edge, "eval")
            return self

        def generate(self, **kwargs):
            record(self.edge, "generate", **kwargs)
            return [[1]]

    def edge_from(path):
        _, source, target = Path(path).name.split("-")
        return source, target

    def tokenizer(path, **kwargs):
        edge = edge_from(path)
        record(edge, "tokenizer", path=path, **kwargs)
        return Tokenizer(edge)

    def model(path, **kwargs):
        edge = edge_from(path)
        record(edge, "model", path=path, **kwargs)
        return Model(edge)

    def voice(path, **kwargs):
        record("uk", "voice", path=path, **kwargs)

        def synthesize(text):
            spoken.append((text, "uk"))
            yield SimpleNamespace(audio_float_array=np.array([.1, -.2], dtype=np.float32),
                                  sample_rate=22050)
        return SimpleNamespace(synthesize=synthesize)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        MarianTokenizer=SimpleNamespace(from_pretrained=tokenizer),
        MarianMTModel=SimpleNamespace(from_pretrained=model)))
    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(
        PiperVoice=SimpleNamespace(load=voice)))
    # The public preflight checks both conversation directions.
    for source, target in EDGES + (("en", "ru"), ("uk", "en")):
        directory = tmp_path / "models" / f"translate-{source}-{target}"
        directory.mkdir(parents=True)
        for name in TRANSLATOR_FILES:
            (directory / name).touch()
    for language in ("ru", "uk"):
        path = tmp_path / "models" / f"voice-{language}" / VOICE_FILES[language]
        path.parent.mkdir(parents=True)
        path.touch()
        Path(str(path) + ".json").touch()
    return engine, state, calls, spoken, baseline, voices


def turn(optional, cancelled=lambda: False, progress=lambda message: None):
    engine = optional[0]
    return engine.interpret_text("Original speech", Route(0, 1, "ru", "uk", "mic", "out"),
                                 3, cancelled, progress)


def recover(optional):
    engine, state, calls, spoken, baseline, voices = optional
    retained = engine._translations.copy()
    retained_voices = engine._voices.copy()
    state.hook = lambda key, stage: None
    calls.clear()
    result = turn(optional)
    assert result.generation == 3 and result.translation == "Final translation"
    assert spoken == [("Final translation", "uk")]
    np.testing.assert_array_equal(result.audio, np.array([.1, -.2], dtype=np.float32))
    assert set(engine._translations) == set(baseline) | set(EDGES)
    assert set(engine._voices) == set(voices) | {"uk"}
    for edge, pair in retained.items():
        assert engine._translations[edge] is pair
    for language, cached in retained_voices.items():
        assert engine._voices[language] is cached
    assert [(key, stage) for key, stage, _ in calls if stage == "tokenizer"] == [
        (edge, "tokenizer") for edge in EDGES if edge not in retained]
    for key, stage, kwargs in calls:
        if stage == "tokenizer":
            assert kwargs == dict(path=engine.root / "models" / f"translate-{key[0]}-{key[1]}",
                                  local_files_only=True)
        elif stage == "model":
            assert kwargs == dict(path=engine.root / "models" / f"translate-{key[0]}-{key[1]}",
                                  local_files_only=True, trust_remote_code=False, weights_only=True)
        elif stage == "to":
            assert kwargs == {"device": "cpu"}
        elif stage == "voice":
            assert kwargs == dict(path=engine.root / "models" / "voice-uk" / VOICE_FILES["uk"],
                                  use_cuda=False)
    calls.clear()
    assert turn(optional) is not None
    assert not any(stage in ("tokenizer", "model", "to", "eval", "voice")
                   for _, stage, _ in calls)


@pytest.mark.parametrize("key,stage", BOUNDARIES)
def test_optional_factory_exception_never_caches_partial_asset_and_retry_recovers(optional, key, stage):
    engine, state, calls, spoken, baseline, voices = optional

    def fail(current, boundary):
        if (current, boundary) == (key, stage):
            raise RuntimeError("Local factory interrupted")
    state.hook = fail
    with pytest.raises(RuntimeError, match="Local factory interrupted"):
        turn(optional)
    assert (calls[-1][0], calls[-1][1]) == (key, stage)
    assert spoken == []
    expected = set(baseline)
    if key == EDGES[1] or key == "uk":
        expected.add(EDGES[0])
    if key == "uk":
        expected.add(EDGES[1])
    assert set(engine._translations) == expected
    assert engine._voices == voices
    recover(optional)


@pytest.mark.parametrize("key,stage", BOUNDARIES)
def test_optional_factory_cancellation_blocks_inference_and_retry_reuses_only_ready_assets(optional, key, stage):
    engine, state, calls, spoken, baseline, voices = optional
    stopped = False

    def cancel(current, boundary):
        nonlocal stopped
        if (current, boundary) == (key, stage):
            stopped = True
    state.hook = cancel
    assert turn(optional, lambda: stopped) is None
    assert spoken == []
    assert not any(boundary in ("tokenize", "generate") for _, boundary, _ in calls)
    # A completed model remains safe to reuse even when the turn was cancelled.
    expected = set(baseline)
    if key == EDGES[1] or key == "uk":
        expected.add(EDGES[0])
    if key == "uk":
        expected.add(EDGES[1])
    elif stage != "tokenizer":
        expected.add(key)
    assert set(engine._translations) == expected
    assert set(engine._voices) == set(voices) | ({"uk"} if key == "uk" else set())
    recover(optional)


@pytest.mark.parametrize("message", ["Loading ru → en translation", "Loading en → uk translation", "Loading uk voice"])
@pytest.mark.parametrize("outcome", ["cancel", "exception"])
def test_optional_progress_interruption_does_not_construct_announced_asset(optional, message, outcome):
    _, _, calls, spoken, *_ = optional
    stopped = False

    def progress(current):
        nonlocal stopped
        if current == message:
            if outcome == "exception":
                raise RuntimeError("Progress interrupted")
            stopped = True
    if outcome == "exception":
        with pytest.raises(RuntimeError, match="Progress interrupted"):
            turn(optional, progress=progress)
    else:
        assert turn(optional, lambda: stopped, progress) is None
    key = {"Loading ru → en translation": EDGES[0],
           "Loading en → uk translation": EDGES[1], "Loading uk voice": "uk"}[message]
    assert not any(current == key for current, _, _ in calls)
    assert spoken == []
    recover(optional)
