"""Baseline initialization failure/retry using local fake model factories."""
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from translitter.core import Route
from translitter.engine import ModelEngine
from translitter.languages import VOICE_FILES


EDGES = (("fi", "en"), ("en", "fi"))
BOUNDARIES = [("torch", "threads"), ("whisper", "construct")] + [
    (edge, stage) for edge in EDGES for stage in ("tokenizer", "model", "to", "eval")
] + [(language, "voice") for language in ("fi", "en")]
MESSAGES = ["Loading speech recognition", "Loading fi → en translation",
            "Loading en → fi translation", "Loading fi voice", "Loading en voice"]


@pytest.fixture
def baseline(monkeypatch, tmp_path):
    calls, objects = [], []
    state = SimpleNamespace(hook=lambda key, stage: None)

    def record(key, stage, **kwargs):
        calls.append((key, stage, kwargs))
        state.hook(key, stage)

    def owned(value):
        objects.append(value)
        return value

    class Tokenizer:
        def __init__(self, edge):
            self.edge = edge

        def __call__(self, texts, **kwargs):
            record(self.edge, "tokenize")
            return {"input_ids": [[1]]}

        def decode(self, *args, **kwargs):
            return "Translated speech"

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
            record(self.edge, "generate")
            return [[1]]

    def edge_from(path):
        return tuple(Path(path).name.split("-")[1:])

    def tokenizer(path, **kwargs):
        edge = edge_from(path)
        record(edge, "tokenizer", path=path, **kwargs)
        return owned(Tokenizer(edge))

    def model(path, **kwargs):
        edge = edge_from(path)
        record(edge, "model", path=path, **kwargs)
        return owned(Model(edge))

    def whisper(path, **kwargs):
        record("whisper", "construct", path=path, **kwargs)

        def transcribe(samples, **options):
            record("whisper", "transcribe")
            return iter([SimpleNamespace(text="Original speech")]), None
        return owned(SimpleNamespace(transcribe=transcribe))

    def voice(path, **kwargs):
        language = "fi" if Path(path).name.startswith("fi_") else "en"
        record(language, "voice", path=path, **kwargs)

        def synthesize(text):
            record(language, "synthesize")
            yield SimpleNamespace(audio_float_array=np.array([.1, -.2], dtype=np.float32),
                                  sample_rate=22050)
        return owned(SimpleNamespace(synthesize=synthesize))

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(
        set_num_threads=lambda count: record("torch", "threads", count=count),
        inference_mode=nullcontext))
    monkeypatch.setitem(sys.modules, "faster_whisper", SimpleNamespace(WhisperModel=whisper))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        MarianTokenizer=SimpleNamespace(from_pretrained=tokenizer),
        MarianMTModel=SimpleNamespace(from_pretrained=model)))
    monkeypatch.setitem(sys.modules, "piper", SimpleNamespace(PiperVoice=SimpleNamespace(load=voice)))
    for name in ("whisper-turbo", "whisper-small", "translate-fi-en", "translate-en-fi"):
        (tmp_path / "models" / name).mkdir(parents=True)
    for language in ("fi", "en"):
        path = tmp_path / "models" / f"voice-{language}" / VOICE_FILES[language]
        path.parent.mkdir(parents=True)
        path.touch()
        Path(str(path) + ".json").touch()
    return ModelEngine(tmp_path), state, calls, objects


def recover(baseline):
    engine, state, calls, objects = baseline
    previous = objects.copy()
    state.hook = lambda key, stage: None
    calls.clear()
    engine.load()
    assert engine._loaded
    assert [(key, stage) for key, stage, _ in calls] == BOUNDARIES
    current = [engine._whisper, *engine._voices.values(),
               *(obj for pair in engine._translations.values() for obj in pair)]
    assert not any(new is old for new in current for old in previous)
    assert set(engine._translations) == set(EDGES)
    assert set(engine._voices) == {"fi", "en"}
    for key, stage, options in calls:
        if stage == "construct":
            assert options == dict(path=str(engine.root / "models" / "whisper-turbo"),
                                   device="cuda", compute_type="int8_float16",
                                   cpu_threads=4, local_files_only=True)
        elif stage in ("model", "tokenizer"):
            expected = dict(path=engine.root / "models" / f"translate-{key[0]}-{key[1]}",
                            local_files_only=True)
            if stage == "model":
                expected.update(trust_remote_code=False, weights_only=True)
            assert options == expected
        elif stage == "voice":
            assert options == dict(path=engine.root / "models" / f"voice-{key}" / VOICE_FILES[key],
                                   use_cuda=False)
    calls.clear()
    engine.load(lambda message: pytest.fail("Successful load must be idempotent"))
    assert not calls
    for speaker, (source, target) in enumerate(EDGES):
        route = Route(speaker, 1 - speaker, source, target, "mic", "out")
        result = engine.interpret(np.ones(160, dtype=np.float32), route, 5)
        assert result.original == "Original speech" and result.translation == "Translated speech"
        assert result.target_language == target and result.generation == 5
        np.testing.assert_array_equal(result.audio, np.array([.1, -.2], dtype=np.float32))


@pytest.mark.parametrize("key,stage", BOUNDARIES)
def test_baseline_factory_failure_blocks_audio_and_retries_cleanly(baseline, key, stage):
    engine, state, calls, _ = baseline

    def fail(current, boundary):
        if (current, boundary) == (key, stage):
            raise RuntimeError("Factory interrupted")
    state.hook = fail
    with pytest.raises(RuntimeError, match="Factory interrupted"):
        engine.load()
    assert not engine._loaded
    assert (calls[-1][0], calls[-1][1]) == (key, stage)
    with pytest.raises(RuntimeError, match="Load models before interpreting"):
        engine.interpret(np.ones(160, dtype=np.float32), Route(0, 1, "fi", "en", "mic", "out"), 1)
    recover(baseline)


@pytest.mark.parametrize("message", MESSAGES)
def test_baseline_progress_failure_retries_without_publishing_readiness(baseline, message):
    engine, _, calls, _ = baseline

    def progress(current):
        if current == message:
            raise RuntimeError("Progress interrupted")
    with pytest.raises(RuntimeError, match="Progress interrupted"):
        engine.load(progress)
    assert not engine._loaded
    assert not any(stage in ("transcribe", "generate", "synthesize") for _, stage, _ in calls)
    recover(baseline)


@pytest.mark.parametrize("prepared", [False, True])
def test_failed_baseline_load_blocks_text_turn_before_inference(baseline, prepared):
    engine, state, calls, _ = baseline

    def fail(key, stage):
        if (key, stage) == ("en", "voice"):
            raise RuntimeError("Last voice interrupted")
    state.hook = fail
    with pytest.raises(RuntimeError, match="Last voice interrupted"):
        engine.load()
    calls.clear()
    # The fi voice and both translators exist, but initialization is incomplete.
    with pytest.raises(RuntimeError, match="Load models before interpreting"):
        engine.interpret_text("Original speech", Route(1, 0, "en", "fi", "mic", "out"),
                              1, prepared=prepared)
    assert not calls
    recover(baseline)


@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize("cancelled", [False, True])
def test_fresh_engine_text_readiness_and_precancellation(baseline, prepared, cancelled):
    engine, _, calls, _ = baseline
    route = Route(0, 1, "fi", "en", "mic", "out")
    if cancelled:
        assert engine.interpret_text("Original speech", route, 1, lambda: True,
                                     prepared=prepared) is None
    else:
        with pytest.raises(RuntimeError, match="Load models before interpreting"):
            engine.interpret_text("Original speech", route, 1, prepared=prepared)
    assert not calls
