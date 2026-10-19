"""Complete-turn ASR boundaries with fake segment iterators, never device I/O."""
from types import SimpleNamespace

import numpy as np
import pytest

from translitter.core import Route
from translitter.engine import ModelEngine


@pytest.fixture
def asr(tmp_path):
    engine = ModelEngine(tmp_path)
    engine._loaded = True
    engine._load_optional = lambda *args: True
    downstream = []
    engine.translate_text = lambda text, *args: downstream.append(("translate", text)) or "Hello"
    engine._synthesize = lambda *args: downstream.append(("synthesize", args[0])) or (np.ones(2, dtype=np.float32), 16000)
    route = Route(0, 1, "fi", "en", "mic", "out")
    return engine, route, downstream


def install_segments(engine, factory):
    engine._whisper = SimpleNamespace(transcribe=lambda *args, **kwargs: (factory(), None))


def recover(asr):
    engine, route, downstream = asr
    install_segments(engine, lambda: iter([SimpleNamespace(text=" Hei"), SimpleNamespace(text=" maailma ")]))
    result = engine.interpret(np.ones(160, dtype=np.float32), route, 2)
    assert result.original == "Hei maailma"
    assert result.generation == 2
    assert downstream == [("translate", "Hei maailma"), ("synthesize", "Hello")]


@pytest.mark.parametrize("invalid", ["\x1cHei", "Hei\x1d", "\x1e", "\x1fHei\x1f"])
def test_segment_edge_controls_cannot_be_stripped_into_valid_speech(asr, invalid):
    engine, route, downstream = asr
    install_segments(engine, lambda: iter([SimpleNamespace(text="Valid prefix"), SimpleNamespace(text=invalid)]))
    assert engine.interpret(np.ones(160, dtype=np.float32), route, 1) is None
    assert downstream == []
    recover(asr)


@pytest.mark.parametrize("boundary", ["transcribe", "first", "second", "exhaustion"])
def test_cancelled_recognition_never_publishes_partial_transcript(asr, boundary):
    engine, route, downstream = asr
    stopped = False
    advances = []

    def segments():
        nonlocal stopped
        for stage in ("first", "second"):
            advances.append(stage)
            if boundary == stage:
                stopped = True
            yield SimpleNamespace(text="Hei")
        advances.append("exhaustion")
        stopped = True

    def transcribe(*args, **kwargs):
        nonlocal stopped
        if boundary == "transcribe":
            stopped = True
        return segments(), None

    engine._whisper = SimpleNamespace(transcribe=transcribe)
    assert engine.interpret(np.ones(160, dtype=np.float32), route, 1, lambda: stopped) is None
    expected = [] if boundary == "transcribe" else ["first", "second", "exhaustion"][:
        ("first", "second", "exhaustion").index(boundary) + 1]
    assert advances == expected
    assert downstream == []
    recover(asr)


@pytest.mark.parametrize("invalid", ["\ufffd", "Hei\x00", "Hei\x0b", "Hei\x0c"])
def test_garbled_later_segment_rejects_entire_transcript(asr, invalid):
    engine, route, downstream = asr
    install_segments(engine, lambda: iter([SimpleNamespace(text="Valid prefix"), SimpleNamespace(text=invalid)]))
    assert engine.interpret(np.ones(160, dtype=np.float32), route, 1) is None
    assert downstream == []
    recover(asr)


@pytest.mark.parametrize("failure", ["exception", "missing-text", "nonstring-text"])
def test_failed_segment_iteration_never_translates_prefix_and_recovers(asr, failure):
    engine, route, downstream = asr

    def segments():
        yield SimpleNamespace(text="Valid prefix")
        if failure == "exception":
            raise RuntimeError("ASR failure after partial text")
        yield SimpleNamespace() if failure == "missing-text" else SimpleNamespace(text=None)

    install_segments(engine, segments)
    with pytest.raises({"exception": RuntimeError, "missing-text": AttributeError, "nonstring-text": TypeError}[failure]):
        engine.interpret(np.ones(160, dtype=np.float32), route, 1)
    assert downstream == []
    recover(asr)


@pytest.mark.parametrize("parts,expected", [
    ([], ""), ([" ", "\t\r\n"], ""), (["<pad>", "!!!"], ""),
    ([" \tHei\n", " maailma\r "], "Hei maailma"),
    (["  Hei", "", "maailma  "], "Hei  maailma"),
])
def test_empty_or_valid_transcript_normalization(asr, parts, expected):
    engine, route, downstream = asr
    install_segments(engine, lambda: iter(SimpleNamespace(text=text) for text in parts))
    result = engine.interpret(np.ones(160, dtype=np.float32), route, 1)
    if expected:
        assert result.original == expected
        assert downstream == [("translate", expected), ("synthesize", "Hello")]
    else:
        assert result is None
        assert downstream == []
