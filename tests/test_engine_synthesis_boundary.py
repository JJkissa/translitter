"""Speech assembly contracts without model downloads or physical audio."""
from types import SimpleNamespace

import numpy as np
import pytest

from translitter.core import Route
from translitter.engine import ModelEngine


@pytest.fixture
def engine(tmp_path):
    engine = ModelEngine(tmp_path)
    engine._loaded = True  # Prepared fake models; initialization is tested separately.
    engine.translate_text = lambda *args: "Translated words"
    return engine


def install_voice(engine, language, synthesize):
    engine._voices = {language: SimpleNamespace(synthesize=synthesize)}


@pytest.mark.parametrize("source,target", [("fi", "en"), ("en", "fi")])
@pytest.mark.parametrize("final_mutation", [0.9, np.nan])
def test_reused_voice_buffer_preserves_each_validated_chunk(engine, source, target, final_mutation):
    first = np.array([0.1, -0.2], dtype=np.float32)
    second = np.array([0.3, -0.4], dtype=np.float32)

    def synthesize(text):
        buffer = first.copy()
        chunk = SimpleNamespace(audio_float_array=buffer, sample_rate=22050)
        yield chunk
        buffer[:] = second
        yield chunk
        buffer[:] = final_mutation

    install_voice(engine, target, synthesize)
    route = Route(0, 1, source, target, "mic", "out")
    result = engine.interpret_text("Original words", route, 7, prepared=True)
    assert result.generation == 7 and result.rate == 22050
    assert result.audio.dtype == np.float32
    np.testing.assert_array_equal(result.audio, np.concatenate([first, second]))
    assert np.isfinite(result.audio).all()


@pytest.mark.parametrize("invalid", [
    np.array([], dtype=np.float32), np.zeros((2, 2), dtype=np.float32),
    np.array([np.nan]), np.array([np.inf]), np.array([-np.inf]),
])
def test_invalid_later_chunk_rejects_entire_turn_and_recovers(engine, invalid):
    def synthesize(text):
        yield SimpleNamespace(audio_float_array=np.array([0.1]), sample_rate=22050)
        yield SimpleNamespace(audio_float_array=invalid, sample_rate=22050)

    install_voice(engine, "en", synthesize)
    route = Route(0, 1, "fi", "en", "mic", "out")
    with pytest.raises(ValueError, match="Voice produced invalid audio"):
        engine.interpret_text("Words", route, 1, prepared=True)
    install_voice(engine, "en", lambda text: iter([
        SimpleNamespace(audio_float_array=np.array([0.2]), sample_rate=22050)]))
    result = engine.interpret_text("Next words", route, 2, prepared=True)
    assert result.generation == 2
    np.testing.assert_array_equal(result.audio, np.array([0.2], dtype=np.float32))


@pytest.mark.parametrize("failure", ["empty", "rate-change", "exception"])
def test_incomplete_synthesis_never_returns_a_partial_result(engine, failure):
    def synthesize(text):
        if failure == "empty":
            return
        yield SimpleNamespace(audio_float_array=np.array([0.1]), sample_rate=22050)
        if failure == "exception":
            raise RuntimeError("Voice failed after first chunk")
        yield SimpleNamespace(audio_float_array=np.array([0.2]), sample_rate=16000)

    install_voice(engine, "en", synthesize)
    route = Route(0, 1, "fi", "en", "mic", "out")
    expected = RuntimeError if failure == "exception" else ValueError
    with pytest.raises(expected):
        engine.interpret_text("Words", route, 1, prepared=True)
    install_voice(engine, "en", lambda text: iter([
        SimpleNamespace(audio_float_array=np.array([0.3]), sample_rate=16000)]))
    result = engine.interpret_text("Next words", route, 2, prepared=True)
    assert result.rate == 16000
    np.testing.assert_array_equal(result.audio, np.array([0.3], dtype=np.float32))


@pytest.mark.parametrize("boundary", ["first", "second", "exhaustion"])
def test_cancellation_during_synthesis_discards_all_audio_and_recovers(engine, boundary):
    stopped = False
    advances = []

    def synthesize(text):
        nonlocal stopped
        for stage in ("first", "second"):
            advances.append(stage)
            if boundary == stage:
                stopped = True
            yield SimpleNamespace(audio_float_array=np.array([0.1]), sample_rate=22050)
        advances.append("exhaustion")
        stopped = True

    install_voice(engine, "en", synthesize)
    route = Route(0, 1, "fi", "en", "mic", "out")
    assert engine.interpret_text("Words", route, 1, lambda: stopped, prepared=True) is None
    assert advances == ["first", "second", "exhaustion"][:
        ("first", "second", "exhaustion").index(boundary) + 1]
    stopped = False
    install_voice(engine, "en", lambda text: iter([
        SimpleNamespace(audio_float_array=np.array([0.4]), sample_rate=22050)]))
    result = engine.interpret_text("Next words", route, 2, lambda: stopped, prepared=True)
    assert result.generation == 2
    np.testing.assert_array_equal(result.audio, np.array([0.4], dtype=np.float32))
