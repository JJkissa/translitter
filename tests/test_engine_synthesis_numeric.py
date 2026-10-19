"""Numeric validation at the real streaming speech assembly boundary."""
from types import SimpleNamespace

import numpy as np
import pytest

from translitter.core import Route
from translitter.engine import ModelEngine


@pytest.fixture
def engine(tmp_path):
    engine = ModelEngine(tmp_path)
    engine._loaded = True
    engine.translate_text = lambda *args: "Translated words"
    return engine


def install(engine, target, audio, rate=22050):
    def synthesize(text):
        yield SimpleNamespace(audio_float_array=np.array([0.1]), sample_rate=22050)
        yield SimpleNamespace(audio_float_array=audio, sample_rate=rate)
    engine._voices = {target: SimpleNamespace(synthesize=synthesize)}


def run(engine, speaker, generation=1):
    source, target = (("fi", "en"), ("en", "fi"))[speaker]
    return engine.interpret_text("Original words", Route(speaker, 1-speaker, source, target, "mic", "out"),
                                 generation, prepared=True)


def assert_recovery(engine, speaker):
    target = ("en", "fi")[speaker]
    install(engine, target, np.array([0.2, -0.3], dtype=np.float32))
    result = run(engine, speaker, 2)
    assert result.generation == 2 and result.rate == 22050
    np.testing.assert_array_equal(result.audio, np.array([0.1, 0.2, -0.3], dtype=np.float32))


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("imaginary", [0, 0.5])
def test_complex_chunk_rejects_whole_turn_without_lossy_cast(engine, speaker, dtype, imaginary):
    install(engine, ("en", "fi")[speaker], np.array([0.2 + imaginary * 1j], dtype=dtype))
    with pytest.raises(ValueError, match="Voice produced invalid audio"):
        run(engine, speaker)
    assert_recovery(engine, speaker)


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("policy", ["warn", "raise"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_float32_overflow_rejects_without_leaking_numpy_policy(engine, speaker, policy, sign):
    install(engine, ("en", "fi")[speaker], np.array([sign * np.finfo(np.float64).max]))
    with np.errstate(over=policy, invalid=policy):
        with pytest.raises(ValueError, match="Voice produced invalid audio"):
            run(engine, speaker)
        assert np.geterr()["over"] == policy
    assert_recovery(engine, speaker)


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("rate", [True, np.bool_(True), 22050.5, "22050", None, np.nan, np.inf])
def test_invalid_rate_is_not_coerced_into_valid_output(engine, speaker, rate):
    install(engine, ("en", "fi")[speaker], np.array([0.2]), rate)
    with pytest.raises(ValueError, match="Voice produced invalid audio"):
        run(engine, speaker)
    assert_recovery(engine, speaker)


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("audio", [
    np.array(["0.2"]), np.array([0.2], dtype=object),
    np.array([1], dtype="timedelta64[s]"), np.array(["2026-10-18"], dtype="datetime64[D]"),
])
def test_nonreal_domains_reject_without_coercion(engine, speaker, audio):
    install(engine, ("en", "fi")[speaker], audio)
    with pytest.raises(ValueError, match="Voice produced invalid audio"):
        run(engine, speaker)
    assert_recovery(engine, speaker)


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("rate", [8000, np.int32(22050), np.int64(48000)])
def test_valid_rate_bounds_and_numpy_integers(engine, speaker, rate):
    audio = np.array([0.2, -0.3], dtype=np.float32)
    engine._voices = {("en", "fi")[speaker]: SimpleNamespace(synthesize=lambda text: iter([
        SimpleNamespace(audio_float_array=audio, sample_rate=rate)]))}
    result = run(engine, speaker)
    assert type(result.rate) is int and result.rate == rate
    np.testing.assert_array_equal(result.audio, audio)


@pytest.mark.parametrize("kind", ["strided", "reversed", "endian", "readonly", "integer", "boolean", "list"])
def test_real_samples_preserve_values_and_owned_copy(engine, kind):
    audio = np.array([0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    if kind == "strided":
        audio = audio[::2]
    elif kind == "reversed":
        audio = audio[::-1]
    elif kind == "endian":
        audio = audio.astype(">f4")
    elif kind == "readonly":
        audio.setflags(write=False)
    elif kind == "integer":
        audio = np.array([-1, 0, 1], dtype=np.int16)
    elif kind == "boolean":
        audio = np.array([False, True])
    elif kind == "list":
        audio = [0.1, -0.2]
    expected = np.asarray(audio, dtype=np.float32).copy()
    install(engine, "en", audio)
    result = run(engine, 0)
    np.testing.assert_array_equal(result.audio, np.concatenate([np.array([0.1], dtype=np.float32), expected]))
    assert result.audio.dtype == np.float32 and result.audio.flags.owndata
