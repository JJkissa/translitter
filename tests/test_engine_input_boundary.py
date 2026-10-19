"""Direct engine numeric admission; fake inference, no audio devices/models."""
from types import SimpleNamespace
import warnings

import numpy as np
import pytest

from translitter.core import Route
from translitter.engine import ModelEngine


@pytest.fixture
def harness(tmp_path):
    engine = ModelEngine(tmp_path)
    engine._loaded = True
    events = []
    received = []
    engine._load_optional = lambda *args: events.append("prepare") or True

    def transcribe(samples, **options):
        events.append("recognize")
        received.append(samples.copy())
        return iter([SimpleNamespace(text="Hei")]), None

    engine._whisper = SimpleNamespace(transcribe=transcribe)
    engine.translate_text = lambda *args: events.append("translate") or "Hello"
    engine._synthesize = lambda *args: events.append("synthesize") or (np.ones(2, np.float32), 16000)
    return engine, events, received


def invoke(engine, method, samples, speaker=0, cancelled=lambda: False):
    route = Route(speaker, 1 - speaker, "fi", "en", "mic", "out")
    if method == "interpret":
        return engine.interpret(samples, route, 3, cancelled)
    return engine.recognize(samples, route, cancelled)


@pytest.mark.parametrize("method", ["interpret", "recognize"])
@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("imaginary", [0, 1])
def test_complex_input_rejected_before_inference_and_recovers(harness, method, speaker, dtype, imaginary):
    engine, events, received = harness
    samples = np.full(160, 0.25 + imaginary * 1j, dtype=dtype)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        with pytest.raises(ValueError, match="finite mono"):
            invoke(engine, method, samples, speaker)
    assert events == []
    assert received == []
    good = np.linspace(-0.5, 0.5, 160, dtype=np.float32)
    result = invoke(engine, method, good, speaker)
    assert (result.original if method == "interpret" else result) == "Hei"
    np.testing.assert_array_equal(received, [good])


@pytest.mark.parametrize("method", ["interpret", "recognize"])
@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("policy", ["warn", "raise"])
@pytest.mark.parametrize("sign", [-1, 1])
def test_float32_overflow_is_sanitized_and_recovers(harness, method, speaker, policy, sign):
    engine, events, received = harness
    samples = np.full(160, sign * np.finfo(np.float64).max)
    with warnings.catch_warnings(), np.errstate(over=policy, invalid=policy):
        warnings.simplefilter("error")
        before = np.geterr().copy()
        with pytest.raises(ValueError, match="finite mono"):
            invoke(engine, method, samples, speaker)
        assert np.geterr() == before
    assert events == []
    assert received == []
    good = np.ones(160, dtype=np.float32)
    result = invoke(engine, method, good, speaker)
    assert (result.original if method == "interpret" else result) == "Hei"
    np.testing.assert_array_equal(received, [good])


@pytest.mark.parametrize("method", ["interpret", "recognize"])
@pytest.mark.parametrize("samples", [
    np.array(["0.25"]), np.array([0.25], dtype=object),
    np.array(["2026-01-01"], dtype="datetime64[D]"),
    np.array([1], dtype="timedelta64[s]"), np.zeros(1, dtype=[("x", "f4")]),
    np.array([np.nan]), np.array([np.inf]), np.array([]),
    np.array(0.25), np.zeros((2, 2)),
])
def test_invalid_domains_and_shapes_never_reach_models(harness, method, samples):
    engine, events, received = harness
    with pytest.raises(ValueError, match="finite mono"):
        invoke(engine, method, samples)
    assert events == received == []


@pytest.mark.parametrize("method", ["interpret", "recognize"])
@pytest.mark.parametrize("kind", ["strided", "reversed", "endian", "readonly", "integer", "boolean", "list"])
def test_valid_real_inputs_preserve_exact_samples(harness, method, kind):
    engine, events, received = harness
    values = np.linspace(-0.5, 0.5, 320, dtype=np.float32)
    samples = {
        "strided": values[::2], "reversed": values[::-1],
        "endian": values.astype(">f4"), "readonly": values.copy(),
        "integer": np.array([-1, 0, 1]), "boolean": np.array([True, False]),
        "list": [0.25, -0.25],
    }[kind]
    if kind == "readonly":
        samples.flags.writeable = False
    original = np.array(samples, copy=True)
    result = invoke(engine, method, samples)
    assert (result.original if method == "interpret" else result) == "Hei"
    np.testing.assert_array_equal(received, [original.astype(np.float32)])
    np.testing.assert_array_equal(samples, original)


@pytest.mark.parametrize("method", ["interpret", "recognize"])
def test_cancelled_and_unloaded_calls_keep_precedence(harness, method):
    engine, events, received = harness
    engine._loaded = False
    invalid = np.array([1j])
    assert invoke(engine, method, invalid, cancelled=lambda: True) is None
    with pytest.raises(RuntimeError, match="Load models"):
        invoke(engine, method, invalid)
    assert events == received == []
