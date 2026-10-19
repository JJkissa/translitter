"""PCM conversion and playback rejection; fake endpoints only."""
import math
import numpy as np
import pytest
from scipy.signal import resample_poly
from PySide6.QtMultimedia import QtAudio

from test_audio import explicit_audio
from translitter.audio import prepare_pcm
from translitter.core import make_routes


INVALID_PCM = [
    pytest.param([0.25, np.nan], 16000, id="nan"),
    pytest.param([np.inf], 22050, id="infinity"),
    pytest.param([[0.25]], 16000, id="stereo"),
    pytest.param(0.25, 16000, id="scalar"),
    pytest.param(["private-malformed-sample"], 16000, id="malformed"),
    pytest.param([0.25], 0, id="zero-rate"),
    pytest.param([0.25], -16000, id="negative-rate"),
    pytest.param([0.25], 16000.5, id="fractional-rate"),
    pytest.param([0.25], True, id="boolean-rate"),
    pytest.param(np.array([0.25 + 0.5j]), 16000, id="complex-samples"),
    pytest.param(np.array([0.25 + 0j]), 22050, id="complex-zero-imaginary"),
]


@pytest.mark.parametrize("listener", [0, 1])
@pytest.mark.parametrize("already_playing", [False, True])
@pytest.mark.parametrize("samples,rate", INVALID_PCM)
def test_invalid_pcm_rejected_without_disturbing_playback(
        explicit_audio, listener, already_playing, samples, rate):
    hub, pairs, opened = explicit_audio
    hub.start(make_routes(*pairs))
    events, failures = [], []
    hub.playing.connect(lambda *args: events.append(args))
    hub.failure.connect(lambda *args: failures.append(args))
    if already_playing:
        assert hub.play(listener, np.full(80, 0.25), 16000)
        assert hub.play(listener, np.full(160, -0.25), 16000)
    before = (dict(hub.buffers), {n: list(q) for n, q in hub.queues.items()},
              list(opened), list(hub.played_bytes), list(events))

    assert hub.play(listener, samples, rate) is False

    assert (hub.buffers, {n: list(q) for n, q in hub.queues.items()},
            opened, hub.played_bytes, events) == before
    assert not failures  # Invalid output must not cancel a healthy conversation.
    assert hub.running
    assert hub.play(listener, np.full(240, 0.5), 16000)
    expected = [np.full(160, -0.25), np.full(240, 0.5)] if already_playing else [np.full(240, 0.5)]
    if already_playing:
        hub._output_state(listener, QtAudio.State.IdleState)
    for values in expected:
        assert bytes(hub.buffers[listener].data()) == prepare_pcm(values, 16000)
        hub._output_state(listener, QtAudio.State.IdleState)
    assert listener not in hub.buffers and not hub.queues[listener]
    assert events[-1] == (listener, False)
    expected_samples = 80 + 160 + 240 if already_playing else 240
    assert hub.played_bytes[listener] == expected_samples * 2


@pytest.mark.parametrize("samples,rate", INVALID_PCM)
def test_conversion_rejects_invalid_values(samples, rate):
    with pytest.raises((TypeError, ValueError, OverflowError)):
        prepare_pcm(samples, rate)


@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed", "big-endian"])
def test_conversion_preserves_order_clips_and_serializes_little_endian(layout):
    values = np.array([-2, -1, -0.5, 0, 0.5, 1, 2], dtype=np.float32)
    if layout == "strided":
        storage = np.zeros(len(values) * 2, dtype=np.float32)
        storage[::2] = values
        values = storage[::2]
    elif layout == "reversed":
        values = values[::-1]
    elif layout == "big-endian":
        values = values.astype(">f4")
    before = values.copy()
    expected = np.array([-32767, -32767, -16383, 0, 16383, 32767, 32767], dtype="<i2")
    if layout == "reversed":
        expected = expected[::-1]
    assert prepare_pcm(values, 16000) == expected.tobytes()
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("rate", [8000, 22050, 24000, 44100, 48000])
def test_resampling_preserves_duration_and_interior_amplitude(rate):
    values = np.full(rate // 10, 0.25, dtype=np.float32)
    before = values.copy()
    pcm = np.frombuffer(prepare_pcm(values, rate), dtype="<i2")
    assert len(pcm) == 1600
    np.testing.assert_allclose(pcm[40:-40], 0.25 * 32767, atol=10, rtol=0)
    assert np.max(np.abs(pcm.astype(np.int32))) < 10000
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("rate", [8000, 22050, 24000, 44100, 48000])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.filterwarnings("error")
def test_finite_extreme_resampling_does_not_overflow(rate, sign):
    values = np.full(rate // 10, sign * np.finfo(np.float32).max, dtype=np.float32)
    before = values.copy()
    pcm = np.frombuffer(prepare_pcm(values, rate), dtype="<i2")
    assert len(pcm) == 1600
    np.testing.assert_array_equal(pcm[40:-40], np.full(1520, sign * 32767))
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize("rate", [8000, 22050, 24000, 44100, 48000])
@pytest.mark.parametrize("pattern", ["step", "impulse", "random"])
@pytest.mark.filterwarnings("error")
def test_extreme_resampling_matches_wide_precision_reference(rate, pattern):
    count = rate // 10
    if pattern == "step":
        values = np.ones(count, dtype=np.float32)
        values[:count // 2] = -1
    elif pattern == "impulse":
        values = np.zeros(count, dtype=np.float32)
        values[count // 2] = 1
    else:
        values = np.random.default_rng(42).choice([-1, 1], count).astype(np.float32)
    values *= np.finfo(np.float32).max
    before = values.copy()
    divisor = math.gcd(rate, 16000)
    reference = resample_poly(values.astype(np.float64), 16000 // divisor, rate // divisor)
    expected = (np.clip(reference, -1, 1) * 32767).astype("<i2")
    actual = np.frombuffer(prepare_pcm(values, rate), dtype="<i2")
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(values, before)
