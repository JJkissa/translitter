"""Finite float32 energy classification without overflow or underflow."""
import math

import numpy as np
import pytest

from translitter.core import Segmenter


@pytest.mark.parametrize("shape", ["constant", "alternating", "impulse", "sum_overflow"])
def test_finite_loud_frames_do_not_overflow_energy(shape):
    frame = np.full(320, np.finfo(np.float32).max, dtype=np.float32)
    if shape == "alternating":
        frame[1::2] *= -1
    elif shape == "impulse":
        frame[1:] = 0
    elif shape == "sum_overflow":
        # Each square is finite in float32; only the reduction overflows.
        frame.fill(2e18)
    rms = math.hypot(*(float(value) for value in frame)) / math.sqrt(len(frame))
    assert math.isfinite(rms) and rms > 0.01
    segmenter = Segmenter(quiet_ms=20)
    speech = np.tile(frame, 13)
    closing = np.zeros(320, dtype=np.float32)
    with np.errstate(all="raise"):
        assert segmenter.push(speech) == []
        turns = segmenter.push(closing)
    assert len(turns) == 1
    np.testing.assert_array_equal(turns[0], np.concatenate((speech, closing)))


@pytest.mark.parametrize("chunk_size", [1, 319, 997, 16384])
@pytest.mark.parametrize("amplitude", [np.float32(1e-24), np.nextafter(np.float32(0), np.float32(1))])
def test_tiny_finite_speech_is_classified_before_float32_square_underflow(chunk_size, amplitude):
    frame = np.full(320, amplitude, dtype=np.float32)
    rms = math.hypot(*(float(value) for value in frame)) / math.sqrt(len(frame))
    segmenter = Segmenter(threshold=rms / 2, quiet_ms=20, max_ms=260)
    speech = np.tile(frame, 27)
    turns = []
    with np.errstate(all="raise"):
        for start in range(0, len(speech), chunk_size):
            turns.extend(segmenter.push(speech[start:start + chunk_size]))
    assert len(turns) == 2
    np.testing.assert_array_equal(
        np.concatenate(turns + segmenter.active + [segmenter.pending]), speech)
    assert segmenter.voiced > 0
    segmenter.reset()
    assert segmenter.push(np.zeros(6400, dtype=np.float32)) == []
    with np.errstate(all="raise"):
        recovered = segmenter.push(np.tile(frame, 13))
    assert len(recovered) == 1
    # The 260 ms cap reserves 4000 voiced samples and retains only the
    # newest 160 quiet preroll samples; the remainder stays in the next turn.
    np.testing.assert_array_equal(
        np.concatenate(recovered + segmenter.active + [segmenter.pending]),
        np.concatenate((np.zeros(160, dtype=np.float32), np.tile(frame, 13))))


@pytest.mark.parametrize("amplitude", [np.float32(0), np.float32(1e-24), np.nextafter(np.float32(0), np.float32(1))])
def test_tiny_samples_stay_quiet_at_default_threshold(amplitude):
    segmenter = Segmenter(quiet_ms=20)
    quiet = np.full(6400, amplitude, dtype=np.float32)
    with np.errstate(all="raise"):
        assert segmenter.push(quiet) == []
    assert not segmenter.active and segmenter.voiced == 0
    np.testing.assert_array_equal(np.concatenate(list(segmenter.preroll)), quiet[-3200:])


@pytest.mark.parametrize("ratio, speaking", [(0.999, False), (1, True), (1.001, True)])
def test_ordinary_threshold_remains_inclusive(ratio, speaking):
    threshold = 0.125  # Exactly representable, including its square.
    segmenter = Segmenter(threshold=threshold, quiet_ms=20)
    speech = np.full(4160, threshold * ratio, dtype=np.float32)
    assert segmenter.push(speech) == []
    turns = segmenter.push(np.zeros(320, dtype=np.float32))
    assert len(turns) == int(speaking)
    if speaking:
        np.testing.assert_array_equal(turns[0], np.concatenate((speech, np.zeros(320, dtype=np.float32))))
