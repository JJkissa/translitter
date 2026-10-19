"""Exact quiet-tail, interrupted speech and reset contracts; no devices/models."""
import numpy as np
import pytest

from translitter.core import Segmenter


CHUNKS = [1, 159, 319, 320, 997, 16384]


def feed(segmenter, samples, chunk_size):
    turns = []
    for start in range(0, len(samples), chunk_size):
        turns.extend(segmenter.push(samples[start:start + chunk_size]))
    return turns


def quiet_frames(count, start=0):
    # Distinct quiet frames expose stale, duplicated or reordered context.
    return np.repeat(np.arange(start + 1, start + count + 1, dtype=np.float32) / 100000, 320)


@pytest.mark.parametrize("chunk_size", CHUNKS)
@pytest.mark.parametrize("quiet_ms", [20, 35, 200])
def test_pause_tail_and_next_onset_have_exact_context(chunk_size, quiet_ms):
    segmenter = Segmenter(quiet_ms=quiet_ms)
    closing_frames = (segmenter.quiet_samples + 319) // 320
    lead = quiet_frames(15)
    speech = np.linspace(0.02, 0.08, 4160, dtype=np.float32)
    tail = quiet_frames(closing_frames + 13, start=20)
    next_speech = np.linspace(0.08, 0.12, 4160, dtype=np.float32)
    closing = quiet_frames(closing_frames, start=60)
    pending = np.full(159, 0.25, dtype=np.float32)
    stream = np.concatenate((lead, speech, tail, next_speech, closing, pending))
    turns = feed(segmenter, stream, chunk_size)
    assert len(turns) == 2
    np.testing.assert_array_equal(turns[0], np.concatenate((lead[-3200:], speech, tail[:closing_frames * 320])))
    np.testing.assert_array_equal(turns[1], np.concatenate((tail[-3200:], next_speech, closing)))
    assert segmenter.active == []
    assert not segmenter.preroll
    assert segmenter.length == segmenter.voiced == segmenter.quiet == 0
    np.testing.assert_array_equal(segmenter.pending, pending)


@pytest.mark.parametrize("chunk_size", CHUNKS)
@pytest.mark.parametrize("long_pause", [False, True])
def test_interrupted_short_speech_requires_contiguous_turn_not_contiguous_voice(chunk_size, long_pause):
    segmenter = Segmenter(quiet_ms=200)
    first = np.full(2240, 0.04, dtype=np.float32)
    second = np.full(2240, 0.08, dtype=np.float32)
    gap = quiet_frames(10 if long_pause else 9)
    closing = quiet_frames(10, start=20)
    turns = feed(segmenter, np.concatenate((first, gap, second, closing)), chunk_size)
    if long_pause:
        assert turns == []  # Each burst is independently below minimum voice.
    else:
        assert len(turns) == 1
        np.testing.assert_array_equal(turns[0], np.concatenate((first, gap, second, closing)))
    assert segmenter.active == []
    assert segmenter.length == segmenter.voiced == segmenter.quiet == 0
    assert len(segmenter.pending) == 0
    # A rejected noise burst must not poison a subsequent eligible turn.
    good = np.full(4160, 0.12, dtype=np.float32)
    recovered = feed(segmenter, np.concatenate((good, closing)), chunk_size)
    assert len(recovered) == 1
    np.testing.assert_array_equal(recovered[0], np.concatenate((good, closing)))


@pytest.mark.parametrize("chunk_size", [159, 320, 997])
@pytest.mark.parametrize("phase", ["idle", "active", "quiet_tail"])
def test_reset_reuse_matches_fresh_instance_with_partial_frame(chunk_size, phase):
    segmenter = Segmenter(quiet_ms=35)
    old = quiet_frames(15)
    if phase != "idle":
        old = np.concatenate((old, np.full(4160, 0.04, dtype=np.float32)))
    if phase == "quiet_tail":
        old = np.concatenate((old, quiet_frames(1, start=20)))
    old = np.concatenate((old, np.full(159, 0.05, dtype=np.float32)))
    assert feed(segmenter, old, chunk_size) == []
    assert len(segmenter.pending) == 159
    segmenter.reset()
    segmenter.reset()  # Idempotent reset must not leave old context or counters.
    fresh = Segmenter(quiet_ms=35)
    stream = np.concatenate((quiet_frames(12, start=30), np.full(4160, 0.1, dtype=np.float32), quiet_frames(2, start=60), np.full(73, 0.2, dtype=np.float32)))
    actual = feed(segmenter, stream, chunk_size)
    expected = feed(fresh, stream, chunk_size)
    assert len(actual) == len(expected) == 1
    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(segmenter.pending, fresh.pending)
    assert segmenter.active == fresh.active == []
    assert not segmenter.preroll and not fresh.preroll
    assert segmenter.length == segmenter.voiced == segmenter.quiet == 0
