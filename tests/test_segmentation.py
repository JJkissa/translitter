"""Small audio buffers remain independent, silent input never submits a turn."""
import numpy as np
import pytest
from translitter import core


@pytest.mark.parametrize("max_ms", [250, 500, 510, 8000, 8010])
def test_maximum_boundary_preserves_every_voiced_sample(max_ms):
    segmenter = core.Segmenter(threshold=0.01, max_ms=max_ms)
    # Non-frame-aligned limits must carry the boundary frame's remainder,
    # not clip and silently lose it on every emitted turn.
    samples = np.linspace(0.02, 0.08, 16000 * 18, dtype=np.float32)
    turns = segmenter.push(samples)
    assert turns
    assert all(len(turn) == segmenter.max_samples for turn in turns)
    retained = turns + segmenter.active + [segmenter.pending]
    np.testing.assert_array_equal(np.concatenate(retained), samples)


@pytest.mark.parametrize("frames, expected", [(12, 0), (13, 1)])
def test_minimum_voiced_duration_at_frame_boundary(frames, expected):
    segmenter = core.Segmenter(threshold=0.125, quiet_ms=200)
    assert segmenter.push(np.full(frames * 320, 0.125, dtype=np.float32)) == []
    turns = segmenter.push(np.zeros(3200, dtype=np.float32))
    assert len(turns) == expected


def test_silence_closes_only_at_complete_quiet_boundary():
    segmenter = core.Segmenter(quiet_ms=200)
    assert segmenter.push(np.full(4160, 0.05, dtype=np.float32)) == []
    assert segmenter.push(np.zeros(3199, dtype=np.float32)) == []
    turns = segmenter.push(np.zeros(1, dtype=np.float32))
    assert len(turns) == 1
    assert len(turns[0]) == 7360
    assert segmenter.push(np.zeros(16000, dtype=np.float32)) == []


@pytest.mark.parametrize("chunk_size", [1, 319, 500, 4096])
def test_callback_chunking_does_not_change_turns(chunk_size):
    samples = np.concatenate((np.zeros(3200), np.full(20000, 0.05),
                              np.zeros(6400))).astype(np.float32)
    expected = core.Segmenter(quiet_ms=200, max_ms=510).push(samples)
    segmenter = core.Segmenter(quiet_ms=200, max_ms=510)
    actual = []
    for start in range(0, len(samples), chunk_size):
        actual.extend(segmenter.push(samples[start:start + chunk_size]))
    assert len(actual) == len(expected)
    for turn, reference in zip(actual, expected):
        np.testing.assert_array_equal(turn, reference)


@pytest.mark.parametrize("max_ms", [250, 260, 400, 440, 450, 510, 8000])
def test_preroll_cannot_crowd_out_minimum_speech_at_small_limits(max_ms):
    segmenter = core.Segmenter(quiet_ms=200, max_ms=max_ms)
    preroll = np.linspace(0.001, 0.005, 3200, dtype=np.float32)
    assert segmenter.push(preroll) == []
    speech = np.linspace(0.02, 0.08, 320000, dtype=np.float32)
    turns = segmenter.push(speech)
    assert turns
    retained = np.concatenate(turns + segmenter.active + [segmenter.pending])
    # Even the first voiced samples must survive a full idle preroll.
    np.testing.assert_array_equal(retained[retained >= 0.01], speech)
    preroll_size = min(len(preroll), segmenter.max_samples - 4000)
    expected_preroll = preroll[-preroll_size:] if preroll_size else preroll[:0]
    np.testing.assert_array_equal(turns[0][:preroll_size], expected_preroll)
    assert all(len(turn) <= segmenter.max_samples for turn in turns)
    assert all(np.count_nonzero(turn >= 0.01) >= 4000 for turn in turns)


@pytest.mark.parametrize("max_ms", [250, 260, 400, 8000])
def test_small_limit_preroll_does_not_make_short_noise_eligible(max_ms):
    segmenter = core.Segmenter(quiet_ms=200, max_ms=max_ms)
    samples = np.concatenate((np.zeros(3200), np.full(3840, 0.05),
                              np.zeros(6400))).astype(np.float32)
    assert segmenter.push(samples) == []


@pytest.mark.parametrize("chunk_size", [1, 319, 4096])
def test_small_limit_preroll_is_callback_independent(chunk_size):
    samples = np.concatenate((np.zeros(3200), np.full(16000, 0.05),
                              np.zeros(6400))).astype(np.float32)
    expected = core.Segmenter(quiet_ms=200, max_ms=260).push(samples)
    segmenter = core.Segmenter(quiet_ms=200, max_ms=260)
    actual = []
    for start in range(0, len(samples), chunk_size):
        actual.extend(segmenter.push(samples[start:start + chunk_size]))
    assert len(actual) == len(expected)
    for turn, reference in zip(actual, expected):
        np.testing.assert_array_equal(turn, reference)


def test_reset_discards_active_preroll_and_partial_frame():
    segmenter = core.Segmenter(quiet_ms=200)
    segmenter.push(np.zeros(3200, dtype=np.float32))
    segmenter.push(np.full(4161, 0.05, dtype=np.float32))
    assert segmenter.active and len(segmenter.pending) == 1
    segmenter.reset()
    assert not segmenter.active and not segmenter.preroll
    assert len(segmenter.pending) == 0
    assert segmenter.length == segmenter.voiced == segmenter.quiet == 0
    assert segmenter.push(np.zeros(6400, dtype=np.float32)) == []
    assert segmenter.push(np.full(4160, 0.1, dtype=np.float32)) == []
    turns = segmenter.push(np.zeros(3200, dtype=np.float32))
    assert len(turns) == 1
    assert np.count_nonzero(turns[0]) == 4160
    assert np.all(turns[0][turns[0] != 0] == np.float32(0.1))


def test_segmenter_closes_a_turn_after_a_pause():
    assert hasattr(core, "Segmenter"), "Speech segmentation is missing"
    segmenter = core.Segmenter(threshold=0.01, quiet_ms=200)
    assert segmenter.push(np.zeros(16000, dtype=np.float32)) == []
    assert segmenter.push(np.full(6400, 0.05, dtype=np.float32)) == []
    turns = segmenter.push(np.zeros(4000, dtype=np.float32))
    assert len(turns) == 1
    assert float(np.max(turns[0])) == np.float32(0.05)


def test_short_noise_is_not_a_turn():
    assert hasattr(core, "Segmenter"), "Speech segmentation is missing"
    segmenter = core.Segmenter(threshold=0.01, quiet_ms=200)
    turns = segmenter.push(np.concatenate((np.full(640, 0.2), np.zeros(6400))).astype(np.float32))
    assert turns == []


def test_long_speech_is_bounded_and_two_inputs_have_separate_state():
    assert hasattr(core, "Segmenter"), "Speech segmentation is missing"
    first = core.Segmenter(threshold=0.01, max_ms=500)
    second = core.Segmenter(threshold=0.01, max_ms=500)
    turns = first.push(np.full(16000, 0.05, dtype=np.float32))
    assert len(turns) >= 1 and all(len(turn) <= 8000 for turn in turns)
    assert second.push(np.zeros(16000, dtype=np.float32)) == []
    first.reset()
    assert first.push(np.zeros(16000, dtype=np.float32)) == []
