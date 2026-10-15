"""Small audio buffers remain independent, silent input never submits a turn."""
import numpy as np
from translitter import core


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
