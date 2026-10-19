"""Accepted explicit turns must not hide concurrent capture/playback feedback."""
import numpy as np
import pytest

from translitter.core import MAX_TURN_SAMPLES
from test_ui import gui, start, turn
from test_ui_capture_numeric import retained


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("capture", ["ptt", "overflow", "auto"])
@pytest.mark.parametrize("pending", [0, 1])
@pytest.mark.parametrize("outcome", ["result", "empty", "error"])
def test_accepted_explicit_turn_preserves_capture_and_recovers(
        gui, qtbot, speaker, capture, pending, outcome):
    window, audio, worker = gui
    if capture == "auto":
        window.mode.setCurrentIndex(window.mode.findData("auto"))
    start(window, worker, qtbot)
    explicit = np.full(8000, .75, dtype=np.float32)
    if pending:
        assert window.submit_audio(speaker, explicit)
    if capture != "auto":
        window._press(speaker)
    prefix = np.full(4161, .125, dtype=np.float32)
    audio.samples.emit(speaker, prefix)
    if capture == "overflow":
        audio.samples.emit(speaker, np.full(MAX_TURN_SAMPLES, .125, dtype=np.float32))
    before = [retained(window, n) for n in (0, 1)]
    assert window.submit_audio(speaker, explicit) is True
    assert window._busy[speaker] == pending + 1
    assert window._busy[1-speaker] == 0
    assert len(worker.submissions) == pending + 1
    np.testing.assert_array_equal(worker.submissions[-1][2], explicit)
    for n in (0, 1):
        assert retained(window, n)[:-2] == before[n][:-2]
        assert window.cards[n].state.text() == before[n][-2]

    if outcome == "result":
        worker.result.emit(turn(window, speaker=speaker, audio=None))
        worker.settled.emit(window.generation, speaker, True)
    else:
        if outcome == "error":
            worker.error.emit(window.generation, "Turn processing failed (RuntimeError)")
        worker.settled.emit(window.generation, speaker, False)
    assert window._busy[speaker] == pending
    assert window.cards[speaker].state.text() == before[speaker][-2]
    assert retained(window, speaker)[:-2] == before[speaker][:-2]

    suffix = np.full(4159, .375, dtype=np.float32)
    audio.samples.emit(speaker, suffix)
    if capture == "auto":
        audio.samples.emit(speaker, np.zeros(9600, dtype=np.float32))
    else:
        window._release(speaker)
    if capture == "overflow":
        assert len(worker.submissions) == pending + 1
        window._press(speaker)
        audio.samples.emit(speaker, explicit)
        window._release(speaker)
        expected = explicit
    else:
        expected = np.concatenate((prefix, suffix))
    assert len(worker.submissions) == pending + 2
    np.testing.assert_array_equal(worker.submissions[-1][2][:len(expected)], expected)
    assert np.count_nonzero(worker.submissions[-1][2]) == len(expected)
    assert window._busy[speaker] == pending + 1


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("mode", ["ptt", "auto"])
@pytest.mark.parametrize("playing", [False, True])
def test_accepted_explicit_turn_idle_and_playback_feedback(gui, qtbot, speaker, mode, playing):
    window, audio, worker = gui
    window.mode.setCurrentIndex(window.mode.findData(mode))
    start(window, worker, qtbot)
    if playing:
        worker.result.emit(turn(window, speaker=1-speaker))
    before = [retained(window, n) for n in (0, 1)]
    samples = np.full(8000, .25, dtype=np.float32)
    assert window.submit_audio(speaker, samples) is True
    assert window._busy[speaker] == 1
    assert window._playback[speaker] is playing
    for n in (0, 1):
        assert retained(window, n)[:-2] == before[n][:-2]
    if playing:
        assert window.cards[speaker].state.text() == before[speaker][-2]
        audio.playing.emit(speaker, False)
        assert "Busy" in window.cards[speaker].state.text()
    else:
        assert "transcribing" in window.cards[speaker].state.text()
    worker.result.emit(turn(window, speaker=speaker, audio=None))
    assert window._busy[speaker] == 0
    assert "Waiting" in window.cards[speaker].state.text()
    np.testing.assert_array_equal(worker.submissions[-1][2], samples)
