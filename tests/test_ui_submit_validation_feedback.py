"""Rejected explicit demo input must not hide live capture/playback state."""
import numpy as np
import pytest

from translitter.core import MAX_TURN_SAMPLES
from test_ui import gui, start, turn
from test_ui_capture_numeric import retained


def rejected_samples(reason):
    if reason == "invalid":
        return np.full(8000, np.nan, dtype=np.float32)
    if reason == "short":
        return np.full(3999, .75, dtype=np.float32)
    return np.full(MAX_TURN_SAMPLES + 1, .75, dtype=np.float32)


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("capture", ["ptt", "overflow", "auto", "playback"])
@pytest.mark.parametrize("reason", ["invalid", "short", "long"])
@pytest.mark.parametrize("outcome", ["result", "empty", "error"])
def test_validation_rejection_preserves_live_feedback(gui, qtbot, speaker, capture, reason, outcome):
    window, audio, worker = gui
    if capture == "auto":
        window.mode.setCurrentIndex(window.mode.findData("auto"))
    start(window, worker, qtbot)
    previous = np.full(8000, .25, dtype=np.float32)
    assert window.submit_audio(speaker, previous)
    if capture == "playback":
        worker.result.emit(turn(window, speaker=1-speaker))
    else:
        if capture != "auto":
            window._press(speaker)
        audio.samples.emit(speaker, np.full(4161, .125, dtype=np.float32))
        if capture == "overflow":
            audio.samples.emit(speaker, np.full(MAX_TURN_SAMPLES, .125, dtype=np.float32))
    before = [retained(window, n) for n in (0, 1)]
    plays = list(audio.plays)
    assert window.submit_audio(speaker, rejected_samples(reason)) is False
    assert len(worker.submissions) == 1
    assert [retained(window, n) for n in (0, 1)] == before
    assert audio.plays == plays
    diagnostic = "maximum" if reason == "long" else "too short or invalid"
    assert diagnostic in window.status.text()

    if outcome == "result":
        worker.result.emit(turn(window, speaker=speaker, audio=None))
        worker.settled.emit(window.generation, speaker, True)
    else:
        if outcome == "error":
            worker.error.emit(window.generation, "Turn processing failed (RuntimeError)")
        worker.settled.emit(window.generation, speaker, False)
    assert window._busy == [0, 0]
    assert window.cards[speaker].state.text() == before[speaker][-2]
    for n in (0, 1):
        assert retained(window, n)[:-2] == before[n][:-2]
    if capture == "playback":
        audio.playing.emit(speaker, False)
        assert "Waiting" in window.cards[speaker].state.text()
        expected = previous
        assert window.submit_audio(speaker, expected)
    else:
        suffix = np.full(4159, .375, dtype=np.float32)
        audio.samples.emit(speaker, suffix)
        if capture == "auto":
            audio.samples.emit(speaker, np.zeros(9600, dtype=np.float32))
        else:
            window._release(speaker)
        if capture == "overflow":
            assert len(worker.submissions) == 1
            window._press(speaker)
            audio.samples.emit(speaker, previous)
            window._release(speaker)
            expected = previous
        else:
            expected = np.concatenate((np.full(4161, .125, dtype=np.float32), suffix))
    assert len(worker.submissions) == 2
    np.testing.assert_array_equal(worker.submissions[-1][2][:len(expected)], expected)
    assert np.count_nonzero(worker.submissions[-1][2]) == len(expected)
    assert window._busy[speaker] == 1


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("reason", ["invalid", "short", "long"])
@pytest.mark.parametrize("pending", [0, 1])
def test_idle_validation_diagnostic_and_recovery(gui, qtbot, speaker, reason, pending):
    window, audio, worker = gui
    start(window, worker, qtbot)
    valid = np.full(8000, .25, dtype=np.float32)
    if pending:
        assert window.submit_audio(speaker, valid)
    before = [retained(window, n) for n in (0, 1)]
    assert window.submit_audio(speaker, rejected_samples(reason)) is False
    diagnostic = "maximum" if reason == "long" else "too short or invalid"
    assert diagnostic in window.cards[speaker].state.text()
    assert diagnostic in window.status.text()
    for n in (0, 1):
        assert retained(window, n)[:-2] == before[n][:-2]
    assert window._busy[speaker] == pending
    assert len(worker.submissions) == pending
    assert window.submit_audio(speaker, valid)
    assert "transcribing" in window.cards[speaker].state.text()
    assert window._busy[speaker] == pending + 1
    np.testing.assert_array_equal(worker.submissions[-1][2], valid)
