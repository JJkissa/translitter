"""Explicit queue rejection must not misrepresent a concurrent live capture."""
import numpy as np
import pytest

from translitter.core import MAX_TURN_SAMPLES
from test_ui import gui, start, turn
from test_ui_capture_numeric import retained


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("capture", ["ptt", "overflow", "auto"])
@pytest.mark.parametrize("pending", [1, 2])
@pytest.mark.parametrize("outcome", ["result", "empty", "error"])
def test_explicit_queue_rejection_preserves_capture_and_settlement_recovery(
        gui, qtbot, speaker, capture, pending, outcome):
    window, audio, worker = gui
    if capture == "auto":
        window.mode.setCurrentIndex(window.mode.findData("auto"))
    start(window, worker, qtbot)
    previous = np.full(8000, .25, dtype=np.float32)
    for _ in range(pending):
        assert window.submit_audio(speaker, previous)
    if capture != "auto":
        window._press(speaker)
    prefix = np.full(4161, .125, dtype=np.float32)
    audio.samples.emit(speaker, prefix)
    if capture == "overflow":
        audio.samples.emit(speaker, np.full(MAX_TURN_SAMPLES, .125, dtype=np.float32))
    before = [retained(window, n) for n in (0, 1)]
    rejected = np.full(8000, .75, dtype=np.float32)
    worker.accept = False
    assert window.submit_audio(speaker, rejected) is False
    assert "not queued" in window.status.text()
    assert [retained(window, n) for n in (0, 1)] == before
    assert len(worker.submissions) == pending + 1
    np.testing.assert_array_equal(worker.submissions[-1][2], rejected)

    if outcome == "result":
        worker.result.emit(turn(window, speaker=speaker, audio=None))
        worker.settled.emit(window.generation, speaker, True)
    else:
        if outcome == "error":
            worker.error.emit(window.generation, "Turn processing failed (RuntimeError)")
        worker.settled.emit(window.generation, speaker, False)
    assert window._busy[speaker] == pending - 1
    assert window.cards[speaker].state.text() == before[speaker][-2]
    assert retained(window, speaker)[:-2] == before[speaker][:-2]

    worker.accept = True
    suffix = np.full(4159, .375, dtype=np.float32)
    audio.samples.emit(speaker, suffix)
    if capture == "auto":
        audio.samples.emit(speaker, np.zeros(9600, dtype=np.float32))
    else:
        window._release(speaker)
    if capture == "overflow":
        assert len(worker.submissions) == pending + 1
        window._press(speaker)
        audio.samples.emit(speaker, previous)
        window._release(speaker)
        expected = previous
    else:
        expected = np.concatenate((prefix, suffix))
    assert len(worker.submissions) == pending + 2
    np.testing.assert_array_equal(worker.submissions[-1][2][:len(expected)], expected)
    assert np.count_nonzero(worker.submissions[-1][2]) == len(expected)
    assert window._busy[speaker] == pending


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("pending", [0, 1])
def test_idle_explicit_queue_rejection_retains_warning_and_next_turn_recovers(
        gui, qtbot, speaker, pending):
    window, audio, worker = gui
    start(window, worker, qtbot)
    voice = np.full(8000, .25, dtype=np.float32)
    if pending:
        assert window.submit_audio(speaker, voice)
    worker.accept = False
    assert window.submit_audio(speaker, voice) is False
    assert "queue full" in window.cards[speaker].state.text()
    assert window._busy[speaker] == pending
    audio.samples.emit(speaker, voice)
    assert len(worker.submissions) == pending + 1
    worker.accept = True
    assert window.submit_audio(speaker, voice)
    assert window._busy[speaker] == pending + 1
    assert "transcribing" in window.cards[speaker].state.text()
