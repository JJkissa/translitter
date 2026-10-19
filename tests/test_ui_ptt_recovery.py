"""Release-only PTT queue rejection and concurrent settlement recovery."""
import numpy as np
import pytest

from translitter.core import MAX_TURN_SAMPLES
from test_ui import gui, start


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("pending", [0, 1])
def test_rejected_release_preserves_busy_count_and_next_hold(gui, qtbot, speaker, pending):
    window, audio, worker = gui
    start(window, worker, qtbot)
    generation = window.generation
    voice = np.full(8000, .12, dtype=np.float32)
    if pending:
        assert window.submit_audio(speaker, voice)
    worker.accept = False
    window._press(speaker)
    audio.samples.emit(speaker, voice)
    window._release(speaker)
    assert window._busy[speaker] == pending
    assert not window._held[speaker] and not window._chunks[speaker]
    assert window._sizes[speaker] == 0 and not window._capture_overflow[speaker]
    assert "queue full" in window.cards[speaker].state.text()
    assert "not queued" in window.status.text()
    attempts = len(worker.submissions)
    window._release(speaker)
    audio.samples.emit(speaker, voice)
    assert len(worker.submissions) == attempts
    worker.accept = True
    fresh = np.full(9000, .23, dtype=np.float32)
    window._press(speaker)
    audio.samples.emit(speaker, fresh)
    window._release(speaker)
    assert len(worker.submissions) == attempts + 1
    np.testing.assert_array_equal(worker.submissions[-1][2], fresh)
    assert worker.submissions[-1][:2] == (generation, speaker)
    assert window._busy[speaker] == pending + 1


@pytest.mark.parametrize("speaker", [0, 1])
def test_overflow_after_rejected_release_resets_before_next_hold(gui, qtbot, speaker):
    window, audio, worker = gui
    start(window, worker, qtbot)
    voice = np.full(8000, .12, dtype=np.float32)
    worker.accept = False
    window._press(speaker)
    audio.samples.emit(speaker, voice)
    window._release(speaker)
    assert window._busy[speaker] == 0
    worker.accept = True
    window._press(speaker)
    audio.samples.emit(speaker, np.full(MAX_TURN_SAMPLES + 1, .2, dtype=np.float32))
    window._release(speaker)
    assert len(worker.submissions) == 1
    assert not window._capture_overflow[speaker]
    assert "discarded" in window.cards[speaker].state.text()
    window._press(speaker)
    audio.samples.emit(speaker, voice)
    window._release(speaker)
    assert len(worker.submissions) == 2 and window._busy[speaker] == 1
    np.testing.assert_array_equal(worker.submissions[-1][2], voice)


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("overflow", [False, True])
@pytest.mark.parametrize("outcome", ["empty", "error"])
def test_prior_turn_settlement_preserves_current_hold_feedback(
        gui, qtbot, speaker, overflow, outcome):
    window, audio, worker = gui
    start(window, worker, qtbot)
    voice = np.full(8000, .12, dtype=np.float32)
    assert window.submit_audio(speaker, voice)
    window._press(speaker)
    held = np.full(MAX_TURN_SAMPLES + 1 if overflow else 9000, .23, dtype=np.float32)
    audio.samples.emit(speaker, held)
    before = window.cards[speaker].state.text()
    if outcome == "error":
        worker.error.emit(window.generation, "Turn processing failed (RuntimeError)")
    worker.settled.emit(window.generation, speaker, False)
    assert window._busy[speaker] == 0
    assert window._held[speaker]
    assert window.cards[speaker].state.text() == before, "Older settlement must not hide an active/rejected hold"
    assert window._capture_overflow[speaker] is overflow
    if not overflow:
        np.testing.assert_array_equal(np.concatenate(window._chunks[speaker]), held)
    window._release(speaker)
    assert len(worker.submissions) == (1 if overflow else 2)
    if not overflow:
        np.testing.assert_array_equal(worker.submissions[-1][2], held)
