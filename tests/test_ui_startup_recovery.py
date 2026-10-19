"""Native startup rollback and retry; never open physical audio endpoints."""
import numpy as np
import pytest
from PySide6.QtCore import Qt

from test_ui import gui, turn


@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("stage", ["worker_exception", "audio_exception", "audio_failure_0", "audio_failure_1"])
def test_startup_failure_preserves_stopped_state_and_allows_retry(
        gui, qtbot, monkeypatch, automatic, stage):
    window, audio, worker = gui
    if automatic:
        window.mode.setCurrentIndex(1)
    original_session = worker.set_session
    original_start = audio.start
    attempted = []

    def set_session(generation, routes):
        original_session(generation, routes)
        attempted.append(generation)
        if stage == "worker_exception":
            raise RuntimeError("injected worker setup failure")

    def audio_start(routes):
        original_start(routes)
        if stage == "audio_exception":
            raise RuntimeError("injected audio startup failure")
        audio.failure.emit(int(stage[-1]), "injected synchronous audio failure")

    monkeypatch.setattr(worker, "set_session", set_session)
    monkeypatch.setattr(audio, "start", audio_start)
    worker.ready.emit()
    qtbot.mouseClick(window.start_button, Qt.LeftButton)
    assert len(attempted) == 1
    failed_generation = attempted[0]
    assert not window.active
    assert window.generation > failed_generation
    assert worker.stops[-1] == window.generation and audio.stops >= 1
    assert all(card.state.text() == "Stopped" for card in window.cards)
    assert "failure" in window.status.text()
    assert "Listening" not in window.status.text()
    assert window.start_button.isEnabled() and not window.stop_button.isEnabled()
    assert all(selector.isEnabled() for selector in window.selectors)
    assert all(not card.hold.isEnabled() and card.meter.value() == 0
               for card in window.cards)
    assert window._held == [False, False] and window._chunks == [[], []]
    assert window._sizes == [0, 0] and window._capture_overflow == [False, False]
    assert window._busy == [0, 0] and window._deferred == [[], []]
    assert window._playback == [False, False] and window._cooldown == [0., 0.]
    status = window.status.text()
    worker.result.emit(turn(window, generation=failed_generation, original="stale startup"))
    worker.progress.emit(failed_generation, "stale progress")
    worker.error.emit(failed_generation, "stale error")
    worker.settled.emit(failed_generation, 0, False)
    assert window.status.text() == status
    assert not audio.plays and not window.cards[0].original.toPlainText()

    monkeypatch.setattr(worker, "set_session", original_session)
    monkeypatch.setattr(audio, "start", original_start)
    qtbot.mouseClick(window.start_button, Qt.LeftButton)
    assert window.active and window.generation > failed_generation
    assert "Listening" in window.status.text()
    assert window.stop_button.isEnabled() and not window.start_button.isEnabled()
    voice = np.full(8000, .15, dtype=np.float32)
    if not automatic:
        window._press(1)
    audio.samples.emit(1, voice)
    if automatic:
        audio.samples.emit(1, np.zeros(9920, dtype=np.float32))
    else:
        window._release(1)
    assert len(worker.submissions) == 1
    assert worker.submissions[0][:2] == (window.generation, 1)
    if automatic:
        assert np.count_nonzero(worker.submissions[0][2]) == len(voice)
    else:
        np.testing.assert_array_equal(worker.submissions[0][2], voice)


@pytest.mark.parametrize("automatic", [False, True])
def test_outer_start_does_not_reset_synchronously_retried_session(
        gui, qtbot, monkeypatch, automatic):
    window, audio, worker = gui
    if automatic:
        window.mode.setCurrentIndex(1)
    original_start = audio.start
    generations = []
    voice = np.full(8000, .18, dtype=np.float32)

    def audio_start(routes):
        original_start(routes)
        generations.append(window.generation)
        audio.failure.emit(0, "injected startup failure")
        monkeypatch.setattr(audio, "start", original_start)
        window.start_session()
        assert window.active
        generations.append(window.generation)
        assert window.submit_audio(0, voice)
        if not automatic:
            window._press(1)
        audio.samples.emit(1, voice)

    monkeypatch.setattr(audio, "start", audio_start)
    worker.ready.emit()
    qtbot.mouseClick(window.start_button, Qt.LeftButton)
    assert len(generations) == 2 and generations[1] > generations[0]
    assert window.active and window.generation == generations[1]
    assert window._busy == [1, 0]
    assert "Busy" in window.cards[0].state.text()
    assert "Talking" in window.cards[1].state.text()
    if automatic:
        assert window.segmenters[1].active
        audio.samples.emit(1, np.zeros(9920, dtype=np.float32))
        assert np.count_nonzero(worker.submissions[-1][2]) == len(voice)
    else:
        assert window._held[1]
        np.testing.assert_array_equal(np.concatenate(window._chunks[1]), voice)
        window._release(1)
        np.testing.assert_array_equal(worker.submissions[-1][2], voice)
    assert len(worker.submissions) == 2
    assert worker.submissions[-1][:2] == (generations[1], 1)
