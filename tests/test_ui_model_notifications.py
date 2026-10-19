"""Base-model notification lifetime and turn-progress isolation; no hardware."""
import numpy as np
import pytest

from test_ui import gui


@pytest.mark.parametrize("notification", ["ready", "loading"])
@pytest.mark.parametrize("phase", ["capturing", "busy", "stopped"])
@pytest.mark.parametrize("mode", ["ptt", "auto"])
def test_completed_base_loading_cannot_replace_session_feedback(
        gui, notification, phase, mode):
    w, audio, worker = gui
    w.mode.setCurrentIndex(w.mode.findData(mode))
    worker.loading.emit("Loading baseline models")
    assert w.status.text() == "Loading baseline models"
    worker.ready.emit()
    assert w.models_ready and w.start_button.isEnabled()
    w.start_session()
    speaker = 1
    if phase == "busy":
        assert w.submit_audio(speaker, np.full(8000, .12, dtype=np.float32))
        worker.progress.emit(w.generation, "Translating current turn")
    elif phase == "capturing":
        if mode == "ptt":
            w._press(speaker)
        audio.samples.emit(speaker, np.full(8161, .23, dtype=np.float32))
    else:
        w.stop_session()
    status = w.status.text()
    states = tuple(card.state.text() for card in w.cards)
    generation = w.generation
    busy = w._busy.copy()
    held = w._held.copy()
    sizes = w._sizes.copy()
    pending = w.segmenters[speaker].pending.copy()
    active = w.active
    start_enabled = w.start_button.isEnabled()
    if notification == "ready":
        worker.ready.emit()
    else:
        worker.loading.emit("Obsolete baseline loading")
    assert w.status.text() == status
    assert tuple(card.state.text() for card in w.cards) == states
    assert w.generation == generation and w.active is active
    assert w.models_ready and w.start_button.isEnabled() is start_enabled
    assert w._busy == busy and w._held == held and w._sizes == sizes
    np.testing.assert_array_equal(w.segmenters[speaker].pending, pending)
    if phase == "capturing":
        if mode == "ptt":
            w._release(speaker)
        else:
            audio.samples.emit(speaker, np.zeros(9920, dtype=np.float32))
        assert len(worker.submissions) == 1
        np.testing.assert_array_equal(
            worker.submissions[0][2][worker.submissions[0][2] != 0],
            np.full(8161, .23, dtype=np.float32))
    if w.active:
        w.stop_session()


def test_turn_progress_only_updates_current_active_generation(gui):
    w, audio, worker = gui
    worker.ready.emit()
    w.start_session()
    old_generation = w.generation
    worker.progress.emit(old_generation, "Current progress")
    assert w.status.text() == "Current progress"
    w.stop_session()
    stopped = w.status.text()
    worker.progress.emit(old_generation, "Obsolete progress")
    worker.progress.emit(w.generation, "Stopped progress")
    assert w.status.text() == stopped
    w.start_session()
    worker.progress.emit(old_generation, "Obsolete progress")
    assert "Obsolete" not in w.status.text()
    worker.progress.emit(w.generation, "Fresh progress")
    assert w.status.text() == "Fresh progress"
    w.stop_session()
