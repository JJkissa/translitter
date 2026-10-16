"""Automatic capture lifecycle at the native Window boundary; no devices."""
import numpy as np
import pytest

from test_ui import gui, start


def start_automatic(window, worker, qtbot):
    window.mode.setCurrentIndex(window.mode.findData("auto"))
    start(window, worker, qtbot)


def voice(value=.12):
    return np.full(8000, value, dtype=np.float32)


def silence():
    return np.zeros(9600, dtype=np.float32)


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("already_busy", [False, True])
def test_automatic_rejected_turn_keeps_queue_full_feedback(gui, qtbot, speaker, already_busy):
    window, audio, worker = gui
    start_automatic(window, worker, qtbot)
    if already_busy:
        assert window.submit_audio(speaker, voice(.2))
    busy_before = window._busy[speaker]
    worker.accept = False
    audio.samples.emit(speaker, voice())
    audio.samples.emit(speaker, silence())
    assert len(worker.submissions) == 1 + int(already_busy)
    assert window._busy[speaker] == busy_before
    assert "turn not queued" in window.status.text()
    assert "queue full" in window.cards[speaker].state.text(), "Automatic idle rendering must not hide a rejected turn"
    audio.samples.emit(speaker, silence())
    assert "queue full" in window.cards[speaker].state.text(), "Idle microphone callbacks must not immediately erase rejection feedback"


@pytest.mark.parametrize("speaker", [0, 1])
def test_automatic_stop_restart_discards_partial_capture(gui, qtbot, speaker):
    window, audio, worker = gui
    start_automatic(window, worker, qtbot)
    old_generation = window.generation
    audio.samples.emit(speaker, voice(.12))
    assert window.segmenters[speaker].active
    window.stop_session()
    audio.samples.emit(speaker, voice(.2))
    assert not worker.submissions
    assert not window.segmenters[speaker].active
    start_automatic(window, worker, qtbot)
    audio.samples.emit(speaker, silence())
    assert not worker.submissions
    audio.samples.emit(speaker, voice(.3))
    audio.samples.emit(speaker, silence())
    assert len(worker.submissions) == 1
    generation, actual_speaker, samples = worker.submissions[0]
    assert generation == window.generation and generation != old_generation
    assert actual_speaker == speaker
    assert not np.any(samples == np.float32(.12))
    assert not np.any(samples == np.float32(.2))
    assert np.count_nonzero(samples == np.float32(.3)) == 8000


@pytest.mark.parametrize("speaker", [0, 1])
def test_automatic_playback_cooldown_discards_partial_and_suppressed_audio(gui, qtbot, monkeypatch, speaker):
    import translitter.ui as ui
    now = [100.0]
    monkeypatch.setattr(ui, "monotonic", lambda: now[0])
    window, audio, worker = gui
    start_automatic(window, worker, qtbot)
    audio.samples.emit(speaker, voice(.12))
    audio.playing.emit(speaker, True)
    assert not window.segmenters[speaker].active
    audio.samples.emit(speaker, voice(.2))
    audio.samples.emit(speaker, silence())
    assert not worker.submissions
    audio.playing.emit(speaker, False)
    now[0] += .299
    audio.samples.emit(speaker, voice(.21))
    assert not window.segmenters[speaker].active
    now[0] += .002
    audio.samples.emit(speaker, silence())
    assert not worker.submissions
    audio.samples.emit(speaker, voice(.3))
    audio.samples.emit(speaker, silence())
    assert len(worker.submissions) == 1
    samples = worker.submissions[0][2]
    for discarded in (.12, .2, .21):
        assert not np.any(samples == np.float32(discarded))
    assert np.count_nonzero(samples == np.float32(.3)) == 8000


@pytest.mark.parametrize("speaker", [0, 1])
def test_automatic_rejected_turn_does_not_contaminate_next_accepted_turn(gui, qtbot, speaker):
    window, audio, worker = gui
    start_automatic(window, worker, qtbot)
    worker.accept = False
    audio.samples.emit(speaker, voice(.12))
    audio.samples.emit(speaker, silence())
    assert window._busy[speaker] == 0
    worker.accept = True
    audio.samples.emit(speaker, voice(.3))
    audio.samples.emit(speaker, silence())
    assert len(worker.submissions) == 2 and window._busy[speaker] == 1
    assert not np.any(worker.submissions[-1][2] == np.float32(.12))
    assert np.count_nonzero(worker.submissions[-1][2] == np.float32(.3)) == 8000
    assert "Busy" in window.cards[speaker].state.text()


@pytest.mark.parametrize("speaker", [0, 1])
def test_automatic_short_turn_returns_to_waiting(gui, qtbot, speaker):
    window, audio, worker = gui
    start_automatic(window, worker, qtbot)
    audio.samples.emit(speaker, np.full(3200, .12, dtype=np.float32))
    assert "Talking" in window.cards[speaker].state.text()
    audio.samples.emit(speaker, silence())
    assert not worker.submissions and window._busy[speaker] == 0
    assert "Waiting" in window.cards[speaker].state.text()
