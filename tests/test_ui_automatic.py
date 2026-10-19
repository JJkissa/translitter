"""Automatic capture lifecycle at the native Window boundary; no devices."""
import numpy as np
import pytest

from test_ui import focus_cycle, gui, start, turn


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


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("prefix_size", [160, 8000])
def test_automatic_focus_loss_preserves_unsubmitted_speech(gui, qtbot, speaker, prefix_size):
    window, audio, worker = gui
    start_automatic(window, worker, qtbot)
    generation = window.generation
    audio.samples.emit(speaker, np.full(prefix_size, .12, dtype=np.float32))
    status = window.status.text()
    states = [card.state.text() for card in window.cards]
    focus_cycle(window)
    assert window.active and window.generation == generation
    assert window.status.text() == status
    assert [card.state.text() for card in window.cards] == states
    assert not worker.submissions
    audio.samples.emit(speaker, voice(.3))
    # One extra frame closes the turn even with a half-frame prefix.
    audio.samples.emit(speaker, np.zeros(9920, dtype=np.float32))
    assert len(worker.submissions) == 1
    actual_generation, actual_speaker, samples = worker.submissions[0]
    assert (actual_generation, actual_speaker) == (generation, speaker)
    assert np.count_nonzero(samples == np.float32(.12)) == prefix_size
    assert np.count_nonzero(samples == np.float32(.3)) == 8000


@pytest.mark.parametrize("speaker", [0, 1])
def test_automatic_focus_loss_preserves_quiet_preroll(gui, qtbot, speaker):
    window, audio, worker = gui
    start_automatic(window, worker, qtbot)
    audio.samples.emit(speaker, np.full(3200, .002, dtype=np.float32))
    focus_cycle(window)
    audio.samples.emit(speaker, voice())
    audio.samples.emit(speaker, silence())
    assert len(worker.submissions) == 1
    samples = worker.submissions[0][2]
    np.testing.assert_array_equal(samples[:3200], np.full(3200, .002, dtype=np.float32))
    assert np.count_nonzero(samples == np.float32(.12)) == 8000


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("remaining", [0, 1])
@pytest.mark.parametrize("completion", ["result", "empty", "error"])
def test_automatic_earlier_turn_completion_keeps_current_talking_state(
        gui, qtbot, speaker, remaining, completion):
    window, audio, worker = gui
    start_automatic(window, worker, qtbot)
    for _ in range(1 + remaining):
        audio.samples.emit(speaker, voice(.2))
        audio.samples.emit(speaker, silence())
    assert window._busy[speaker] == 1 + remaining
    audio.samples.emit(speaker, voice(.3))
    assert window.segmenters[speaker].active
    talking = window.cards[speaker].state.text()
    assert "Talking" in talking
    if completion == "result":
        worker.result.emit(turn(window, speaker=speaker, audio=None))
        worker.settled.emit(window.generation, speaker, True)
        assert window.cards[speaker].original.toPlainText()
    else:
        if completion == "error":
            worker.error.emit(window.generation, "test interpretation failure")
            assert "Interpretation error" in window.status.text()
        worker.settled.emit(window.generation, speaker, False)
    assert window._busy[speaker] == remaining
    assert window.cards[speaker].state.text() == talking
    assert window.segmenters[speaker].active
    # No further speech callback is needed to repair the display. Complete the
    # new turn and verify its speech survived the earlier worker notification.
    audio.samples.emit(speaker, silence())
    assert window._busy[speaker] == remaining + 1
    assert np.count_nonzero(worker.submissions[-1][2] == np.float32(.3)) == 8000
    assert "Busy" in window.cards[speaker].state.text()


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("remaining", [0, 1])
@pytest.mark.parametrize("produced", [False, True])
def test_automatic_completion_without_new_speech_keeps_idle_feedback(
        gui, qtbot, speaker, remaining, produced):
    window, audio, worker = gui
    start_automatic(window, worker, qtbot)
    for _ in range(1 + remaining):
        audio.samples.emit(speaker, voice())
        audio.samples.emit(speaker, silence())
    assert not window.segmenters[speaker].active
    if produced:
        worker.result.emit(turn(window, speaker=speaker, audio=None))
    worker.settled.emit(window.generation, speaker, produced)
    assert window._busy[speaker] == remaining
    state = window.cards[speaker].state.text()
    if remaining:
        assert "Busy" in state
    elif produced:
        assert state == "Waiting · automatic conversation"
    else:
        assert state == "Waiting · turn produced no transcript"
