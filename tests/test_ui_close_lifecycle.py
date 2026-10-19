"""Native asynchronous close with fake endpoints and pending inference only."""
import numpy as np
import pytest
from PySide6.QtCore import QTimer, Qt

from test_ui import gui, start, turn


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("mode", ["ptt", "auto"])
@pytest.mark.parametrize("pending", [False, True])
def test_close_cancels_capture_and_ignores_late_notifications(
        gui, qtbot, monkeypatch, speaker, mode, pending):
    window, audio, worker = gui
    window.mode.setCurrentIndex(window.mode.findData(mode))
    start(window, worker, qtbot)
    voice = np.full(8000, .12, dtype=np.float32)
    assert window.submit_audio(speaker, voice)
    generation = window.generation
    stale = turn(window, speaker=speaker, original="late closed transcript")
    if mode == "ptt":
        window._press(speaker)
    audio.samples.emit(speaker, np.full(8161, .23, dtype=np.float32))
    if mode == "auto":
        assert window.segmenters[speaker].active
        assert window.segmenters[speaker].pending.size == 161
    else:
        assert window._sizes[speaker] == 8161
    audio.level.emit(speaker, .12)
    # Simulate existing output plus a queued translation independently from
    # capture, so closing must clear both sides without draining the FIFO.
    audio.playing.emit(1 - speaker, True)
    window._deferred[1 - speaker].append((voice, 16000))
    window._discovery.start(800)
    worker.finished = not pending
    waits = []
    original_wait = worker.wait

    def nonblocking_wait(ms):
        waits.append(ms)
        return original_wait(ms)

    monkeypatch.setattr(worker, "wait", nonblocking_wait)
    stopped_before = audio.stops
    assert window.close() is (not pending)
    try:
        assert window._closing and not window.active
        assert window.generation > generation
        closed_generation = window.generation
        assert audio.stops == stopped_before + 1
        assert worker.shutdowns == 1
        assert worker.stops[-1] == closed_generation
        assert not window._discovery.isActive()
        assert window._close_poll.isActive() is pending
        assert window.isVisible() is pending
        assert not window.centralWidget().isEnabled()
        assert window._held == [False, False]
        assert window._chunks == [[], []] and window._sizes == [0, 0]
        assert window._capture_overflow == [False, False]
        assert all(not segment.active and segment.pending.size == 0
                   for segment in window.segmenters)
        assert window._busy == [0, 0] and window._deferred == [[], []]
        assert window._playback == [False, False] and window._cooldown == [0., 0.]
        assert all(card.meter.value() == 0 for card in window.cards)
        status = window.status.text()
        states = [card.state.text() for card in window.cards]
        captions = [(card.original.toPlainText(), card.translated.toPlainText())
                    for card in window.cards]
        submitted, played, started = (len(worker.submissions), len(audio.plays),
                                      len(audio.starts))
        delivered = []

        def late_notifications():
            worker.result.emit(stale)
            worker.result.emit(turn(window, speaker=speaker))
            worker.settled.emit(generation, speaker, False)
            worker.settled.emit(closed_generation, speaker, False)
            worker.progress.emit(closed_generation, "late progress")
            worker.error.emit(closed_generation, "late error")
            worker.error.emit(-1, "late loading failure")
            worker.loading.emit("late loading")
            worker.ready.emit()
            audio.failure.emit(speaker, "late audio failure")
            audio.pairs = []
            audio.devices_changed.emit()
            audio.samples.emit(speaker, voice)
            audio.level.emit(speaker, .5)
            audio.playing.emit(speaker, True)
            audio.playing.emit(1 - speaker, False)
            window._release(speaker)
            window.start_session()
            delivered.append(True)

        QTimer.singleShot(0, late_notifications)
        qtbot.waitUntil(lambda: bool(delivered))
        if pending:
            # The repeated native close and multiple poll cycles must not
            # repeat cancellation/shutdown or block the event loop.
            assert not window.close()
            qtbot.wait(120)
            assert window.isVisible() and window._close_poll.isActive()
        assert window.status.text() == status
        assert [card.state.text() for card in window.cards] == states
        assert [(card.original.toPlainText(), card.translated.toPlainText())
                for card in window.cards] == captions
        assert audio.stops == stopped_before + 1 and worker.shutdowns == 1
        assert window.generation == closed_generation
        assert len(worker.submissions) == submitted and len(audio.plays) == played
        assert len(audio.starts) == started
        assert not window._discovery.isActive()
        assert window._deferred == [[], []] and window._busy == [0, 0]
        assert window._playback == [False, False]
        assert all(card.meter.value() == 0 for card in window.cards)
        assert not window.start_button.isEnabled()
        assert not window.stop_button.isEnabled()
        worker.finished = True
        if pending:
            qtbot.waitUntil(lambda: not window.isVisible())
        assert not window._close_poll.isActive()
        assert not window._discovery.isActive()
        assert worker.shutdowns == 1 and audio.stops == stopped_before + 1
        assert waits and all(ms == 0 for ms in waits)
        # A closed window no longer intercepts application key events.
        qtbot.keyPress(window.cards[speaker].original,
                       Qt.Key_F1 if speaker == 0 else Qt.Key_F2)
        assert not any(window._held)
    finally:
        worker.finished = True
        window.close()
