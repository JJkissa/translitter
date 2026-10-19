"""Stop/restart generation isolation with held or rejected PTT input.

Only fake audio/worker endpoints are used; queued callbacks model late Qt
notifications from an already cancelled worker generation.
"""
import numpy as np
import pytest
from PySide6.QtCore import QTimer

from translitter.core import MAX_TURN_SAMPLES
from test_ui import gui, start, turn


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("capture", ["held", "overflow", "queue_rejected"])
@pytest.mark.parametrize("restart", [False, True])
def test_stop_isolates_late_notifications_and_recovers_ptt(
        gui, qtbot, speaker, capture, restart):
    window, audio, worker = gui
    start(window, worker, qtbot)
    old_generation = window.generation
    old_voice = np.full(8000, .12, dtype=np.float32)
    # Outstanding accepted work provides a legitimate source of late callbacks.
    assert window.submit_audio(speaker, old_voice)
    window._press(speaker)
    audio.samples.emit(speaker, np.full(
        MAX_TURN_SAMPLES + 1 if capture == "overflow" else 9000,
        .23, dtype=np.float32))
    if capture == "queue_rejected":
        worker.accept = False
        window._release(speaker)
        assert "queue full" in window.cards[speaker].state.text()
    else:
        assert window._held[speaker]
        assert window._capture_overflow[speaker] is (capture == "overflow")
    stale = turn(window, speaker=speaker, original="cancelled old transcript")
    # Also populate pending output, playback, meter and cooldown so Stop's
    # cleanup is checked independently from the worker-generation guards.
    window._deferred[1 - speaker].append((old_voice, 16000))
    window._playback[1 - speaker] = True
    window._cooldown[speaker] = float("inf")
    audio.level.emit(speaker, .1)
    window.stop_session()
    assert not window.active
    assert window.generation > old_generation
    assert worker.stops[-1] == window.generation
    assert window._held == [False, False]
    assert window._chunks == [[], []] and window._sizes == [0, 0]
    assert window._capture_overflow == [False, False]
    assert window._busy == [0, 0] and window._deferred == [[], []]
    assert window._playback == [False, False] and window._cooldown == [0., 0.]
    assert all(card.meter.value() == 0 for card in window.cards)
    attempts = len(worker.submissions)
    window._release(speaker)
    audio.samples.emit(speaker, old_voice)
    assert len(worker.submissions) == attempts

    worker.accept = True
    fresh = np.linspace(.05, .25, 10001, dtype=np.float32)
    if restart:
        start(window, worker, qtbot)
        assert window.generation > old_generation
        assert window.submit_audio(speaker, old_voice)
        window._press(speaker)
        audio.samples.emit(speaker, fresh)
    status = window.status.text()
    states = [card.state.text() for card in window.cards]
    captions = [(card.original.toPlainText(), card.translated.toPlainText())
                for card in window.cards]
    busy = window._busy.copy()
    played = len(audio.plays)
    delivered = []

    def late_notifications():
        worker.result.emit(stale)
        worker.settled.emit(old_generation, speaker, False)
        worker.settled.emit(old_generation, speaker, True)
        worker.error.emit(old_generation, "cancelled old error")
        worker.progress.emit(old_generation, "cancelled old progress")
        delivered.append(True)

    QTimer.singleShot(0, late_notifications)
    qtbot.waitUntil(lambda: bool(delivered))
    assert window.status.text() == status
    assert [card.state.text() for card in window.cards] == states
    assert [(card.original.toPlainText(), card.translated.toPlainText())
            for card in window.cards] == captions
    assert window._busy == busy
    assert len(audio.plays) == played and window._deferred == [[], []]
    if not restart:
        start(window, worker, qtbot)
        window._press(speaker)
        audio.samples.emit(speaker, fresh)
    assert window._held[speaker] and not window._capture_overflow[speaker]
    np.testing.assert_array_equal(np.concatenate(window._chunks[speaker]), fresh)
    before_release = len(worker.submissions)
    window._release(speaker)
    assert len(worker.submissions) == before_release + 1
    assert worker.submissions[-1][:2] == (window.generation, speaker)
    np.testing.assert_array_equal(worker.submissions[-1][2], fresh)
    assert window._busy[speaker] == (2 if restart else 1)
    assert not window._held[speaker] and not window._chunks[speaker]
