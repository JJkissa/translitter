"""Old worker notifications cannot mutate a busy replacement conversation."""
import numpy as np
import pytest
from PySide6.QtCore import QTimer

from test_ui import gui, start, turn


def snapshot(window, audio, worker):
    """Include exact capture bytes and output ownership, not just UI labels."""
    return (
        window.generation, window.active, window.status.text(),
        tuple((c.state.text(), c.original.toPlainText(), c.translated.toPlainText(),
               c.meter.value()) for c in window.cards),
        tuple(window._held), tuple(window._sizes), tuple(window._capture_overflow),
        tuple(tuple(x.tobytes() for x in parts) for parts in window._chunks),
        tuple((s.pending.tobytes(), tuple(x.tobytes() for x in s.active),
               tuple(x.tobytes() for x in s.preroll), s.length, s.voiced, s.quiet)
              for s in window.segmenters),
        tuple(window._busy), tuple(window._playback), tuple(window._cooldown),
        tuple(tuple((id(x), x.tobytes(), rate) for x, rate in queue)
              for queue in window._deferred),
        tuple((listener, id(x), x.tobytes(), rate) for listener, x, rate in audio.plays),
        tuple((g, s, x.tobytes()) for g, s, x in worker.submissions),
    )


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("mode", ["ptt", "auto"])
@pytest.mark.parametrize("notification", ["result", "progress", "empty", "produced", "error"])
def test_old_generation_preserves_live_capture_and_output_fifo(
        gui, qtbot, speaker, mode, notification):
    window, audio, worker = gui
    window.mode.setCurrentIndex(window.mode.findData(mode))
    start(window, worker, qtbot)
    old_generation = window.generation
    voice = np.linspace(.05, .25, 8161, dtype=np.float32)
    assert window.submit_audio(speaker, voice)
    stale = turn(window, speaker=speaker, original="retired transcript")
    window.stop_session()
    start(window, worker, qtbot)
    assert window.generation != old_generation

    # Seed live captions, real playing notifications and a deferred second output.
    assert window.submit_audio(speaker, voice)
    worker.result.emit(turn(window, speaker=speaker, original="fresh caption"))
    assert window.submit_audio(speaker, voice)
    worker.result.emit(turn(window, speaker=speaker, original="fresh second caption"))
    listener = 1 - speaker
    assert window._playback[listener] and len(window._deferred[listener]) == 1
    for _ in range(2):
        assert window.submit_audio(speaker, voice)
    if mode == "ptt":
        window._press(speaker)
    else:
        audio.samples.emit(speaker, np.zeros(640, dtype=np.float32))
    audio.samples.emit(speaker, voice)
    audio.level.emit(speaker, .08)
    if mode == "auto":
        assert window.segmenters[speaker].active
        assert len(window.segmenters[speaker].pending) == 161
    before = snapshot(window, audio, worker)
    delivered = []

    def notify():
        if notification == "result":
            worker.result.emit(stale)
        elif notification == "progress":
            worker.progress.emit(old_generation, "retired progress")
        elif notification == "error":
            worker.error.emit(old_generation, "retired error")
        else:
            worker.settled.emit(old_generation, speaker, notification == "produced")
        delivered.append(True)

    QTimer.singleShot(0, notify)
    qtbot.waitUntil(lambda: bool(delivered))
    assert snapshot(window, audio, worker) == before

    # Current-generation settlement and capture/output completion must still work.
    worker.settled.emit(window.generation, speaker, False)
    worker.settled.emit(window.generation, speaker, False)
    assert window._busy[speaker] == 0
    assert "Talking" in window.cards[speaker].state.text()
    submitted = len(worker.submissions)
    if mode == "ptt":
        window._release(speaker)
        np.testing.assert_array_equal(worker.submissions[-1][2], voice)
    else:
        audio.samples.emit(speaker, np.zeros(9920, dtype=np.float32))
        actual = worker.submissions[-1][2]
        np.testing.assert_array_equal(actual[640:640 + len(voice)], voice)
    assert len(worker.submissions) == submitted + 1
    assert worker.submissions[-1][:2] == (window.generation, speaker)
    assert window._busy[speaker] == 1
    audio.playing.emit(listener, False)
    assert len(audio.plays) == 2 and not window._deferred[listener]
    assert window._playback[listener]
    audio.playing.emit(listener, False)
    assert not window._playback[listener]
    assert "Waiting" in window.cards[listener].state.text()
