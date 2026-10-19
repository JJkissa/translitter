"""Worker-capacity rejection must not hide active headphone playback."""
import numpy as np
import pytest

from test_ui import gui, start, turn
from test_ui_capture_numeric import retained


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("mode", ["ptt", "auto"])
@pytest.mark.parametrize("pending", [1, 2])
@pytest.mark.parametrize("outcome", ["result", "empty", "error"])
def test_queue_rejection_during_playback_preserves_feedback_and_recovery(
        gui, qtbot, speaker, mode, pending, outcome):
    window, audio, worker = gui
    window.mode.setCurrentIndex(window.mode.findData(mode))
    start(window, worker, qtbot)
    accepted = np.full(8000, .25, dtype=np.float32)
    for _ in range(pending):
        assert window.submit_audio(speaker, accepted)
    output = turn(window, speaker=1-speaker)
    worker.result.emit(output)
    assert window._playback[speaker]
    assert audio.plays[-1][1] is output.audio

    # Independent live input must remain byte-exact, including an auto tail.
    other = 1-speaker
    if mode == "ptt":
        window._press(other)
    prefix = np.full(4161, .125, dtype=np.float32)
    audio.samples.emit(other, prefix)
    before = [retained(window, n) for n in (0, 1)]
    playback = tuple(audio.plays)
    deferred = [list(queue) for queue in window._deferred]
    rejected = np.full(8000, .75, dtype=np.float32)
    worker.accept = False
    assert window.submit_audio(speaker, rejected) is False
    assert "not queued" in window.status.text()
    assert [retained(window, n) for n in (0, 1)] == before
    assert window._playback[speaker]
    assert tuple(audio.plays) == playback
    assert window._deferred == deferred
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
    assert retained(window, other) == before[other][:-1] + (tuple(window._busy),)

    audio.playing.emit(speaker, False)
    assert not window._playback[speaker]
    expected = "Busy" if pending > 1 else "Waiting"
    assert expected in window.cards[speaker].state.text()
    assert len(worker.submissions) == pending + 1  # no retry of rejected input
    assert len(audio.plays) == 1

    worker.accept = True
    window._cooldown[speaker] = 0.0
    # Both the independent retained capture and the rejected speaker recover.
    suffix = np.full(4159, .375, dtype=np.float32)
    audio.samples.emit(other, suffix)
    if mode == "auto":
        audio.samples.emit(other, np.zeros(9600, dtype=np.float32))
    else:
        window._release(other)
    np.testing.assert_array_equal(worker.submissions[-1][2][:8320],
                                  np.concatenate((prefix, suffix)))
    assert np.count_nonzero(worker.submissions[-1][2]) == 8320
    assert window.submit_audio(speaker, accepted)
    assert len(worker.submissions) == pending + 3
    np.testing.assert_array_equal(worker.submissions[-1][2], accepted)
    assert window._busy[speaker] == pending
    assert window._busy[other] == 1
    assert "transcribing" in window.cards[speaker].state.text()
