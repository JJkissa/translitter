"""Failed headphone dispatch must not strand later deferred translations."""
import numpy as np
import pytest

from test_ui import gui, start, turn, focus_cycle


@pytest.mark.parametrize("listener", [0, 1])
@pytest.mark.parametrize("trigger", ["release", "focus_loss", "completion"])
@pytest.mark.parametrize("last_succeeds", [False, True])
def test_failed_dispatch_drains_remaining_fifo(
        gui, qtbot, monkeypatch, listener, trigger, last_succeeds):
    window, audio, worker = gui
    start(window, worker, qtbot)
    generation = window.generation
    voice = np.full(8000, .12, dtype=np.float32)
    if trigger == "completion":
        worker.result.emit(turn(window, speaker=1 - listener))
        assert window._playback[listener]
    else:
        window._press(listener)
        audio.samples.emit(listener, voice)
    pending = [turn(window, speaker=1 - listener, original=name)
               for name in ("failed", "next")]
    for item in pending:
        worker.result.emit(item)
    assert len(window._deferred[listener]) == 2
    attempts = []
    original_play = audio.play

    def dispatch(target, samples, rate):
        attempts.append((target, samples, rate))
        if len(attempts) == 2 and last_succeeds:
            return original_play(target, samples, rate)
        return False

    monkeypatch.setattr(audio, "play", dispatch)
    if trigger == "release":
        window._release(listener)
    elif trigger == "focus_loss":
        focus_cycle(window)
    else:
        audio.playing.emit(listener, False)
    assert len(attempts) == 2, "A rejected dispatch must not stall the next translation"
    assert all(attempt[0] == listener for attempt in attempts)
    assert all(attempt[1] is item.audio for attempt, item in zip(attempts, pending))
    assert window._deferred == [[], []]
    assert window._playback[listener] is last_succeeds
    assert window.active and window.generation == generation
    assert "headphone playback unavailable" in window.status.text()
    assert window.cards[1 - listener].original.toPlainText().endswith("next")
    if trigger == "release":
        assert len(worker.submissions) == 1
        np.testing.assert_array_equal(worker.submissions[0][2], voice)
    else:
        assert not worker.submissions
    if last_succeeds:
        audio.playing.emit(listener, False)
        assert not window._playback[listener]
        assert len(attempts) == 2


@pytest.mark.parametrize("listener", [0, 1])
@pytest.mark.parametrize("action", ["stop", "text_only"])
def test_failed_dispatch_rechecks_lifecycle_before_next_attempt(
        gui, qtbot, monkeypatch, listener, action):
    window, audio, worker = gui
    start(window, worker, qtbot)
    window._press(listener)
    for _ in range(2):
        worker.result.emit(turn(window, speaker=1 - listener))
    attempts = []

    def dispatch(*args):
        attempts.append(args)
        if action == "stop":
            window.stop_session()
        else:
            window.text_only.setChecked(True)
        return False

    monkeypatch.setattr(audio, "play", dispatch)
    window._release(listener)
    assert len(attempts) == 1
    assert window._deferred == [[], []]
    assert not any(window._playback)
