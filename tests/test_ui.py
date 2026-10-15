"""Native widgets with injectable backends; never capture or play hardware."""
from importlib.util import find_spec
from types import SimpleNamespace
import numpy as np
import pytest
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtWidgets import QPlainTextEdit
from translitter.core import Headset, make_routes


class FakeAudio(QObject):
    samples = Signal(int, object)
    level = Signal(int, float)
    failure = Signal(int, str)
    playing = Signal(int, bool)
    devices_changed = Signal()

    def __init__(self):
        super().__init__()
        self.pairs = [Headset("a", "mic-a", "out-a", "USB Alpha"),
                      Headset("b", "mic-b", "out-b", "USB Beta")]
        self.starts, self.plays, self.interrupts = [], [], []
        self.stops = 0
        self.volume = None

    def headsets(self): return self.pairs
    def start(self, routes): self.starts.append(routes)
    def stop(self): self.stops += 1
    def set_volume(self, value): self.volume = value
    def play(self, listener, samples, rate):
        self.plays.append((listener, samples, rate))
        self.playing.emit(listener, True)
        return True
    def stop_playback(self, listener):
        self.interrupts.append(listener)
        self.playing.emit(listener, False)


class FakeWorker(QObject):
    ready = Signal()
    loading = Signal(str)
    progress = Signal(int, str)
    result = Signal(object)
    settled = Signal(int, int, bool)
    error = Signal(int, str)

    def __init__(self):
        super().__init__()
        self.starts = self.shutdowns = 0
        self.sessions, self.stops, self.submissions = [], [], []
        self.accept = True
        self.finished = True
    def start(self): self.starts += 1
    def set_session(self, generation, routes): self.sessions.append((generation, routes))
    def stop_session(self, generation): self.stops.append(generation)
    def submit(self, generation, speaker, samples):
        self.submissions.append((generation, speaker, samples.copy()))
        return self.accept
    def shutdown(self): self.shutdowns += 1
    def wait(self, ms): return self.finished


@pytest.fixture
def gui(qtbot, tmp_path):
    assert find_spec("translitter.ui") is not None, "Native interpreter window is missing"
    from translitter.ui import Window
    audio, worker = FakeAudio(), FakeWorker()
    window = Window(tmp_path, worker=worker, audio=audio, start_worker=False)
    qtbot.addWidget(window)
    window.show()
    return window, audio, worker


def start(window, worker, qtbot):
    worker.ready.emit()
    qtbot.mouseClick(window.start_button, Qt.LeftButton)
    assert window.active


def test_native_launch_selection_and_explicit_routes(qtbot, request):
    assert find_spec("translitter.ui") is not None, "Native interpreter window is missing"
    w, audio, worker = request.getfixturevalue("gui")
    assert worker.starts == 0 and not audio.starts
    assert not w.start_button.isEnabled()
    assert w.mode.currentData() == "ptt"
    assert w.volume.value() == 45 and w.volume.maximum() == 80
    assert audio.volume == .45
    assert len(w.cards) == 2
    assert all(c.original.isReadOnly() and c.translated.isReadOnly() for c in w.cards)
    assert all(c.original.maximumBlockCount() == 100 for c in w.cards)
    assert "mic-a" in w.selectors[0].toolTip() and "out-a" in w.selectors[0].toolTip()
    worker.loading.emit("Loading local models")
    assert "Loading local models" in w.status.text()
    worker.ready.emit()
    assert w.start_button.isEnabled()
    w.selectors[1].setCurrentIndex(0)
    assert not w.start_button.isEnabled()
    w.selectors[1].setCurrentIndex(1)
    qtbot.mouseClick(w.swap_button, Qt.LeftButton)
    assert w.selectors[0].currentData().key == "b"
    expected = make_routes(audio.pairs[1], audio.pairs[0])
    qtbot.mouseClick(w.start_button, Qt.LeftButton)
    assert audio.starts == [expected] and worker.sessions[-1] == (w.generation, expected)
    assert not w.selectors[0].isEnabled() and w.stop_button.isEnabled()
    qtbot.mouseClick(w.stop_button, Qt.LeftButton)
    assert not w.active and audio.stops >= 1 and worker.stops[-1] == w.generation
    audio.pairs = []
    qtbot.mouseClick(w.refresh_button, Qt.LeftButton)
    assert w.selectors[0].count() == 0 and not w.start_button.isEnabled()


@pytest.mark.parametrize("automatic", [False, True])
def test_capture_only_eligible_speaker_and_bounded_turns(gui, qtbot, automatic):
    w, audio, worker = gui
    if automatic:
        w.mode.setCurrentIndex(1)
    start(w, worker, qtbot)
    voice = np.ones(8000, dtype=np.float32) * .12
    if not automatic:
        audio.samples.emit(0, voice)
        assert not worker.submissions
        qtbot.mousePress(w.cards[0].hold, Qt.LeftButton)
    audio.samples.emit(0, voice)
    if automatic:
        assert "Talking" in w.cards[0].state.text()
        audio.samples.emit(0, np.zeros(9600, dtype=np.float32))
    else:
        audio.samples.emit(1, voice)
        qtbot.mouseRelease(w.cards[0].hold, Qt.LeftButton)
    assert len(worker.submissions) == 1
    generation, speaker, samples = worker.submissions[0]
    assert generation == w.generation and speaker == 0 and len(samples) >= 8000
    assert "Busy" in w.cards[0].state.text()
    if not automatic:
        qtbot.keyPress(w.cards[1].original, Qt.Key_F2)
        audio.samples.emit(1, np.ones(128000, dtype=np.float32) * .12)
        qtbot.keyRelease(w.cards[1].original, Qt.Key_F2)
        assert len(worker.submissions) == 2
        assert worker.submissions[1][1] == 1 and len(worker.submissions[1][2]) == 128000
        assert "Busy" in w.cards[1].state.text(), "Eight-second automatic submit must retain busy state after release"
    worker.accept = False
    assert w.submit_audio(1, voice) is False
    assert "busy" in w.status.text().lower()
    count = len(worker.submissions)
    for invalid in (np.array([np.nan] * 8000), np.ones((2, 8000)), np.zeros(10)):
        assert w.submit_audio(0, invalid) is False
    assert len(worker.submissions) == count
    audio.level.emit(0, .08)
    assert w.cards[0].meter.value() > 0
    qtbot.mouseClick(w.stop_button, Qt.LeftButton)
    audio.samples.emit(0, voice)
    assert len(worker.submissions) == count


def turn(w, speaker=0, **changes):
    values = dict(generation=w.generation, speaker=speaker, listener=1-speaker,
                  source_language="fi" if speaker == 0 else "en",
                  target_language="en" if speaker == 0 else "fi",
                  original="<b>Hei</b>", translation="<script>Hello</script>",
                  audio=np.ones(8000, dtype=np.float32) * .1, rate=16000, elapsed=.5)
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize("automatic", [False, True])
def test_results_feedback_guard_interruption_and_stale_cancellation(gui, qtbot, automatic):
    w, audio, worker = gui
    if automatic:
        w.mode.setCurrentIndex(1)
    start(w, worker, qtbot)
    current = turn(w)
    worker.result.emit(current)
    assert w.cards[0].original.toPlainText() == current.original
    assert w.cards[0].translated.toPlainText() == current.translation
    assert audio.plays[0][0] == 1 and "Paused" in w.cards[1].state.text()
    voice = np.ones(8000, dtype=np.float32) * .12
    audio.samples.emit(1, voice)
    audio.samples.emit(1, np.zeros(9600, dtype=np.float32))
    assert not worker.submissions
    audio.playing.emit(1, False)
    audio.samples.emit(1, voice)
    audio.samples.emit(1, np.zeros(9600, dtype=np.float32))
    assert not worker.submissions
    if automatic:
        qtbot.wait(330)
        audio.samples.emit(1, voice)
        audio.samples.emit(1, np.zeros(9600, dtype=np.float32))
    else:
        worker.result.emit(current)
        qtbot.mousePress(w.cards[1].hold, Qt.LeftButton)
        assert audio.interrupts[-1] == 1
        audio.samples.emit(1, voice)
        qtbot.mouseRelease(w.cards[1].hold, Qt.LeftButton)
    assert len(worker.submissions) == 1 and worker.submissions[0][1] == 1
    qtbot.mouseClick(w.text_only, Qt.LeftButton)
    played = len(audio.plays)
    worker.result.emit(turn(w, speaker=1))
    assert len(audio.plays) == played
    before = w.cards[0].original.toPlainText()
    worker.result.emit(turn(w, generation=w.generation-1, original="stale"))
    worker.result.emit(turn(w, listener=0, original="wrong route"))
    assert w.cards[0].original.toPlainText() == before
    for _ in range(105):
        worker.result.emit(turn(w, original="<b>plain</b>"))
    assert w.cards[0].original.document().blockCount() == 100
    qtbot.mouseClick(w.stop_button, Qt.LeftButton)
    stopped = w.status.text()
    worker.result.emit(turn(w, original="inactive, even with same generation"))
    worker.error.emit(w.generation, "late error")
    worker.progress.emit(w.generation, "late progress")
    assert "inactive" not in w.cards[0].original.toPlainText()
    assert w.status.text() == stopped
    start(w, worker, qtbot)
    worker.result.emit(current)
    assert w.status.text() != "late progress" and len(audio.plays) == played


def focus_cycle(window):
    from PySide6.QtCore import QEvent
    from PySide6.QtWidgets import QApplication
    QApplication.sendEvent(window, QEvent(QEvent.WindowDeactivate))
    QApplication.sendEvent(window, QEvent(QEvent.WindowActivate))


@pytest.mark.parametrize("listener", [0, 1])
def test_focus_loss_cancels_partial_capture_and_drains_deferred(gui, qtbot, listener):
    w, audio, worker = gui
    start(w, worker, qtbot)
    generation = w.generation
    key = Qt.Key_F1 if listener == 0 else Qt.Key_F2
    qtbot.keyPress(w.cards[listener].original, key)
    audio.samples.emit(listener, np.full(8000, .12, dtype=np.float32))
    translated = turn(w, speaker=1-listener)
    worker.result.emit(translated)
    assert w._sizes[listener] == 8000 and len(w._deferred[listener]) == 1
    assert not audio.plays
    focus_cycle(w)
    qtbot.keyRelease(w.cards[listener].original, key)
    assert not worker.submissions, "Focus loss must cancel, not submit partial microphone audio"
    assert len(audio.plays) == 1
    assert audio.plays[0][0] == listener and audio.plays[0][1] is translated.audio
    assert w._deferred == [[], []] and w._held == [False, False]
    assert w._sizes == [0, 0] and w._chunks == [[], []]
    assert w.generation == generation and w.active
    assert all("release" not in card.state.text().lower() for card in w.cards)
    assert "cancel" in w.status.text().lower() and "focus" in w.status.text().lower()
    cancelled = w.status.text()
    focus_cycle(w)
    assert w.status.text() == cancelled and len(audio.plays) == 1


@pytest.mark.parametrize("listener", [0, 1])
def test_focus_loss_drains_two_deferred_in_fifo_order(gui, qtbot, listener):
    w, audio, worker = gui
    start(w, worker, qtbot)
    w._press(listener)
    audio.samples.emit(listener, np.full(8000, .12, dtype=np.float32))
    first, second = [turn(w, speaker=1-listener) for _ in range(2)]
    worker.result.emit(first)
    worker.result.emit(second)
    assert len(w._deferred[listener]) == 2 and not audio.plays
    focus_cycle(w)
    assert len(audio.plays) == 1 and audio.plays[0][1] is first.audio
    assert len(w._deferred[listener]) == 1
    cancelled = w.status.text()
    audio.playing.emit(listener, False)
    assert len(audio.plays) == 2 and audio.plays[1][1] is second.audio
    assert all(play[0] == listener for play in audio.plays)
    audio.playing.emit(listener, False)
    w._release(listener)
    assert w._deferred == [[], []] and not worker.submissions
    assert not w._playback[listener] and "release" not in w.cards[listener].state.text()
    assert "cancel" in cancelled.lower() and w.status.text() == cancelled


def test_focus_loss_preserves_queued_worker_turns_and_busy_counters(gui, qtbot):
    w, audio, worker = gui
    start(w, worker, qtbot)
    voice = np.full(8000, .12, dtype=np.float32)
    assert w.submit_audio(0, voice) and w.submit_audio(1, voice)
    generation, sessions = w.generation, list(worker.sessions)
    w._press(1)
    audio.samples.emit(1, voice)
    worker.result.emit(turn(w))
    assert w._busy == [0, 1]
    focus_cycle(w)
    w._release(1)
    assert w._busy == [0, 1] and len(worker.submissions) == 2
    assert w.generation == generation and worker.sessions == sessions and not worker.stops
    audio.playing.emit(1, False)
    assert "Busy" in w.cards[1].state.text()
    worker.result.emit(turn(w, speaker=1))
    assert w._busy == [0, 0] and len(audio.plays) == 2


def test_focus_loss_without_held_capture_preserves_status_and_cards(gui, qtbot):
    w, audio, worker = gui
    start(w, worker, qtbot)
    assert w.submit_audio(0, np.full(8000, .12, dtype=np.float32))
    status = w.status.text()
    states = [card.state.text() for card in w.cards]
    focus_cycle(w)
    assert w.status.text() == status and [card.state.text() for card in w.cards] == states
    assert w._busy == [1, 0] and len(worker.submissions) == 1 and not audio.plays


@pytest.mark.parametrize("listener", [0, 1])
def test_focus_loss_does_not_mask_headphone_playback_failure(gui, qtbot, listener, monkeypatch):
    w, audio, worker = gui
    start(w, worker, qtbot)
    w._press(listener)
    audio.samples.emit(listener, np.full(8000, .12, dtype=np.float32))
    worker.result.emit(turn(w, speaker=1-listener))
    attempts = []
    def unavailable(*args):
        attempts.append(args)
        assert "cancel" in w.status.text().lower()
        return False
    monkeypatch.setattr(audio, "play", unavailable)
    focus_cycle(w)
    w._release(listener)
    assert len(attempts) == 1 and attempts[0][0] == listener
    assert "headphone playback unavailable" in w.status.text()
    assert w._deferred == [[], []] and not worker.submissions


@pytest.mark.parametrize("listener", [0, 1])
@pytest.mark.parametrize("action", ["stop", "text_only", "close"])
def test_focus_loss_respects_stop_text_only_and_closing_guards(gui, qtbot, listener, action):
    w, audio, worker = gui
    start(w, worker, qtbot)
    w._press(listener)
    audio.samples.emit(listener, np.full(8000, .12, dtype=np.float32))
    old = turn(w, speaker=1-listener)
    worker.result.emit(old)
    assert len(w._deferred[listener]) == 1
    if action == "stop":
        w.stop_session()
    elif action == "text_only":
        w.text_only.setChecked(True)
    else:
        worker.finished = False
        w.close()
        assert w._closing and not w.active
    status = w.status.text()
    states = [card.state.text() for card in w.cards]
    focus_cycle(w)
    w._release(listener)
    audio.playing.emit(listener, False)
    worker.result.emit(old)
    assert not audio.plays and not worker.submissions and w._deferred == [[], []]
    if action == "text_only":
        assert "cancel" in w.status.text().lower() or "translated locally" in w.status.text()
        assert all("release" not in card.state.text().lower() for card in w.cards)
        w.text_only.setChecked(False)
        focus_cycle(w)
        assert not audio.plays
    else:
        assert w.status.text() == status and [card.state.text() for card in w.cards] == states
    worker.finished = True


def test_translation_during_active_ptt_preserves_exact_voice_until_release(gui, qtbot):
    w, audio, worker = gui
    start(w, worker, qtbot)
    voice = np.full(8000, .1, dtype=np.float32)
    w._press(1)
    audio.samples.emit(1, voice)
    assert w._sizes[1] == 8000
    translated = turn(w)
    worker.result.emit(translated)
    assert not audio.plays
    assert w._held[1] and w._sizes[1] == 8000
    w._release(1)
    assert len(worker.submissions) == 1
    assert worker.submissions[0][:2] == (w.generation, 1)
    np.testing.assert_array_equal(worker.submissions[0][2], voice)
    assert len(audio.plays) == 1 and audio.plays[0][1] is translated.audio


@pytest.mark.parametrize("listener", [0, 1])
def test_deferred_playback_is_bounded_fifo_through_release_and_followups(gui, qtbot, listener):
    w, audio, worker = gui
    start(w, worker, qtbot)
    voice = np.full(8000, .1, dtype=np.float32)
    w._press(listener)
    audio.samples.emit(listener, voice)
    first, second, overflow, followup = [turn(w, speaker=1-listener, original=name)
                                       for name in ("first", "second", "overflow", "followup")]
    for result in (first, second, overflow):
        worker.result.emit(result)
    assert "queue full" in w.status.text().lower()
    assert "text" in w.status.text().lower()
    assert "overflow" in w.cards[1-listener].original.toPlainText()
    assert len(w._deferred[listener]) == 2 and not audio.plays
    assert w._sizes[listener] == 8000
    # Release must submit the captured turn before any playback callback runs.
    original_play = audio.play
    def play_after_submit(*args):
        assert len(worker.submissions) == 1
        np.testing.assert_array_equal(worker.submissions[0][2], voice)
        return original_play(*args)
    audio.play = play_after_submit
    w._release(listener)
    assert len(audio.plays) == 1 and audio.plays[0][1] is first.audio
    worker.result.emit(followup)
    assert len(audio.plays) == 1
    for expected in (second, followup):
        audio.playing.emit(listener, False)
        assert audio.plays[-1][1] is expected.audio
    audio.playing.emit(listener, False)
    assert len(audio.plays) == 3 and not w._deferred[listener]
    audio.samples.emit(listener, voice)
    assert len(worker.submissions) == 1, "Playback feedback must not create a new turn"


@pytest.mark.parametrize("action", ["stop", "restart", "text_only", "close"])
def test_deferred_playback_cleared_by_panic_and_text_only(gui, qtbot, action):
    w, audio, worker = gui
    start(w, worker, qtbot)
    old = turn(w)
    w._press(1)
    audio.samples.emit(1, np.full(8000, .1, dtype=np.float32))
    worker.result.emit(old)
    assert len(w._deferred[1]) == 1
    if action == "text_only":
        w.text_only.setChecked(True)
        assert w._held[1] and w._sizes[1] == 8000
        w._release(1)
        assert len(worker.submissions) == 1
        w.text_only.setChecked(False)
    elif action == "close":
        w.close()
    else:
        # Real Stop emits playing(False); neither that nor late results may flush.
        original_stop = audio.stop
        def stop_with_signals():
            original_stop()
            audio.playing.emit(0, False)
            audio.playing.emit(1, False)
        audio.stop = stop_with_signals
        w.stop_session()
        if action == "restart":
            start(w, worker, qtbot)
    assert w._deferred == [[], []]
    audio.playing.emit(1, False)
    if action != "text_only":
        worker.result.emit(old)
    w._release(1)
    assert not audio.plays
    if action != "text_only":
        assert not worker.submissions


def test_deferred_interruption_prioritizes_ptt_and_ignores_late_playing_signals(gui, qtbot):
    w, audio, worker = gui
    start(w, worker, qtbot)
    first, second, third = [turn(w) for _ in range(3)]
    w._press(1)
    worker.result.emit(first)
    worker.result.emit(second)
    w._release(1)
    assert len(audio.plays) == 1
    w._press(1)
    assert audio.interrupts[-1] == 1 and len(audio.plays) == 1
    voice = np.full(8000, .1, dtype=np.float32)
    audio.samples.emit(1, voice)
    audio.playing.emit(1, True)
    audio.playing.emit(1, False)
    assert w._held[1] and w._sizes[1] == 8000 and not w._playback[1]
    assert len(audio.plays) == 1
    worker.result.emit(third)
    w._release(1)
    np.testing.assert_array_equal(worker.submissions[0][2], voice)
    assert audio.plays[-1][1] is second.audio
    audio.playing.emit(1, False)
    assert audio.plays[-1][1] is third.audio


def test_deferred_playback_waits_for_release_after_eight_second_ptt_submit(gui, qtbot):
    w, audio, worker = gui
    start(w, worker, qtbot)
    w._press(1)
    worker.result.emit(turn(w))
    audio.samples.emit(1, np.full(128000, .1, dtype=np.float32))
    audio.samples.emit(1, np.full(8000, .2, dtype=np.float32))
    assert len(worker.submissions) == 1 and not audio.plays and w._held[1]
    w._release(1)
    assert [len(item[2]) for item in worker.submissions] == [128000, 8000]
    assert len(audio.plays) == 1


def test_loading_audio_failure_disconnect_and_nonblocking_close(gui, qtbot):
    w, audio, worker = gui
    start(w, worker, qtbot)
    worker.error.emit(w.generation - 1, "stale")
    assert "stale" not in w.status.text()
    worker.progress.emit(w.generation, "Translating locally")
    assert w.status.text() == "Translating locally"
    worker.error.emit(w.generation, "Local model issue")
    assert "Local model issue" in w.status.text()
    audio.failure.emit(0, "Microphone disconnected")
    assert not w.active and "Microphone disconnected" in w.status.text()
    start(w, worker, qtbot)
    audio.pairs = audio.pairs[:1]
    audio.devices_changed.emit()
    assert not w.active and not w.start_button.isEnabled()
    worker.error.emit(-1, "Missing offline model")
    assert not w.models_ready and "Missing offline model" in w.status.text()
    worker.finished = False
    w.close()
    assert worker.shutdowns == 1 and w.isVisible()
    assert "Closing" in w.status.text() and not w.active
    worker.ready.emit()
    assert not w.start_button.isEnabled()
    worker.finished = True
    qtbot.waitUntil(lambda: not w.isVisible(), timeout=1000)
    assert worker.shutdowns == 1 and audio.stops >= 3


def test_worker_start_is_async_and_capture_requires_explicit_start(qtbot, tmp_path):
    from translitter.ui import Window
    audio, worker = FakeAudio(), FakeWorker()
    w = Window(tmp_path, worker=worker, audio=audio)
    qtbot.addWidget(w)
    assert worker.starts == 1 and not audio.starts
    worker.ready.emit()
    assert w.start_button.isEnabled()
    w.close()
    assert worker.shutdowns == 1


def test_cli_defaults_and_native_bootstrap(qapp, qtbot, monkeypatch, tmp_path):
    assert find_spec("translitter.__main__") is not None, "Module launcher is missing"
    import translitter.__main__ as entry
    defaults = entry.build_parser().parse_args([])
    assert defaults.model == "whisper-turbo" and defaults.device == "cuda"
    seen = {}
    class Worker(FakeWorker):
        def __init__(self, root, model_name, device):
            super().__init__()
            seen.update(root=root, model=model_name, device=device, worker=self)
    from translitter.ui import Window
    def create_window(root, worker):
        w = Window(root, worker=worker, audio=FakeAudio())
        seen["window"] = w
        qtbot.addWidget(w)
        from PySide6.QtCore import QTimer
        QTimer.singleShot(50, w.close)
        return w
    monkeypatch.setattr(entry, "ModelWorker", Worker)
    monkeypatch.setattr(entry, "Window", create_window)
    assert entry.main(["--model", "whisper-small", "--device", "cpu"]) == 0
    assert seen["model"] == "whisper-small" and seen["device"] == "cpu"
    assert seen["root"].name == "translitter" and seen["root"].is_dir()
    assert seen["worker"].starts == 1 and seen["worker"].shutdowns == 1
    assert not seen["window"].audio.starts


def test_close_retains_actual_running_qthread_until_finished(qtbot, tmp_path):
    from threading import Event
    from PySide6.QtCore import QThread, QTimer
    from translitter.ui import Window
    class ThreadWorker(QThread):
        ready = Signal()
        loading = Signal(str)
        progress = Signal(int, str)
        result = Signal(object)
        error = Signal(int, str)
        def __init__(self):
            super().__init__()
            self.done = Event()
        def run(self):
            self.ready.emit()
            self.done.wait(2)
        def stop_session(self, generation): pass
        def shutdown(self): QTimer.singleShot(100, self.done.set)
    worker = ThreadWorker()
    w = Window(tmp_path, worker=worker, audio=FakeAudio())
    qtbot.addWidget(w)
    w.show()
    qtbot.waitUntil(lambda: w.models_ready)
    w.close()
    assert w.isVisible() and worker.isRunning()
    qtbot.waitUntil(lambda: not w.isVisible(), timeout=1000)
    assert worker.wait(0) and not worker.isRunning()


def test_silent_and_failed_turns_release_busy_state_without_fake_caption(gui, qtbot):
    w, audio, worker = gui
    start(w, worker, qtbot)
    assert w.submit_audio(0, np.zeros(16000, dtype=np.float32))
    assert w._busy[0] == 1
    worker.settled.emit(w.generation, 0, False)
    assert w._busy[0] == 0
    assert "Busy" not in w.cards[0].state.text()
    assert not w.cards[0].original.toPlainText() and not audio.plays
    worker.settled.emit(w.generation - 1, 0, False)
    assert w._busy[0] == 0


def test_actual_worker_loading_failure_allows_safe_window_close(qtbot, tmp_path):
    from translitter.worker import ModelWorker
    from translitter.ui import Window
    worker = ModelWorker(tmp_path)
    w = Window(tmp_path, worker=worker, audio=FakeAudio())
    qtbot.addWidget(w)
    w.show()
    qtbot.waitUntil(lambda: "Model loading failed" in w.status.text())
    assert not w.start_button.isEnabled()
    w.close()
    qtbot.waitUntil(lambda: not w.isVisible())
    assert worker.wait(1000)

