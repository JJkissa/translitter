"""Initial model ownership and notifications on a real Qt worker thread."""
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from translitter.worker import ModelWorker
from test_worker_quality import ROUTES


@pytest.fixture
def initial_loading(monkeypatch, qtbot, tmp_path):
    from translitter import worker as module

    state = SimpleNamespace(entered=Event(), release=Event(), phase="load",
                            outcome="success", close_error=False, calls=[], turns=[])

    def gate(phase):
        if state.phase == phase:
            state.entered.set()
            assert state.release.wait(3), "initial model stage was not released"
            if state.outcome == "error":
                raise RuntimeError("private model details /home/private/model")

    class Engine:
        def __init__(self, *args):
            state.calls.append(("construct", get_ident()))
            gate("construct")

        def load(self, progress):
            state.calls.append(("load", get_ident()))
            progress("Initial stage")
            gate("load")
            progress("Final stage")

        def set_translation_backend(self, backend, cancelled, progress):
            state.calls.append(("configure", get_ident()))
            return not cancelled()

        def interpret(self, samples, route, generation, cancelled, progress):
            state.calls.append(("interpret", get_ident()))
            state.turns.append((generation, route, samples.copy()))
            return None

        def close(self):
            state.calls.append(("close", get_ident()))
            if state.close_error:
                raise RuntimeError("private cleanup details")

    monkeypatch.setattr(module, "ModelEngine", Engine)
    worker = ModelWorker(tmp_path)
    events = []
    worker.loading.connect(lambda text: events.append(("loading", text)))
    worker.ready.connect(lambda: events.append(("ready",)))
    worker.error.connect(lambda *args: events.append(("error", *args)))
    worker.translation_ready.connect(lambda *args: events.append(("configured", *args)))
    worker.settled.connect(lambda *args: events.append(("settled", *args)))
    h = SimpleNamespace(worker=worker, state=state, events=events, qtbot=qtbot)
    yield h
    state.release.set()
    worker.shutdown()
    assert worker.wait(5000)
    qtbot.wait(1)


def assert_terminal(h):
    assert h.worker._closed and h.worker._routes is None
    assert h.worker._config_pending is None and not h.worker._translation_ready
    assert all(not queue for queue in h.worker._pending)
    with pytest.raises(ValueError):
        h.worker.set_session(3, ROUTES)
    with pytest.raises(ValueError):
        h.worker.set_translation_backend("marian")
    assert not h.worker.submit(3, 0, np.ones(160, dtype=np.float32))
    assert len({thread for _, thread in h.state.calls}) == 1
    assert h.state.calls[0][1] != get_ident()


@pytest.mark.parametrize("close_error", [False, True])
def test_shutdown_during_construction_skips_loading_and_closes_owned_engine(
        initial_loading, close_error):
    h = initial_loading
    h.state.phase = "construct"
    h.state.close_error = close_error
    h.worker.set_translation_backend("translategemma")
    h.worker.start()
    h.qtbot.waitUntil(h.state.entered.is_set)
    h.worker.shutdown()
    h.worker.shutdown()
    h.state.release.set()
    assert h.worker.wait(1000)
    h.qtbot.wait(10)
    assert [name for name, _ in h.state.calls] == ["construct", "close"]
    assert not h.events
    assert_terminal(h)


@pytest.mark.parametrize("close_error", [False, True])
def test_shutdown_during_loading_suppresses_late_progress(initial_loading, close_error):
    h = initial_loading
    h.state.close_error = close_error
    h.worker.set_translation_backend("translategemma")
    h.worker.start()
    h.qtbot.waitUntil(h.state.entered.is_set)
    h.qtbot.waitUntil(lambda: bool(h.events))
    assert h.events == [("loading", "Initial stage")]
    h.worker.shutdown()
    h.state.release.set()
    assert h.worker.wait(1000)
    h.qtbot.wait(10)
    assert h.events == [("loading", "Initial stage")]
    assert [name for name, _ in h.state.calls] == ["construct", "load", "close"]
    assert_terminal(h)


@pytest.mark.parametrize("phase", ["construct", "load"])
@pytest.mark.parametrize("pending", ["turns", "configuration"])
@pytest.mark.parametrize("close_error", [False, True])
def test_initial_failure_is_terminal_and_discards_pending_work(
        initial_loading, phase, pending, close_error):
    h = initial_loading
    h.state.phase = phase
    h.state.outcome = "error"
    h.state.close_error = close_error
    if pending == "configuration":
        h.worker.set_translation_backend("translategemma")
        with pytest.raises(ValueError, match="Translation backend is not ready"):
            h.worker.set_session(1, ROUTES)
        assert not h.worker.submit(1, 0, np.ones(160, dtype=np.float32))
    else:
        # Preserve legacy callers that enqueue before baseline load.
        h.worker.set_session(1, ROUTES)
        for speaker in (0, 1):
            assert h.worker.submit(1, speaker, np.ones(160, dtype=np.float32))
    h.worker.start()
    h.qtbot.waitUntil(h.state.entered.is_set)
    h.state.release.set()
    assert h.worker.wait(1000)
    h.qtbot.wait(10)
    expected = [] if phase == "construct" else [("loading", "Initial stage")]
    assert h.events == expected + [("error", -1, "Model loading failed (RuntimeError)")]
    expected_calls = ["construct"] if phase == "construct" else ["construct", "load", "close"]
    assert [name for name, _ in h.state.calls] == expected_calls
    assert not h.state.turns
    assert_terminal(h)


@pytest.mark.parametrize("phase", ["construct", "load"])
def test_shutdown_suppresses_initial_failure(initial_loading, phase):
    h = initial_loading
    h.state.phase = phase
    h.state.outcome = "error"
    h.worker.start()
    h.qtbot.waitUntil(h.state.entered.is_set)
    if phase == "load":
        h.qtbot.waitUntil(lambda: bool(h.events))
    h.worker.shutdown()
    h.state.release.set()
    assert h.worker.wait(1000)
    h.qtbot.wait(10)
    assert h.events == ([] if phase == "construct" else [("loading", "Initial stage")])
    assert [name for name, _ in h.state.calls] == (
        ["construct"] if phase == "construct" else ["construct", "load", "close"])
    assert_terminal(h)


@pytest.mark.parametrize("speaker", [0, 1])
def test_successful_initial_load_publishes_ready_before_queued_turn(initial_loading, speaker):
    h = initial_loading
    audio = np.linspace(-0.2, 0.2, 160, dtype=np.float32)
    h.worker.set_session(1, ROUTES)
    assert h.worker.submit(1, speaker, audio)
    h.worker.start()
    h.qtbot.waitUntil(h.state.entered.is_set)
    h.qtbot.waitUntil(lambda: bool(h.events))
    assert not h.state.turns
    assert h.events == [("loading", "Initial stage")]
    h.state.release.set()
    h.qtbot.waitUntil(lambda: any(event[0] == "settled" for event in h.events))
    assert h.events == [("loading", "Initial stage"), ("loading", "Final stage"),
                        ("ready",), ("settled", 1, speaker, False)]
    assert len(h.state.turns) == 1
    generation, route, accepted_audio = h.state.turns[0]
    assert generation == 1 and route == ROUTES[speaker]
    np.testing.assert_array_equal(accepted_audio, audio)
    h.worker.shutdown()
    assert h.worker.wait(1000)
    assert [name for name, _ in h.state.calls] == ["construct", "load", "interpret", "close"]
    assert_terminal(h)


def test_shutdown_before_thread_start_constructs_nothing(initial_loading):
    h = initial_loading
    h.worker.set_translation_backend("translategemma")
    h.worker.shutdown()
    h.worker.start()
    assert h.worker.wait(1000)
    h.qtbot.wait(10)
    assert not h.events and not h.state.calls
    with pytest.raises(ValueError):
        h.worker.set_session(1, ROUTES)
    assert not h.worker.submit(1, 0, np.ones(160, dtype=np.float32))
