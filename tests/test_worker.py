"""Real Qt thread scheduling with tiny fake model stages; no audio devices."""
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

from translitter.core import Route
from translitter.engine import TurnResult

ROUTES = (Route(0, 1, "fi", "en", "mic0", "out1"),
          Route(1, 0, "en", "fi", "mic1", "out0"))


@pytest.fixture
def harness(monkeypatch, qtbot, tmp_path):
    from translitter import worker as module
    state = SimpleNamespace(loads=0, turns=[], gate=None, entered=Event(),
                            load_error=None, turn_error=None, hook=lambda: None,
                            constructed=[])
    class Engine:
        def __init__(self, root, model_name, device):
            state.constructed.append((root, model_name, device))
        def load(self, progress):
            state.loads += 1
            progress("Local models")
            if state.load_error:
                raise RuntimeError(state.load_error)
        def interpret(self, samples, route, generation, cancelled, progress):
            state.turns.append((generation, route.speaker, samples.copy()))
            state.entered.set()
            if state.gate is not None:
                assert state.gate.wait(5), "test model gate timed out"
            state.hook()
            progress("Stage complete")
            if state.turn_error:
                raise RuntimeError(state.turn_error)
            # Deliberately return even on cancellation: worker is an independent guard.
            return TurnResult(generation, route.speaker, route.listener,
                              route.source_language, route.target_language,
                              "Original", "Translation", samples, 22050, 0.1)
    monkeypatch.setattr(module, "ModelEngine", Engine)
    instance = module.ModelWorker(tmp_path)
    results, errors, progress, loading, ready = [], [], [], [], []
    instance.result.connect(results.append)
    instance.error.connect(lambda gen, text: errors.append((gen, text)))
    instance.progress.connect(lambda gen, text: progress.append((gen, text)))
    instance.loading.connect(loading.append)
    instance.ready.connect(lambda: ready.append(True))
    yield SimpleNamespace(worker=instance, state=state, results=results, errors=errors,
                          progress=progress, loading=loading, ready=ready, qtbot=qtbot)
    if state.gate:
        state.gate.set()
    instance.shutdown()
    assert instance.wait(5000), "worker failed to join"
    qtbot.wait(1)


def test_thread_loads_once_then_emits_routed_result(harness):
    h = harness
    h.worker.set_session(1, ROUTES)
    h.worker.start()
    h.qtbot.waitUntil(lambda: bool(h.ready))
    assert h.worker.submit(1, 0, np.ones(160, dtype=np.float32))
    h.qtbot.waitUntil(lambda: len(h.results) == 1)
    assert h.results[0].generation == 1 and h.results[0].listener == 1
    assert h.state.loads == 1 and h.loading == ["Local models"]
    assert h.progress == [(1, "Stage complete")]
    assert not h.errors


def test_pending_queue_rejects_third_turn_without_displacing_accepted_turns(harness):
    h = harness
    h.worker.set_session(1, ROUTES)
    for speaker in (0, 1):
        assert h.worker.submit(1, speaker, np.array([1], dtype=np.float32))
        assert h.worker.submit(1, speaker, np.array([2], dtype=np.float32))
        assert not h.worker.submit(1, speaker, np.array([3], dtype=np.float32))
    h.worker.start()
    h.qtbot.waitUntil(lambda: len(h.results) == 4)
    assert sorted(float(turn.audio[0]) for turn in h.results) == [1, 1, 2, 2]


def test_busy_speakers_are_served_round_robin(harness):
    h = harness
    h.worker.set_session(1, ROUTES)
    for speaker in (0, 0, 1, 1):
        assert h.worker.submit(1, speaker, np.ones(160, dtype=np.float32))
    h.worker.start()
    h.qtbot.waitUntil(lambda: len(h.results) == 4)
    assert [turn.speaker for turn in h.results] == [0, 1, 0, 1]


def test_stop_discards_active_and_queued_turns_and_stale_progress(harness):
    h = harness
    h.state.gate = Event()
    h.worker.set_session(1, ROUTES)
    h.worker.start()
    assert h.worker.submit(1, 0, np.ones(160, dtype=np.float32))
    h.qtbot.waitUntil(h.state.entered.is_set)
    assert h.worker.submit(1, 1, np.ones(160, dtype=np.float32))
    assert hasattr(h.worker, "stop_session"), "Session cancellation is missing"
    h.worker.stop_session(2)
    h.state.gate.set()
    h.qtbot.wait(60)
    assert not h.results and not h.progress
    assert len(h.state.turns) == 1
    assert not h.worker.submit(1, 0, np.ones(160, dtype=np.float32))
    assert not h.worker.submit(2, 0, np.ones(160, dtype=np.float32))
    h.worker.set_session(3, ROUTES)
    assert h.worker.submit(3, 1, np.ones(160, dtype=np.float32))
    h.qtbot.waitUntil(lambda: len(h.results) == 1)
    assert h.results[0].generation == 3


@pytest.mark.parametrize("speaker", [-1, 2, True, "0"])
def test_invalid_speakers_are_rejected_not_indexed(harness, speaker):
    h = harness
    h.worker.set_session(1, ROUTES)
    assert not h.worker.submit(1, speaker, np.ones(160, dtype=np.float32))


def test_stale_unconfigured_closed_and_bad_audio_submissions_are_rejected(harness):
    h = harness
    samples = np.ones(160, dtype=np.float32)
    assert not h.worker.submit(1, 0, samples)
    h.worker.set_session(1, ROUTES)
    assert not h.worker.submit(0, 0, samples)
    for invalid in (np.array([]), np.zeros((2, 2)), np.array([np.nan]), np.zeros(128001)):
        assert not h.worker.submit(1, 0, invalid)
    h.worker.shutdown()
    assert not h.worker.submit(1, 0, samples)
    with pytest.raises(ValueError):
        h.worker.set_session(2, ROUTES)


def test_generation_cannot_be_reused_and_route_contract_cannot_be_bypassed(harness):
    h = harness
    h.worker.set_session(2, ROUTES)
    with pytest.raises(ValueError):
        h.worker.set_session(2, ROUTES)
    with pytest.raises(ValueError):
        h.worker.set_session(3, (ROUTES[0], ROUTES[0]))


def test_loading_failure_is_signalled_and_thread_exits(harness):
    h = harness
    h.state.load_error = "model missing"
    h.worker.start()
    h.qtbot.waitUntil(lambda: bool(h.errors))
    assert h.worker.wait(1000)
    assert h.errors == [(-1, "Model loading failed (RuntimeError)")]
    assert not h.ready


def test_inference_failure_is_signalled_without_killing_future_turns(harness):
    h = harness
    h.worker.set_session(1, ROUTES)
    h.state.turn_error = "stage failed"
    h.worker.start()
    assert h.worker.submit(1, 0, np.ones(160, dtype=np.float32))
    h.qtbot.waitUntil(lambda: bool(h.errors))
    assert h.errors == [(1, "Turn processing failed (RuntimeError)")]
    assert not h.results
    h.state.turn_error = None
    assert h.worker.submit(1, 1, np.ones(160, dtype=np.float32))
    h.qtbot.waitUntil(lambda: bool(h.results))


def test_shutdown_cancels_inflight_model_callback_and_suppresses_output(harness):
    h = harness
    h.state.gate = Event()
    h.worker.set_session(1, ROUTES)
    h.worker.start()
    assert h.worker.submit(1, 0, np.ones(160, dtype=np.float32))
    h.qtbot.waitUntil(h.state.entered.is_set)
    h.worker.shutdown()
    h.state.gate.set()
    assert h.worker.wait(1000)
    h.qtbot.wait(10)
    assert not h.results and not h.progress


def test_stop_after_closed_worker_is_safe_for_window_teardown(harness):
    h = harness
    h.worker.shutdown()
    h.worker.stop_session(1)
    assert not h.worker.submit(1, 0, np.ones(160, dtype=np.float32))


def test_silent_turn_settles_without_emitting_a_translation(harness, monkeypatch):
    h = harness
    from translitter import worker as module
    class SilentEngine:
        def __init__(self, *args): pass
        def load(self, progress): pass
        def interpret(self, *args): return None
    monkeypatch.setattr(module, "ModelEngine", SilentEngine)
    assert hasattr(h.worker, "settled"), "Empty turns must release GUI busy state"
    settled = []
    h.worker.settled.connect(lambda generation, speaker, produced: settled.append((generation, speaker, produced)))
    h.worker.set_session(1, ROUTES)
    h.worker.start()
    assert h.worker.submit(1, 0, np.ones(160, dtype=np.float32))
    h.qtbot.waitUntil(lambda: bool(settled))
    assert settled == [(1, 0, False)] and not h.results

