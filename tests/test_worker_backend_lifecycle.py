"""Real worker-thread backend replacement at controlled completion boundaries."""
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from translitter.worker import ModelWorker
from test_worker_quality import ROUTES


@pytest.fixture
def backend_lifecycle(monkeypatch, qtbot, tmp_path):
    from translitter import worker as module

    state = SimpleNamespace(entered=Event(), release=Event(), calls=[], closed=[],
                            cancelled=[], outcome="success", latest="success", turns=[])

    class Engine:
        def __init__(self, *args):
            pass

        def load(self, progress):
            pass

        def set_translation_backend(self, backend, cancelled, progress):
            state.calls.append((backend, get_ident()))
            if len(state.calls) == 1:
                state.entered.set()
                if not state.release.wait(3):
                    raise AssertionError("test did not release backend preparation")
                state.cancelled.append(cancelled())
                outcome = state.outcome
            else:
                state.cancelled.append(cancelled())
                outcome = state.latest
            if outcome == "error":
                raise RuntimeError("private transcript /home/private/runtime")
            return outcome == "success"

        def interpret(self, samples, route, generation, cancelled, progress):
            state.turns.append((samples.copy(), route, generation, get_ident()))
            return None

        def close(self):
            state.closed.append(get_ident())

    monkeypatch.setattr(module, "ModelEngine", Engine)
    worker = ModelWorker(tmp_path)
    configured, ready, errors, settled = [], [], [], []
    worker.translation_ready.connect(lambda *args: configured.append(args))
    worker.ready.connect(lambda: ready.append(True))
    worker.error.connect(lambda *args: errors.append(args))
    worker.settled.connect(lambda *args: settled.append(args))
    h = SimpleNamespace(worker=worker, state=state, configured=configured, ready=ready,
                        errors=errors, settled=settled, qtbot=qtbot)
    yield h
    state.release.set()
    worker.shutdown()
    assert worker.wait(5000)


@pytest.mark.parametrize("obsolete_outcome", ["success", "false", "error"])
@pytest.mark.parametrize("latest_outcome", ["success", "false", "error"])
def test_replaced_inflight_completion_cannot_override_latest_backend(
        backend_lifecycle, obsolete_outcome, latest_outcome):
    h = backend_lifecycle
    h.state.outcome = obsolete_outcome
    h.state.latest = latest_outcome
    first = h.worker.set_translation_backend("translategemma")
    h.worker.start()
    h.qtbot.waitUntil(h.state.entered.is_set)
    h.qtbot.waitUntil(lambda: bool(h.ready))
    intermediate = h.worker.set_translation_backend("translategemma")
    latest = h.worker.set_translation_backend("marian")
    assert first < intermediate < latest
    with pytest.raises(ValueError, match="Translation backend is not ready"):
        h.worker.set_session(1, ROUTES)
    assert not h.worker.submit(1, 0, np.ones(800, dtype=np.float32))
    assert not h.configured
    h.state.release.set()
    h.qtbot.waitUntil(lambda: bool(h.configured))
    assert len(h.configured) == 1
    revision, backend, ready, error = h.configured[0]
    assert (revision, backend, ready) == (latest, "marian", latest_outcome == "success")
    assert bool(error) == (latest_outcome == "error")
    assert "private" not in error and "/home/" not in error
    assert [name for name, _ in h.state.calls] == ["translategemma", "marian"]
    assert h.state.cancelled == [True, False]
    assert not h.errors
    if latest_outcome != "success":
        with pytest.raises(ValueError, match="Translation backend is not ready"):
            h.worker.set_session(1, ROUTES)
        h.state.latest = "success"
        recovery = h.worker.set_translation_backend("marian")
        h.qtbot.waitUntil(lambda: len(h.configured) == 2)
        assert h.configured[-1] == (recovery, "marian", True, "")
    h.worker.set_session(1, ROUTES)
    audio = np.linspace(-0.1, 0.1, 800, dtype=np.float32)
    assert h.worker.submit(1, 1, audio)
    h.qtbot.waitUntil(lambda: bool(h.settled))
    assert h.settled == [(1, 1, False)]
    assert len(h.state.turns) == 1
    samples, route, generation, thread = h.state.turns[0]
    np.testing.assert_array_equal(samples, audio)
    assert route == ROUTES[1] and generation == 1
    assert thread != get_ident()
    h.worker.shutdown()
    assert h.worker.wait(1000)
    h.qtbot.wait(10)
    assert h.state.closed == [thread]
    assert all(tid == thread for _, tid in h.state.calls)


@pytest.mark.parametrize("outcome", ["success", "false", "error"])
def test_shutdown_discards_uncooperative_preflight_and_pending_replacement(
        backend_lifecycle, outcome):
    h = backend_lifecycle
    h.state.outcome = outcome
    h.worker.set_translation_backend("translategemma")
    h.worker.start()
    h.qtbot.waitUntil(h.state.entered.is_set)
    h.qtbot.waitUntil(lambda: bool(h.ready))
    h.worker.set_translation_backend("marian")
    h.worker.shutdown()
    h.worker.shutdown()
    assert not h.worker.wait(0)
    h.state.release.set()
    assert h.worker.wait(1000)
    h.qtbot.wait(10)
    assert [name for name, _ in h.state.calls] == ["translategemma"]
    assert h.state.cancelled == [True]
    assert not h.configured and not h.errors and not h.settled and not h.state.turns
    assert h.state.closed == [h.state.calls[0][1]]
    assert h.state.closed[0] != get_ident()
    assert h.worker._config_pending is None
    assert not h.worker._translation_ready
    with pytest.raises(ValueError):
        h.worker.set_translation_backend("marian")
    with pytest.raises(ValueError):
        h.worker.set_session(1, ROUTES)
    assert not h.worker.submit(1, 0, np.ones(800, dtype=np.float32))
