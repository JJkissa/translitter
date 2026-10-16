"""Optional translation preflight uses the real serial Qt worker, never models."""
from threading import Event, get_ident
from types import SimpleNamespace

import pytest

from translitter.core import Route
from translitter.worker import ModelWorker

ROUTES = (Route(0, 1, "fi", "en", "mic0", "out1"),
          Route(1, 0, "en", "fi", "mic1", "out0"))


@pytest.fixture
def quality_worker(monkeypatch, qtbot, tmp_path):
    from translitter import worker as module
    state = SimpleNamespace(calls=[], closed=[], entered=Event(), gate=None,
                            error=None, load_error=False, cancellations=[], turns=[])

    class Engine:
        def __init__(self, *args): pass
        def load(self, progress):
            if state.load_error:
                raise RuntimeError("private loading details")
        def set_translation_backend(self, backend, cancelled, progress):
            state.calls.append((backend, get_ident()))
            state.entered.set()
            if state.gate is not None and backend == "translategemma":
                while not state.gate.wait(.01):
                    if cancelled():
                        state.cancellations.append(backend)
                        return False
            if state.error and backend == "translategemma":
                raise state.error
            return not cancelled()
        def interpret(self, *args):
            state.turns.append(get_ident())
            return None
        def close(self): state.closed.append(get_ident())

    monkeypatch.setattr(module, "ModelEngine", Engine)
    worker = ModelWorker(tmp_path)
    ready, configured, errors = [], [], []
    worker.ready.connect(lambda: ready.append(True))
    if hasattr(worker, "translation_ready"):
        worker.translation_ready.connect(lambda *args: configured.append(args))
    worker.error.connect(lambda *args: errors.append(args))
    yield SimpleNamespace(worker=worker, state=state, ready=ready,
                          configured=configured, errors=errors, qtbot=qtbot)
    if state.gate:
        state.gate.set()
    worker.shutdown()
    assert worker.wait(5000)


def test_preflight_is_serial_async_and_guards_session_until_confirmed(quality_worker):
    h = quality_worker
    h.state.gate = Event()
    caller = get_ident()
    request = h.worker.set_translation_backend("translategemma")
    assert type(request) is int
    assert not h.state.calls
    with pytest.raises(ValueError, match="Translation"):
        h.worker.set_session(1, ROUTES)
    h.worker.start()
    h.qtbot.waitUntil(h.state.entered.is_set)
    assert h.state.calls == [("translategemma", h.state.calls[0][1])]
    assert h.state.calls[0][1] != caller
    assert not h.configured
    with pytest.raises(ValueError, match="Translation"):
        h.worker.set_session(1, ROUTES)
    h.state.gate.set()
    h.qtbot.waitUntil(lambda: bool(h.configured))
    assert h.configured == [(request, "translategemma", True, "")]
    h.worker.set_session(1, ROUTES)
    with pytest.raises(ValueError):
        h.worker.set_translation_backend("marian")


@pytest.mark.parametrize("load_error", [False, True])
def test_engine_resources_close_on_worker_exit_and_loading_failure(quality_worker, load_error):
    h = quality_worker
    h.state.load_error = load_error
    h.worker.start()
    h.qtbot.waitUntil(lambda: bool(h.errors if load_error else h.ready))
    h.worker.shutdown()
    assert h.worker.wait(1000)
    assert len(h.state.closed) == 1
    assert h.state.closed[0] != get_ident()
    with pytest.raises(ValueError):
        h.worker.set_translation_backend("marian")


def test_obsolete_preflight_is_cancelled_and_pending_requests_coalesce(quality_worker):
    h = quality_worker
    h.state.gate = Event()
    first = h.worker.set_translation_backend("translategemma")
    obsolete = h.worker.set_translation_backend("marian")
    third = h.worker.set_translation_backend("translategemma")
    assert first < obsolete < third
    h.worker.start()
    h.qtbot.waitUntil(h.state.entered.is_set)
    latest = h.worker.set_translation_backend("marian")
    h.qtbot.waitUntil(lambda: bool(h.configured))
    assert latest > third
    assert h.state.cancellations == ["translategemma"]
    assert [backend for backend, _ in h.state.calls] == ["translategemma", "marian"]
    assert h.configured == [(latest, "marian", True, "")]
    h.worker.set_session(1, ROUTES)


def test_shutdown_cancels_preflight_without_emitting_ready(quality_worker):
    h = quality_worker
    h.state.gate = Event()
    h.worker.set_translation_backend("translategemma")
    h.worker.start()
    h.qtbot.waitUntil(h.state.entered.is_set)
    h.worker.shutdown()
    assert h.worker.wait(1000)
    assert h.state.cancellations == ["translategemma"]
    assert h.state.closed and not h.configured


def test_configuration_failure_is_sanitized_and_marian_recovers(quality_worker):
    h = quality_worker
    h.state.error = RuntimeError("secret transcript /home/private/model")
    request = h.worker.set_translation_backend("translategemma")
    h.worker.start()
    h.qtbot.waitUntil(lambda: bool(h.configured))
    revision, backend, ready, error = h.configured[0]
    assert (revision, backend, ready) == (request, "translategemma", False)
    assert error and "secret" not in error and "/home/" not in error
    with pytest.raises(ValueError, match="Translation"):
        h.worker.set_session(1, ROUTES)
    latest = h.worker.set_translation_backend("marian")
    h.qtbot.waitUntil(lambda: len(h.configured) == 2)
    assert h.configured[-1] == (latest, "marian", True, "")
    h.worker.set_session(1, ROUTES)
