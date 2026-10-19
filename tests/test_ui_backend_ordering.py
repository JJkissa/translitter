"""Native backend-request ordering probes; no models or physical audio."""
from itertools import permutations

import pytest
from PySide6.QtCore import QTimer

from test_ui_quality import quality_gui, select


def snapshot(window):
    return (window.status.text(), window.models_ready,
            window._translation_is_ready, window._translation_problem,
            window.start_button.isEnabled(), window.generation,
            tuple(card.state.text() for card in window.cards))


@pytest.mark.parametrize("order", tuple(permutations((1, 2, 3))))
@pytest.mark.parametrize("latest_ready", [False, True])
def test_rapid_backend_changes_only_accept_latest_completion(quality_gui, qtbot,
                                                            order, latest_ready):
    w, audio, worker = quality_gui
    worker.ready.emit()
    for backend in ("translategemma", "marian", "translategemma"):
        select(w, backend)
        assert not w.start_button.isEnabled()
        w.start_session()
        assert not w.active and not audio.starts and not worker.sessions
    assert worker.requests == [(1, "translategemma"), (2, "marian"),
                               (3, "translategemma")]
    delivered = []

    def deliver():
        for request in order:
            before = snapshot(w)
            backend = worker.requests[request - 1][1]
            ready = latest_ready if request == 3 else not latest_ready
            worker.translation_ready.emit(request, backend, ready,
                                          "current unavailable" if request == 3 else "obsolete error")
            if request != 3:
                assert snapshot(w) == before
            delivered.append(request)

    QTimer.singleShot(0, deliver)
    qtbot.waitUntil(lambda: len(delivered) == 3)
    assert w._translation_is_ready is latest_ready
    assert w.start_button.isEnabled() is latest_ready
    assert w._translation_problem == ("" if latest_ready else "current unavailable")
    assert "obsolete" not in w.status.text()
    assert not audio.starts and not audio.plays and not worker.sessions
    if not latest_ready:
        w.start_session()
        assert not w.active and not audio.starts and not worker.sessions
        select(w, "marian")
        assert worker.requests[-1] == (4, "marian")
        worker.translation_ready.emit(3, "translategemma", True, "")
        assert not w.start_button.isEnabled()
        worker.translation_ready.emit(4, "marian", True, "")
    w.start_session()
    assert w.active and len(audio.starts) == len(worker.sessions) == 1
    before = snapshot(w)
    worker.translation_ready.emit(3, "translategemma", False, "late failure")
    assert snapshot(w) == before and w.active
    w.stop_session()


@pytest.mark.parametrize("base_ready_first", [False, True])
@pytest.mark.parametrize("backend_ready", [False, True])
def test_base_and_backend_readiness_require_both_gates(quality_gui,
                                                      base_ready_first, backend_ready):
    w, audio, worker = quality_gui
    select(w, "translategemma")
    if base_ready_first:
        worker.ready.emit()
    else:
        worker.translation_ready.emit(1, "translategemma", backend_ready, "backend unavailable")
    assert not w.start_button.isEnabled()
    w.start_session()
    assert not w.active and not audio.starts and not worker.sessions
    if base_ready_first:
        worker.translation_ready.emit(1, "translategemma", backend_ready, "backend unavailable")
    else:
        worker.ready.emit()
    assert w.models_ready
    assert w.start_button.isEnabled() is backend_ready
    if not backend_ready:
        assert "backend unavailable" in w.status.text()
    assert not audio.starts


def test_request_exception_stays_disabled_and_recovers_on_new_request(quality_gui, monkeypatch):
    w, audio, worker = quality_gui
    worker.ready.emit()
    original = worker.set_translation_backend

    def unavailable(backend):
        raise RuntimeError("injected request failure")

    monkeypatch.setattr(worker, "set_translation_backend", unavailable)
    select(w, "translategemma")
    assert not w.start_button.isEnabled()
    assert w._translation_request is None
    before = snapshot(w)
    worker.translation_ready.emit(1, "translategemma", True, "")
    assert snapshot(w) == before
    worker.ready.emit()
    w.start_session()
    assert not audio.starts and not worker.sessions
    assert "unavailable" in w.status.text()
    monkeypatch.setattr(worker, "set_translation_backend", original)
    select(w, "marian")
    worker.translation_ready.emit(1, "marian", True, "")
    assert w.start_button.isEnabled()
    w.start_session()
    assert w.active and len(audio.starts) == 1
    w.stop_session()


@pytest.mark.parametrize("ready", [False, True])
def test_close_during_backend_loading_ignores_queued_completions(quality_gui, qtbot, ready):
    w, audio, worker = quality_gui
    worker.ready.emit()
    select(w, "translategemma")
    select(w, "marian")
    worker.finished = False
    w.close()
    before = snapshot(w)
    delivered = []

    def deliver():
        worker.translation_ready.emit(1, "translategemma", not ready, "obsolete")
        worker.translation_ready.emit(2, "marian", ready, "late unavailable")
        worker.loading.emit("late loading")
        worker.ready.emit()
        worker.error.emit(-1, "late base error")
        delivered.append(True)

    try:
        QTimer.singleShot(0, deliver)
        qtbot.waitUntil(lambda: bool(delivered))
        assert snapshot(w) == before
        assert not w.start_button.isEnabled() and not w.backend_selector.isEnabled()
        select(w, "translategemma")
        assert w.backend_selector.currentData() == "marian"
        assert len(worker.requests) == 2
        w.start_session()
        assert not w.active and not audio.starts and not audio.plays
        assert worker.shutdowns == 1
    finally:
        worker.finished = True
        w._poll_close()
    assert not w._close_poll.isActive() and not w.isVisible()
