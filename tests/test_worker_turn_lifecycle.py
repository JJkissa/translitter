"""Accepted-turn settlement and queue recovery on a real model thread."""
from threading import Event, get_ident
from types import SimpleNamespace

import numpy as np
import pytest

from translitter.engine import TurnResult
from translitter.worker import ModelWorker
from test_worker import ROUTES


@pytest.fixture
def turn_lifecycle(monkeypatch, qtbot, tmp_path):
    from translitter import worker as module

    state = SimpleNamespace(entered=Event(), release=Event(), calls=[], closed=[],
                            cancelled=[], outcome="success")

    class Engine:
        def __init__(self, *args):
            pass

        def load(self, progress):
            pass

        def interpret(self, samples, route, generation, cancelled, progress):
            first = not state.calls
            state.calls.append((generation, route, samples.copy(), get_ident()))
            if first:
                state.entered.set()
                assert state.release.wait(3), "first turn was not released"
            state.cancelled.append(cancelled())
            progress("Stage complete")
            if first and state.outcome == "error":
                raise RuntimeError("private transcript /home/private/model")
            if first and state.outcome == "empty":
                return None
            # Ignore cancellation deliberately: the worker must guard publication.
            return TurnResult(generation, route.speaker, route.listener,
                              route.source_language, route.target_language,
                              "Original", "Translation", samples.copy(), 22050, 0.1)

        def close(self):
            state.closed.append(get_ident())

    monkeypatch.setattr(module, "ModelEngine", Engine)
    worker = ModelWorker(tmp_path)
    events = []
    worker.result.connect(lambda turn: events.append(("result", turn)))
    worker.error.connect(lambda *args: events.append(("error", args)))
    worker.settled.connect(lambda *args: events.append(("settled", args)))
    worker.progress.connect(lambda *args: events.append(("progress", args)))
    h = SimpleNamespace(worker=worker, state=state, events=events, qtbot=qtbot)
    yield h
    state.release.set()
    worker.shutdown()
    assert worker.wait(5000)
    qtbot.wait(1)


def audio(label):
    return np.full(160, label / 10, dtype=np.float32)


def begin(h, speaker, outcome):
    h.state.outcome = outcome
    h.worker.set_session(1, ROUTES)
    h.worker.start()
    assert h.worker.submit(1, speaker, audio(1))
    h.qtbot.waitUntil(h.state.entered.is_set)
    # Both participants retain accepted work while the first call is blocked.
    for who, label in ((speaker, 2), (1 - speaker, 3), (speaker, 4), (1 - speaker, 5)):
        assert h.worker.submit(1, who, audio(label))
    assert not h.worker.submit(1, speaker, audio(6))
    assert not h.worker.submit(1, 1 - speaker, audio(7))
    assert not h.events


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("outcome", ["success", "empty", "error"])
def test_first_turn_outcome_settles_once_and_preserves_fair_pending_work(
        turn_lifecycle, speaker, outcome):
    h = turn_lifecycle
    begin(h, speaker, outcome)
    h.state.release.set()
    h.qtbot.waitUntil(lambda: sum(kind == "settled" for kind, _ in h.events) == 5)
    expected_speakers = [speaker, 1 - speaker, speaker, 1 - speaker, speaker]
    labels = [1, 3, 2, 5, 4]
    assert [(gen, route.speaker) for gen, route, _, _ in h.state.calls] == [
        (1, who) for who in expected_speakers]
    for (_, route, samples, thread), label in zip(h.state.calls, labels):
        assert route == ROUTES[route.speaker]
        np.testing.assert_array_equal(samples, audio(label))
        assert thread != get_ident()
    expected = []
    for index, who in enumerate(expected_speakers):
        expected.append(("progress", (1, "Stage complete")))
        if index == 0 and outcome == "error":
            expected.append(("error", (1, "Turn processing failed (RuntimeError)")))
        produced = index != 0 or outcome == "success"
        if produced:
            expected.append(("result", (1, who, 1 - who)))
        expected.append(("settled", (1, who, produced)))
    actual = [(kind, (value.generation, value.speaker, value.listener)
               if kind == "result" else value) for kind, value in h.events]
    assert actual == expected
    assert h.state.cancelled == [False] * 5
    assert all(not queue for queue in h.worker._pending)
    h.worker.shutdown()
    assert h.worker.wait(1000)
    assert h.state.closed == [h.state.calls[0][3]]


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("outcome", ["success", "empty", "error"])
@pytest.mark.parametrize("transition", ["stop", "restart", "shutdown"])
def test_retired_inflight_outcome_cannot_publish_or_consume_old_pending_work(
        turn_lifecycle, speaker, outcome, transition):
    h = turn_lifecycle
    begin(h, speaker, outcome)
    if transition == "shutdown":
        h.worker.shutdown()
    else:
        h.worker.stop_session(2)
        if transition == "restart":
            h.worker.set_session(3, ROUTES)
            for who in (1, 0):
                assert h.worker.submit(3, who, audio(8 + who))
    h.state.release.set()
    if transition == "shutdown":
        assert h.worker.wait(1000)
        h.qtbot.wait(10)
        assert not h.events
        assert len(h.state.calls) == 1
        assert h.state.cancelled == [True]
        assert h.state.closed == [h.state.calls[0][3]]
        assert not h.worker.submit(3, speaker, audio(9))
        return
    if transition == "stop":
        # An engine return barrier, not an arbitrary sleep, orders the restart.
        h.qtbot.waitUntil(lambda: bool(h.state.cancelled))
        h.worker.set_session(3, ROUTES)
        for who in (1, 0):
            assert h.worker.submit(3, who, audio(8 + who))
    h.qtbot.waitUntil(lambda: sum(kind == "settled" for kind, _ in h.events) == 2)
    assert h.state.cancelled == [True, False, False]
    assert len(h.state.calls) == 3
    assert [(gen, route.speaker) for gen, route, _, _ in h.state.calls[1:]] == [(3, 0), (3, 1)]
    for _, route, samples, thread in h.state.calls[1:]:
        np.testing.assert_array_equal(samples, audio(8 + route.speaker))
        assert thread == h.state.calls[0][3]
    assert [(kind, (value.generation, value.speaker) if kind == "result" else value)
            for kind, value in h.events] == [
        ("progress", (3, "Stage complete")), ("result", (3, 0)), ("settled", (3, 0, True)),
        ("progress", (3, "Stage complete")), ("result", (3, 1)), ("settled", (3, 1, True))]
    assert all(not queue for queue in h.worker._pending)
    assert not h.worker.submit(1, speaker, audio(1))
