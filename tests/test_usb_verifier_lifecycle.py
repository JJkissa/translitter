"""Exercise real verifier callbacks without opening USB devices or models.

Only function definitions are loaded: the opt-in script's top-level hardware
setup must never execute in the unattended quick suite.
"""
import ast
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def verifier(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/verify_usb.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    callbacks = ast.Module(body=[node for node in tree.body
                                if isinstance(node, ast.FunctionDef)], type_ignores=[])
    events = []
    pending = []
    state = {"done": False, "playing": 1, "failed": False}
    evidence = {"turns": [{}], "errors": [], "playback_states": []}

    def stop():
        events.append("stop")
        namespace["on_playing"](1, False)

    hub = SimpleNamespace(stop=stop, headsets=lambda: events.append("headsets"))
    worker = SimpleNamespace(
        shutdown=lambda: events.append("shutdown"),
        submit=lambda *args: events.append(("submit", args)))
    namespace = dict(json=json, state=state, evidence=evidence, hub=hub, worker=worker,
                     app=SimpleNamespace(quit=lambda: events.append("quit")),
                     QTimer=SimpleNamespace(singleShot=lambda delay, call: pending.append(call)),
                     fixtures=[object(), object()], root=tmp_path,
                     subprocess=SimpleNamespace(check_output=lambda *a, **kw: events.append("graph")))
    exec(compile(callbacks, str(path), "exec"), namespace)
    return SimpleNamespace(ns=namespace, events=events, pending=pending,
                           state=state, evidence=evidence)


def test_failure_teardown_does_not_schedule_more_inference(verifier):
    verifier.ns["fail"]("fixture failure")
    assert verifier.state["failed"]
    assert verifier.events == ["stop", "shutdown", "quit"]
    assert verifier.pending == []


def test_failure_is_idempotent_under_duplicate_error_signals(verifier):
    verifier.ns["fail"]("first failure")
    verifier.ns["fail"]("second failure")
    assert verifier.events == ["stop", "shutdown", "quit"]
    assert verifier.evidence["errors"] == ["first failure"]


@pytest.mark.parametrize("callback,args", [
    ("begin", ()), ("receive", (SimpleNamespace(generation=99),)),
    ("inspect_graph", (1,)), ("finish", ()), ("on_playing", (1, False)),
])
def test_queued_callback_after_failure_is_inert(verifier, callback, args):
    verifier.state["failed"] = True
    verifier.ns[callback](*args)
    assert verifier.events == []
    assert verifier.pending == []
    assert verifier.evidence == {"turns": [{}], "errors": [], "playback_states": []}


def test_already_queued_second_turn_is_cancelled_by_failure(verifier):
    verifier.ns["on_playing"](1, False)
    assert len(verifier.pending) == 1
    verifier.ns["fail"]("fixture failure")
    verifier.events.clear()
    verifier.pending.pop(0)()
    assert verifier.events == []


def test_live_playback_completion_still_submits_second_fixture(verifier):
    verifier.ns["on_playing"](1, False)
    assert len(verifier.pending) == 1
    verifier.pending.pop(0)()
    assert verifier.events == [("submit", (1, 1, verifier.ns["fixtures"][1]))]


def test_live_second_completion_still_schedules_finish(verifier):
    verifier.evidence["turns"].append({})
    verifier.ns["on_playing"](1, False)
    assert verifier.pending == [verifier.ns["finish"]]
