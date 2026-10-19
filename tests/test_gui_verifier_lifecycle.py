"""Real GUI verifier callbacks, without module-level models or USB setup."""
import ast
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def verifier(tmp_path):
    path = Path(__file__).resolve().parents[1] / "scripts/verify_gui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    callbacks = ast.Module(body=[node for node in tree.body
                                if isinstance(node, ast.FunctionDef)], type_ignores=[])
    events = []
    clock = SimpleNamespace(monotonic=lambda: 0.)
    namespace = dict(time=clock, json=json, root=tmp_path)
    # Use the script's real initial state without executing its hardware setup.
    for node in tree.body:
        if (isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in {"state", "evidence"}
                for target in node.targets)):
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    worker = SimpleNamespace(isRunning=lambda: False)
    (tmp_path / ".local").mkdir()
    cards = [SimpleNamespace(translated=SimpleNamespace(
        toPlainText=lambda: events.append("caption.read") or "")) for _ in range(2)]
    window = SimpleNamespace(models_ready=True, worker=worker, cards=cards,
        audio=SimpleNamespace(headsets=lambda: events.append("headsets") or []),
        isVisible=lambda: False,
        close=lambda: events.append("close"))
    namespace.update(window=window,
        timer=SimpleNamespace(stop=lambda: events.append("timer.stop")),
        app=SimpleNamespace(exit=lambda code: events.append(("exit", code)),
                            quit=lambda: events.append("quit")))
    exec(compile(callbacks, str(path), "exec"), namespace)
    return SimpleNamespace(ns=namespace, events=events, state=namespace["state"],
                           evidence=namespace["evidence"], window=window, root=tmp_path)


def test_abort_is_idempotent_under_duplicate_signals(verifier):
    verifier.ns["abort"]("first")
    verifier.ns["abort"]("second")
    assert verifier.evidence["errors"] == ["first"]
    assert verifier.events.count("close") == 1
    assert verifier.events.count(("exit", 1)) == 1
    assert verifier.events.count("timer.stop") == 1


def test_abort_marks_failure_before_synchronous_close_notification(verifier):
    def close():
        verifier.events.append("close")
        # A close notification can synchronously deliver an error or timer work.
        if verifier.events.count("close") == 1:
            verifier.ns["abort"]("close error")
            verifier.ns["check"]()
    verifier.window.close = close
    verifier.ns["abort"]("original error")
    assert verifier.events.count("close") == 1
    assert "headsets" not in verifier.events
    assert verifier.evidence["errors"] == ["original error"]


@pytest.mark.parametrize("step", [0, 1, 6])
def test_queued_check_after_abort_is_inert(verifier, step):
    verifier.state["step"] = step
    verifier.ns["abort"]("failure")
    before = deepcopy(verifier.evidence)
    verifier.events.clear()
    verifier.ns["check"]()
    assert verifier.events == []
    assert verifier.evidence == before
    assert not verifier.state["done"]
    assert not (verifier.root / ".local/gui-integration.json").exists()


def test_expired_queued_check_after_abort_does_not_repeat_failure(verifier):
    verifier.ns["abort"]("first")
    verifier.ns["time"].monotonic = lambda: 100.
    verifier.events.clear()
    verifier.ns["check"]()
    assert verifier.events == []
    assert verifier.evidence["errors"] == ["first"]


def test_live_check_still_polls_readiness(verifier):
    verifier.ns["check"]()
    assert verifier.events == ["headsets"]
    assert verifier.evidence["errors"] == []


def test_live_timeout_still_aborts(verifier):
    verifier.ns["time"].monotonic = lambda: 100.
    verifier.ns["check"]()
    assert verifier.evidence["errors"] == ["AssertionError: GUI verification timed out"]
    assert verifier.events.count("close") == 1
    assert verifier.events.count(("exit", 1)) == 1


def test_live_close_waits_before_publishing_success(verifier):
    verifier.state["step"] = 6
    verifier.window.worker.isRunning = lambda: True
    verifier.ns["check"]()
    assert verifier.events == []
    assert not verifier.state["done"]
    assert not (verifier.root / ".local/gui-integration.json").exists()


def test_live_closed_window_still_completes(verifier, capsys):
    verifier.state["step"] = 6
    verifier.ns["check"]()
    assert verifier.state["done"]
    assert verifier.events == ["timer.stop", "quit"]
    saved = json.loads((verifier.root / ".local/gui-integration.json").read_text())
    assert saved == verifier.evidence
    assert "PASS_NATIVE_GUI" in capsys.readouterr().out
