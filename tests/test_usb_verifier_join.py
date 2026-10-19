"""Real USB verifier trailing cleanup, without module-level hardware setup."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def teardown():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_usb.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    index = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and ast.unparse(node.value.func) == "app.exec")
    callbacks = ast.Module(body=[node for node in tree.body if isinstance(node, ast.FunctionDef)],
                           type_ignores=[])
    code = compile(ast.Module(body=tree.body[index + 1:], type_ignores=[]), str(path), "exec")
    events, pending, faults = [], [], set()
    state = {"done": True, "failed": False, "playing": 1}
    evidence = {"errors": [], "turns": [{}], "playback_states": []}

    def operation(name, result=None):
        events.append(name)
        if name in faults:
            raise RuntimeError("fixture " + name)
        return result

    worker = SimpleNamespace(shutdown=lambda: operation("shutdown"),
                             wait=lambda timeout: operation(("wait", timeout), True))
    hub = SimpleNamespace(stop=lambda: operation("stop"))
    ns = dict(hub=hub, worker=worker, state=state, evidence=evidence,
              QTimer=SimpleNamespace(singleShot=lambda delay, call: pending.append(call)))
    exec(compile(callbacks, str(path), "exec"), ns)
    return SimpleNamespace(run=lambda: exec(code, ns), events=events, pending=pending,
                           state=state, evidence=evidence, faults=faults, worker=worker,
                           hub=hub, ns=ns)


@pytest.mark.parametrize("faults", [{"stop"}, {"shutdown"}, {"stop", "shutdown"}])
@pytest.mark.parametrize("after_abort", [False, True])
def test_cleanup_faults_still_attempt_audio_shutdown_and_join(teardown, faults, after_abort):
    t = teardown
    t.faults.update(faults)
    if after_abort:
        t.state.update(done=False, failed=True, cleanup_started=True)
        t.evidence["errors"].append("original failure")
    with pytest.raises(AssertionError):
        t.run()
    assert t.events == ["stop", "shutdown", ("wait", 15000)]
    assert t.state["failed"]
    expected = len(faults) + int(after_abort)
    assert len(t.evidence["errors"]) == expected
    if after_abort:
        assert t.evidence["errors"][0] == "original failure"
    for fault in faults:
        assert any("fixture " + fault in error for error in t.evidence["errors"])


@pytest.mark.parametrize("kind", ["timeout", "exception"])
@pytest.mark.parametrize("after_abort", [False, True])
def test_join_faults_are_recorded_with_original_diagnostics(teardown, kind, after_abort):
    t = teardown
    if after_abort:
        t.state.update(done=False, failed=True)
        t.evidence["errors"].append("original failure")

    def wait(timeout):
        t.events.append(("wait", timeout))
        if kind == "exception":
            raise RuntimeError("fixture wait")
        return False

    t.worker.wait = wait
    with pytest.raises(AssertionError):
        t.run()
    assert t.events == ["stop", "shutdown", ("wait", 15000)]
    assert t.state["failed"]
    assert len(t.evidence["errors"]) == 1 + int(after_abort)
    if after_abort:
        assert t.evidence["errors"][0] == "original failure"
    assert "wait" in t.evidence["errors"][-1].lower() or "join" in t.evidence["errors"][-1].lower()


def test_normal_trailing_cleanup_stops_audio_and_joins(teardown):
    teardown.run()
    assert teardown.events == ["stop", "shutdown", ("wait", 15000)]
    assert not teardown.state["failed"]
    assert teardown.evidence["errors"] == []


def test_post_abort_retries_idempotent_cleanup_and_preserves_error(teardown):
    t = teardown
    t.state.update(done=False, failed=True, cleanup_started=True)
    t.evidence["errors"].append("original failure")
    with pytest.raises(AssertionError):
        t.run()
    assert t.events == ["stop", "shutdown", ("wait", 15000)]
    assert t.evidence["errors"] == ["original failure"]


def test_abnormal_loop_exit_releases_audio_without_scheduling_inference(teardown):
    t = teardown
    t.state.update(done=False, failed=False)

    def stop():
        t.events.append("stop")
        t.ns["on_playing"](1, False)

    t.hub.stop = stop
    with pytest.raises(AssertionError):
        t.run()
    assert t.events == ["stop", "shutdown", ("wait", 15000)]
    assert t.pending == []
    assert t.evidence["playback_states"] == []
