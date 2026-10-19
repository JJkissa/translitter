"""Exercise actual post-event-loop statements without opening models/audio."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def teardown():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_gui.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    index = next(i for i, node in enumerate(tree.body)
                 if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                 and ast.unparse(node.value.func) == "app.exec")
    code = compile(ast.Module(body=tree.body[index + 1:], type_ignores=[]), str(path), "exec")
    events = []
    state = {"done": True, "failed": False}
    evidence = {"errors": []}
    faults = set()

    def operation(name, result=None):
        events.append(name)
        if name in faults:
            raise RuntimeError("fixture " + name)
        return result

    worker = SimpleNamespace(shutdown=lambda: operation("shutdown"),
                             wait=lambda timeout: operation(("wait", timeout), True))
    window = SimpleNamespace(audio=SimpleNamespace(stop=lambda: operation("stop")), worker=worker)
    ns = dict(window=window, state=state, evidence=evidence)
    return SimpleNamespace(run=lambda: exec(code, ns), events=events, state=state,
                           evidence=evidence, faults=faults, worker=worker)


@pytest.mark.parametrize("faults", [{"stop"}, {"shutdown"}, {"stop", "shutdown"}])
def test_cleanup_exceptions_do_not_skip_shutdown_or_join(teardown, faults):
    t = teardown
    t.faults.update(faults)
    with pytest.raises(AssertionError):
        t.run()
    assert t.events == ["stop", "shutdown", ("wait", 15000)]
    assert t.state["failed"]
    assert len(t.evidence["errors"]) == len(faults)
    for fault in faults:
        assert any("fixture " + fault in error for error in t.evidence["errors"])


@pytest.mark.parametrize("kind", ["timeout", "exception"])
def test_join_failure_is_recorded_without_hiding_prior_failure(teardown, kind):
    t = teardown
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
    assert t.evidence["errors"][0] == "original failure"
    assert len(t.evidence["errors"]) == 2
    assert "wait" in t.evidence["errors"][1].lower() or "close" in t.evidence["errors"][1].lower()


def test_normal_post_loop_cleanup_still_joins(teardown):
    teardown.run()
    assert teardown.events == ["stop", "shutdown", ("wait", 15000)]
    assert not teardown.state["failed"]
    assert teardown.evidence["errors"] == []


def test_post_abort_cleanup_preserves_original_failure_and_joins(teardown):
    t = teardown
    t.state.update(done=False, failed=True)
    t.evidence["errors"].append("original failure")
    with pytest.raises(AssertionError):
        t.run()
    assert t.events == ["stop", "shutdown", ("wait", 15000)]
    assert t.evidence["errors"] == ["original failure"]
