"""GUI abort cleanup boundaries without module-level hardware/model setup."""
import pytest

from test_gui_verifier_lifecycle import verifier


@pytest.mark.parametrize("stage", ["close", "audio", "worker", "both"])
def test_failed_close_attempts_remaining_cleanup_and_exits(verifier, capsys, stage):
    v = verifier
    path = v.root / ".local/gui-integration.json"
    previous = b'{"previous": true}\n'
    path.write_bytes(previous)

    def close():
        v.events.append("close")
        # Simulate failure after closeEvent marked itself closing, so a second
        # close cannot be relied on to repeat shutdown.
        v.ns["abort"]("synchronous close error")
        v.ns["check"]()
        raise RuntimeError("fixture close")

    def stop():
        v.events.append("audio.stop")
        v.ns["abort"]("synchronous audio error")
        if stage in ("audio", "both"):
            raise RuntimeError("fixture audio")

    def shutdown():
        v.events.append("worker.shutdown")
        v.ns["check"]()
        if stage in ("worker", "both"):
            raise RuntimeError("fixture worker")

    v.window.close = close
    v.window.audio.stop = stop
    v.window.worker.shutdown = shutdown
    v.ns["abort"]("original failure")
    assert v.events == ["close", "audio.stop", "worker.shutdown", "timer.stop", ("exit", 1)]
    assert v.state["failed"] and not v.state["done"]
    assert v.evidence["errors"][0] == "original failure"
    assert "fixture close" in v.evidence["errors"][1]
    expected = 4 if stage == "both" else 2 if stage == "close" else 3
    assert len(v.evidence["errors"]) == expected
    assert path.read_bytes() == previous
    assert "PASS_NATIVE_GUI" not in capsys.readouterr().out
    v.events.clear()
    v.ns["abort"]("duplicate")
    v.state["step"] = 6
    v.ns["check"]()
    assert v.events == []


def test_timer_stop_exception_does_not_prevent_failure_exit(verifier):
    v = verifier

    def stop():
        v.events.append("timer.stop")
        raise RuntimeError("fixture timer")

    v.ns["timer"].stop = stop
    v.ns["abort"]("original failure")
    assert v.events == ["close", "timer.stop", ("exit", 1)]
    assert v.evidence["errors"][0] == "original failure"
    assert "fixture timer" in v.evidence["errors"][1]
    assert v.state["failed"] and not v.state["done"]


def test_pending_async_close_is_not_forced_or_repeated(verifier):
    v = verifier
    v.window.worker.isRunning = lambda: True
    v.window.close = lambda: v.events.append("close") or False
    v.window.audio.stop = lambda: pytest.fail("normal close owns audio cleanup")
    v.window.worker.shutdown = lambda: pytest.fail("normal close owns worker shutdown")
    v.ns["abort"]("original failure")
    assert v.events == ["close", "timer.stop", ("exit", 1)]
    assert v.evidence["errors"] == ["original failure"]


def test_close_exception_during_check_cannot_escape_abort(verifier, capsys):
    v = verifier

    def close():
        v.events.append("close")
        raise RuntimeError("fixture close")

    v.window.close = close
    v.window.audio.stop = lambda: v.events.append("audio.stop")
    v.window.worker.shutdown = lambda: v.events.append("worker.shutdown")
    v.ns["time"].monotonic = lambda: 100.
    v.ns["check"]()
    assert v.events[-1] == ("exit", 1)
    assert v.state["failed"] and not v.state["done"]
    assert v.evidence["errors"][0] == "AssertionError: GUI verification timed out"
    assert "PASS_NATIVE_GUI" not in capsys.readouterr().out
