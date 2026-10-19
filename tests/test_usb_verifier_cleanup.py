"""Teardown boundaries of real USB callbacks, without device/model setup."""
import json

import pytest

from test_usb_verifier_evidence import verifier
from test_usb_verifier_lifecycle import verifier as callback_verifier


@pytest.mark.parametrize("stage", ["stop", "stop_session", "shutdown"])
def test_finish_reentrant_failure_never_publishes_success(verifier, capsys, stage):
    v = verifier
    path = v.root / ".local/usb-pipeline-check.json"
    previous = b'{"previous": true}\n'
    path.write_bytes(previous)

    def operation(name):
        def call(*args):
            v.events.append(name)
            if name == stage:
                v.ns["fail"]("synchronous teardown failure")
        return call

    v.ns["hub"].stop = operation("stop")
    v.ns["worker"].stop_session = operation("stop_session")
    v.ns["worker"].shutdown = operation("shutdown")
    v.ns["finish"]()
    assert v.state["failed"] and not v.state["done"]
    assert path.read_bytes() == previous
    assert v.events == ["stop", "stop_session", "shutdown", "quit"]
    assert v.evidence["errors"] == ["synchronous teardown failure"]
    assert "PASS_BOTH" not in capsys.readouterr().out


@pytest.mark.parametrize("stage", ["stop", "stop_session", "shutdown"])
def test_finish_cleanup_exception_attempts_remaining_teardown(verifier, capsys, stage):
    v = verifier

    def operation(name):
        def call(*args):
            v.events.append(name)
            if name == stage:
                raise RuntimeError("fixture cleanup exception")
        return call

    v.ns["hub"].stop = operation("stop")
    v.ns["worker"].stop_session = operation("stop_session")
    v.ns["worker"].shutdown = operation("shutdown")
    v.ns["finish"]()
    assert v.events == ["stop", "stop_session", "shutdown", "quit"]
    assert v.state["failed"] and not v.state["done"]
    assert len(v.evidence["errors"]) == 1
    assert "fixture cleanup exception" in v.evidence["errors"][0]
    assert not (v.root / ".local/usb-pipeline-check.json").exists()
    assert "PASS_BOTH" not in capsys.readouterr().out


@pytest.mark.parametrize("stage", ["stop", "shutdown", "both"])
def test_failure_cleanup_exceptions_do_not_escape_or_prevent_quit(verifier, stage):
    v = verifier

    def stop():
        v.events.append("stop")
        if stage in ("stop", "both"):
            raise RuntimeError("fixture stop")

    def shutdown():
        v.events.append("shutdown")
        if stage in ("shutdown", "both"):
            raise RuntimeError("fixture shutdown")

    v.ns["hub"].stop = stop
    v.ns["worker"].shutdown = shutdown
    v.ns["fail"]("original failure")
    assert v.events == ["stop", "shutdown", "quit"]
    assert v.state["failed"] and not v.state["done"]
    assert v.evidence["errors"][0] == "original failure"
    assert len(v.evidence["errors"]) == (3 if stage == "both" else 2)
    v.ns["fail"]("duplicate")
    assert v.events == ["stop", "shutdown", "quit"]


def test_cleanup_notifications_cannot_reenter_finish_or_schedule_work(verifier):
    v = verifier

    def stop():
        v.events.append("stop")
        v.ns["finish"]()
        v.ns["on_playing"](1, False)
        v.ns["begin"]()

    v.ns["hub"].stop = stop
    v.ns["finish"]()
    assert v.state["done"] and not v.state["failed"]
    assert v.events == ["stop", ("stop_session", 2), "shutdown", "quit"]
    assert v.pending == []
    saved = json.loads((v.root / ".local/usb-pipeline-check.json").read_text(encoding="utf-8"))
    assert saved["errors"] == []


def test_ordinary_finish_cleanup_still_publishes_success(verifier, capsys):
    v = verifier
    v.ns["finish"]()
    assert v.events == ["stop", ("stop_session", 2), "shutdown", "quit"]
    assert v.state["done"] and not v.state["failed"]
    assert "PASS_BOTH" in capsys.readouterr().out
