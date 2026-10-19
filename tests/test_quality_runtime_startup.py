"""Startup/cleanup boundaries with tiny assets and fake owned children only."""
from pathlib import Path
import subprocess

import pytest

from translitter import quality_runtime as qr
from translitter.quality import QualityUnavailable
from test_quality_runtime import small_model, stub_runtime  # noqa: F401


@pytest.mark.parametrize("outcome", [True, False, QualityUnavailable("private detail")])
@pytest.mark.parametrize("elapsed", [60.0, 60.5])
def test_probe_finishing_at_or_after_deadline_cannot_publish_ready(stub_runtime, monkeypatch, outcome, elapsed):
    state = stub_runtime

    def probe(client):
        state.probes.append(client.timeout)
        state.now = elapsed
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(qr.LocalTranslator, "probe", probe)
    with pytest.raises(qr.RuntimeUnavailable, match="^Quality runtime readiness timed out$"):
        state.runtime.start()
    assert state.probes == [1]
    assert len(state.clients) == 1
    assert state.events == ["terminate", ("wait", 2)]
    assert state.runtime._process is state.runtime._socket_directory is None


@pytest.mark.parametrize("ready", [False, True])
def test_child_death_during_probe_prevents_client_publication_and_allows_retry(stub_runtime, monkeypatch, ready):
    state = stub_runtime
    original_probe = qr.LocalTranslator.probe

    def probe(client):
        state.probes.append(client.timeout)
        state.returncode = 23
        return ready

    monkeypatch.setattr(qr.LocalTranslator, "probe", probe)
    with pytest.raises(qr.RuntimeUnavailable, match="^Quality runtime exited before readiness$"):
        state.runtime.start()
    assert state.events == [("wait", 2)]
    assert len(state.clients) == 1
    assert state.runtime._process is state.runtime._socket_directory is None
    monkeypatch.setattr(qr.LocalTranslator, "probe", original_probe)
    state.returncode = None
    assert state.runtime.start().timeout == 20
    assert len(state.spawns) == 2


@pytest.mark.parametrize("failure", ["spawn", "timeout", "probe", "cancel"])
def test_failed_start_releases_socket_and_can_retry_same_instance(stub_runtime, failure):
    state = stub_runtime
    if failure == "spawn":
        state.spawn_error = OSError("private path")
    elif failure == "timeout":
        state.outcomes = [False]
    elif failure == "probe":
        state.outcomes = [RuntimeError("private diagnostic")]
    if failure == "cancel":
        assert state.runtime.start(cancelled=lambda: bool(state.probes)) is None
    else:
        with pytest.raises(qr.RuntimeUnavailable):
            state.runtime.start()
    old_path = Path(state.spawns[0][0][state.spawns[0][0].index("--host") + 1]).parent
    assert not old_path.exists()
    assert state.runtime._process is state.runtime._socket_directory is None
    state.spawn_error = None
    state.returncode = None
    state.outcomes = [True]
    assert state.runtime.start().timeout == 20
    assert len(state.spawns) == 2
    new_path = Path(state.runtime._socket_directory.name)
    assert new_path != old_path and new_path.exists()


@pytest.mark.parametrize("cleanup_failures", [1, 2])
def test_socket_cleanup_failure_retains_ownership_and_blocks_overlapping_start(stub_runtime, monkeypatch, cleanup_failures):
    state = stub_runtime
    state.runtime.start()
    directory = state.runtime._socket_directory
    original_cleanup = directory.cleanup
    attempts = []

    def cleanup():
        attempts.append(True)
        if len(attempts) <= cleanup_failures:
            raise OSError("private directory path")
        original_cleanup()

    monkeypatch.setattr(directory, "cleanup", cleanup)
    for _ in range(cleanup_failures):
        with pytest.raises(qr.RuntimeUnavailable, match="^Quality runtime could not stop$") as error:
            state.runtime.close()
        assert error.value.__cause__ is None
        assert state.runtime._process is state.process
        assert state.runtime._socket_directory is directory
        assert Path(directory.name).exists()
        with pytest.raises(qr.RuntimeUnavailable, match="already started"):
            state.runtime.start()
        assert len(state.spawns) == 1
    state.runtime.close()
    assert state.runtime._process is state.runtime._socket_directory is None
    assert not Path(directory.name).exists()
    assert state.events.count("terminate") == 1
    state.returncode = None
    assert state.runtime.start().timeout == 20


@pytest.mark.parametrize("failure", ["cancel", "probe"])
def test_failed_start_cleanup_retains_child_until_bounded_shutdown_retry(stub_runtime, monkeypatch, failure):
    state = stub_runtime
    original_wait = state.process.wait
    original_terminate = state.process.terminate
    original_kill = state.process.kill

    def wait(timeout):
        state.events.append(("wait", timeout))
        raise subprocess.TimeoutExpired("private argv", timeout)

    def terminate():
        state.events.append("terminate")

    def kill():
        state.events.append("kill")

    monkeypatch.setattr(state.process, "wait", wait)
    monkeypatch.setattr(state.process, "terminate", terminate)
    monkeypatch.setattr(state.process, "kill", kill)
    if failure == "probe":
        state.outcomes = [RuntimeError("private failure")]
    with pytest.raises(qr.RuntimeUnavailable, match="^Quality runtime could not stop$"):
        state.runtime.start(cancelled=lambda: failure == "cancel" and bool(state.probes))
    assert state.runtime._process is state.process
    directory = state.runtime._socket_directory
    assert directory is not None and Path(directory.name).exists()
    with pytest.raises(qr.RuntimeUnavailable, match="already started"):
        state.runtime.start()
    assert len(state.spawns) == 1
    assert state.events == ["terminate", ("wait", 2), "kill", ("wait", 2)]
    monkeypatch.setattr(state.process, "wait", original_wait)
    monkeypatch.setattr(state.process, "terminate", original_terminate)
    monkeypatch.setattr(state.process, "kill", original_kill)
    state.runtime.close()
    assert not Path(directory.name).exists()
    assert state.runtime._process is state.runtime._socket_directory is None
    state.returncode = None
    state.outcomes = [True]
    assert state.runtime.start().timeout == 20


def test_final_probe_uses_remaining_budget_without_oversleep(stub_runtime, monkeypatch):
    state = stub_runtime

    def probe(client):
        state.probes.append(client.timeout)
        if len(state.probes) == 1:
            state.now = 59.85
            return False
        return True

    monkeypatch.setattr(qr.LocalTranslator, "probe", probe)
    assert state.runtime.start().timeout == 20
    assert len(state.probes) == 2
    assert state.probes[0] == 1
    assert state.probes[1] == pytest.approx(0.05)
    assert state.now == pytest.approx(59.95)
    assert state.events == []
