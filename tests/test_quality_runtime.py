"""Owned runtime tests: no real sockets, child servers, or model assets."""
import hashlib
import importlib
import io
import os
from pathlib import Path
from types import SimpleNamespace
import subprocess

from translitter.quality import QualityUnavailable

import pytest

from translitter import quality_runtime as qr


@pytest.mark.parametrize("contents", [None, b"wrong size"])
def test_missing_or_wrong_size_model_refuses_start_without_path_leak(tmp_path, contents):
    model = tmp_path / "models/quality-translategemma-4b/translategemma-4b-llamacpp-Q4_K_M.gguf"
    if contents is not None:
        model.parent.mkdir(parents=True)
        model.write_bytes(contents)
    with pytest.raises(qr.RuntimeUnavailable, match="^Quality model is missing or invalid$") as error:
        qr.QualityRuntime(tmp_path).start()
    assert str(tmp_path) not in str(error.value)


@pytest.fixture
def small_model(tmp_path, monkeypatch):
    model = tmp_path / qr.MODEL_RELATIVE_PATH
    model.parent.mkdir(parents=True)
    contents = b"fixture model bytes"
    model.write_bytes(contents)
    monkeypatch.setattr(qr, "MODEL_SIZE", len(contents))
    monkeypatch.setattr(qr, "MODEL_SHA256", hashlib.sha256(contents).hexdigest())
    return model


def test_hash_mismatch_is_rejected_with_bounded_reads(small_model, monkeypatch):
    sizes = []

    class Blocks(io.BytesIO):
        def read(self, size=-1):
            sizes.append(size)
            return super().read(size)

    contents = small_model.read_bytes()
    monkeypatch.setattr(Path, "open", lambda *a, **k: Blocks(contents))
    monkeypatch.setattr(qr, "MODEL_SHA256", "0" * 64)
    with pytest.raises(qr.RuntimeUnavailable, match="^Quality model is missing or invalid$"):
        qr.QualityRuntime(small_model.parents[2]).start()
    assert sizes and all(0 < size <= 1048576 for size in sizes)


def test_cancellation_during_hash_stops_before_another_block(small_model, monkeypatch):
    reads = []
    contents = small_model.read_bytes()

    class CancelAfterRead(io.BytesIO):
        def read(self, size=-1):
            reads.append(size)
            return super().read(size)

    stream = CancelAfterRead(contents)
    monkeypatch.setattr(Path, "open", lambda *a, **k: stream)
    runtime = qr.QualityRuntime(small_model.parents[2])
    assert runtime.start(cancelled=lambda: bool(reads)) is None
    assert reads == [1048576]
    assert stream.closed
    runtime.close()


@pytest.mark.parametrize("versions,expected", [([], None), (["2.9.9", "2.50.0", "2.10.1"], "2.50.0")])
def test_discovery_only_uses_installed_executable_semver_backends(tmp_path, monkeypatch, versions, expected):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    backends = tmp_path / ".lmstudio/extensions/backends"
    for version in versions:
        server = backends / f"llama.cpp-linux-x86_64-vulkan-avx2-{version}" / "llama-server"
        server.parent.mkdir(parents=True)
        server.write_bytes(b"fixture executable")
        server.chmod(0o700)
    for name in ["llama.cpp-linux-x86_64-vulkan-avx2-99.0.0", "llama.cpp-linux-x86_64-vulkan-avx2-latest", "llama.cpp-linux-x86_64-cuda-999.0.0"]:
        invalid = backends / name / "llama-server"
        invalid.parent.mkdir(parents=True)
        invalid.write_bytes(b"not selected")
        invalid.chmod(0o600 if "99.0.0" in name else 0o700)
    if expected is None:
        with pytest.raises(qr.RuntimeUnavailable, match="^Installed quality runtime was not found$"):
            qr._find_backend()
    else:
        assert qr._find_backend() == backends / f"llama.cpp-linux-x86_64-vulkan-avx2-{expected}" / "llama-server"


def test_pre_cancelled_start_does_nothing_and_close_is_idempotent(tmp_path):
    runtime_module = importlib.import_module("translitter.quality_runtime")
    runtime = runtime_module.QualityRuntime(tmp_path)
    assert runtime.start(cancelled=lambda: True) is None
    runtime.close()
    runtime.close()


@pytest.fixture
def stub_runtime(small_model, monkeypatch):
    state = SimpleNamespace(spawns=[], probes=[], outcomes=[True], clients=[],
                            bindings=[], socket_closed=False, occupied=False,
                            now=0.0, events=[], returncode=None, spawn_error=None,
                            socket_options=[], require_reuse=False)

    class Process:
        pid = os.getpid()

        def poll(self):
            return state.returncode

        def terminate(self):
            state.events.append("terminate")
            state.returncode = -15

        def kill(self):
            state.events.append("kill")
            state.returncode = -9

        def wait(self, timeout):
            state.events.append(("wait", timeout))
            return state.returncode

    state.process = Process()

    def spawn(argv, **kwargs):
        state.spawns.append((argv, kwargs))
        if state.spawn_error:
            raise state.spawn_error
        return state.process

    class Client:
        def __init__(self, timeout, opener=None):
            self.opener = opener
            self.timeout = timeout
            state.clients.append(self)

        def probe(self):
            state.probes.append(self.timeout)
            outcome = state.outcomes.pop(0) if len(state.outcomes) > 1 else state.outcomes[0]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

    def sleep(seconds):
        state.now += seconds

    monkeypatch.setattr(qr, "subprocess", SimpleNamespace(Popen=spawn, DEVNULL=subprocess.DEVNULL,
                                                        TimeoutExpired=subprocess.TimeoutExpired), raising=False)
    monkeypatch.setattr(qr, "time", SimpleNamespace(monotonic=lambda: state.now, sleep=sleep), raising=False)
    monkeypatch.setattr(qr, "LocalTranslator", Client, raising=False)
    monkeypatch.setattr(qr, "_find_backend", lambda: Path("/installed/llama-server"))
    state.runtime = qr.QualityRuntime(small_model.parents[2])
    state.model = small_model
    yield state
    state.runtime.close()


def test_start_spawns_private_unix_runtime_and_waits_for_readiness(stub_runtime):
    state = stub_runtime
    state.outcomes = [QualityUnavailable("not ready"), False, True]
    client = state.runtime.start()
    assert client is state.clients[-1]
    assert client.timeout == 20
    assert state.probes == [1, 1, 1]
    socket_path = Path(state.spawns[0][0][state.spawns[0][0].index("--host") + 1])
    assert socket_path.is_absolute()
    assert socket_path.name == "api.sock"
    assert socket_path.parent.stat().st_mode & 0o777 == 0o700
    assert len(os.fsencode(socket_path)) < 108
    assert state.bindings == []
    assert state.clients[0].opener is client.opener
    from translitter.quality_transport import OwnedUnixHTTPHandler
    assert any(isinstance(h, OwnedUnixHTTPHandler) for h in client.opener.handlers)
    assert state.spawns == [([
        "/installed/llama-server", "--model", str(state.model),
        "--host", str(socket_path), "--alias", "translitter-quality",
        "--ctx-size", "2048", "--parallel", "1", "--n-gpu-layers", "99",
        "--threads", "4", "--threads-batch", "4", "--offline", "--no-webui",
        "--no-jinja", "--chat-template", "gemma", "--log-disable",
        "--cors-origins", "", "--no-slots", "--no-agent", "--no-ui-mcp-proxy",
    ], dict(stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, start_new_session=True,
            env={key: value for key, value in os.environ.items() if not key.startswith("LLAMA_")}))]


def test_runtime_disables_browser_access_tools_and_inherited_overrides(stub_runtime, monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_RPC", "outside.example:50052")
    monkeypatch.setenv("LLAMA_ARG_LOG_FILE", "/private/conversation.log")
    monkeypatch.setenv("LLAMA_ARG_MCP_SERVERS_JSON", "untrusted")
    monkeypatch.setenv("TRANSLITTER_KEEP_NATIVE_ENV", "preserve")
    state = stub_runtime
    state.runtime.start()
    argv, options = state.spawns[0]
    assert argv[argv.index("--cors-origins") + 1] == ""
    assert all(flag in argv for flag in ("--no-slots", "--no-agent", "--no-ui-mcp-proxy"))
    assert not any(key.startswith("LLAMA_") for key in options["env"])
    assert options["env"]["TRANSLITTER_KEEP_NATIVE_ENV"] == "preserve"


def test_runtime_never_binds_or_connects_a_tcp_socket(stub_runtime, monkeypatch):
    def forbidden_socket(*args, **kwargs):
        pytest.fail("Runtime must not reserve, adopt, or connect a TCP port")
    monkeypatch.setattr(qr.socket, "socket", forbidden_socket)
    assert stub_runtime.runtime.start() is not None
    argv = stub_runtime.spawns[0][0]
    assert "--port" not in argv and "127.0.0.1" not in argv
    stub_runtime.runtime.close()


def test_spawn_error_is_sanitized_without_touching_other_processes(stub_runtime):
    state = stub_runtime
    state.spawn_error = OSError("/private/model/path and child details")
    with pytest.raises(qr.RuntimeUnavailable, match="^Quality runtime could not start$") as error:
        state.runtime.start()
    assert error.value.__cause__ is None
    assert state.events == state.probes == []


def test_close_reaps_only_owned_child_and_is_idempotent(stub_runtime):
    state = stub_runtime
    state.runtime.start()
    socket_directory = Path(state.runtime._socket_directory.name)
    assert socket_directory.exists()
    state.runtime.close()
    state.runtime.close()
    assert not socket_directory.exists()
    assert state.runtime._socket_directory is None
    assert state.events == ["terminate", ("wait", 2)]
    assert state.runtime._process is None


@pytest.mark.parametrize("during_probe", [False, True])
def test_cancellation_cleans_owned_child_even_when_probe_reports_ready(stub_runtime, during_probe):
    state = stub_runtime
    assert state.runtime.start(cancelled=lambda: bool(state.probes if during_probe else state.spawns)) is None
    assert state.events == ["terminate", ("wait", 2)]
    assert len(state.probes) == int(during_probe)
    assert state.runtime._process is None


def test_cancel_after_validation_never_spawns(stub_runtime, monkeypatch):
    state = stub_runtime
    stopped = []

    def backend():
        stopped.append(True)
        return Path("/installed/llama-server")

    monkeypatch.setattr(qr, "_find_backend", backend)
    assert state.runtime.start(cancelled=lambda: bool(stopped)) is None
    assert state.spawns == state.probes == state.events == []


def test_startup_death_is_sanitized_and_child_reaped_without_probe(stub_runtime):
    state = stub_runtime
    state.returncode = 17
    with pytest.raises(qr.RuntimeUnavailable, match="^Quality runtime exited before readiness$"):
        state.runtime.start()
    assert state.probes == []
    assert state.events == [("wait", 2)]
    assert state.runtime._process is None


def test_readiness_deadline_cleans_child_without_real_waits(stub_runtime, monkeypatch):
    state = stub_runtime
    state.outcomes = [QualityUnavailable("private child diagnostic")]

    def sleep(seconds):
        state.now += seconds
        if state.now > 61:
            raise AssertionError("readiness exceeded its 60-second deadline")

    monkeypatch.setattr(qr.time, "sleep", sleep)
    with pytest.raises(qr.RuntimeUnavailable, match="^Quality runtime readiness timed out$"):
        state.runtime.start()
    assert 60 <= state.now <= 61
    assert state.events == ["terminate", ("wait", 2)]
    assert state.runtime._process is None


def test_unexpected_readiness_failure_is_sanitized_and_cleaned(stub_runtime):
    state = stub_runtime
    state.outcomes = [RuntimeError("private failure")]
    with pytest.raises(qr.RuntimeUnavailable, match="^Quality runtime could not start$"):
        state.runtime.start()
    assert state.events == ["terminate", ("wait", 2)]
    assert state.runtime._process is None


def test_repeated_start_rejects_without_losing_owned_child(stub_runtime):
    state = stub_runtime
    state.runtime.start()
    with pytest.raises(qr.RuntimeUnavailable, match="^Quality runtime is already started$"):
        state.runtime.start()
    assert len(state.spawns) == 1
    assert state.events == []
    state.runtime.close()
    assert state.events == ["terminate", ("wait", 2)]


@pytest.mark.parametrize("failure", ["terminate", "wait"])
def test_shutdown_escalates_to_kill_and_bounded_reap(stub_runtime, monkeypatch, failure):
    state = stub_runtime
    state.runtime.start()
    original_wait = state.process.wait

    def terminate():
        state.events.append("terminate")
        if failure == "terminate":
            raise OSError("private signal details")

    def wait(timeout):
        if state.returncode is None:
            state.events.append(("wait", timeout))
            raise subprocess.TimeoutExpired("private argv", timeout)
        return original_wait(timeout)

    monkeypatch.setattr(state.process, "terminate", terminate)
    monkeypatch.setattr(state.process, "wait", wait)
    state.runtime.close()
    state.runtime.close()
    expected = ["terminate"]
    if failure == "wait":
        expected.append(("wait", 2))
    assert state.events == expected + ["kill", ("wait", 2)]
    assert state.runtime._process is None


def test_failed_kill_is_sanitized_without_losing_child_and_close_can_retry(stub_runtime, monkeypatch):
    state = stub_runtime
    state.runtime.start()
    terminate = state.process.terminate
    wait = state.process.wait
    kill = state.process.kill

    def fail_terminate():
        state.events.append("terminate")
        raise OSError("private terminate details")

    def fail_kill():
        state.events.append("kill")
        raise OSError("private kill details")

    def fail_wait(timeout):
        state.events.append(("wait", timeout))
        raise subprocess.TimeoutExpired("private argv", timeout)

    monkeypatch.setattr(state.process, "terminate", fail_terminate)
    monkeypatch.setattr(state.process, "kill", fail_kill)
    monkeypatch.setattr(state.process, "wait", fail_wait)
    with pytest.raises(qr.RuntimeUnavailable, match="^Quality runtime could not stop$"):
        state.runtime.close()
    assert state.events == ["terminate", "kill", ("wait", 2)]
    assert state.runtime._process is state.process
    assert Path(state.runtime._socket_directory.name).exists()
    monkeypatch.setattr(state.process, "terminate", terminate)
    monkeypatch.setattr(state.process, "kill", kill)
    monkeypatch.setattr(state.process, "wait", wait)
    state.runtime.close()
    assert state.runtime._process is None


def test_start_can_run_again_after_close(stub_runtime):
    state = stub_runtime
    first = state.runtime.start()
    state.runtime.close()
    state.returncode = None
    second = state.runtime.start()
    assert second is not first
    assert len(state.spawns) == 2
    state.runtime.close()
    assert state.events == ["terminate", ("wait", 2), "terminate", ("wait", 2)]
