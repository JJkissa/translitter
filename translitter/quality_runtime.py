"""Optional, explicitly started local quality runtime; never manages LM Studio."""
import hashlib
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time
from urllib.request import ProxyHandler, build_opener

from .quality import LocalTranslator, QualityUnavailable, _NoRedirects
from .quality_transport import OwnedUnixHTTPHandler

MODEL_RELATIVE_PATH = Path("models/quality-translategemma-4b/translategemma-4b-llamacpp-Q4_K_M.gguf")
MODEL_SIZE = 2489909312
MODEL_SHA256 = "526747309109c016db547c6fc1c7b0c9c286b5e7a7556827b5419fd9543a09cd"


class RuntimeUnavailable(RuntimeError):
    """A sanitized local runtime failure, without child output or user text."""


def _find_backend():
    """Choose only an installed Vulkan/AVX2 backend, numerically by version."""
    directory = Path.home() / ".lmstudio/extensions/backends"
    candidates = []
    try:
        for server in directory.glob("llama.cpp-linux-x86_64-vulkan-avx2-*/llama-server"):
            match = re.fullmatch(r"llama\.cpp-linux-x86_64-vulkan-avx2-(\d+)\.(\d+)\.(\d+)", server.parent.name)
            if match and server.is_file() and os.access(server, os.X_OK):
                candidates.append((tuple(map(int, match.groups())), server))
    except OSError:
        raise RuntimeUnavailable("Installed quality runtime was not found") from None
    if not candidates:
        raise RuntimeUnavailable("Installed quality runtime was not found")
    return max(candidates)[1]


class QualityRuntime:
    def __init__(self, root):
        self.root = Path(root)
        self._process = None
        self._socket_directory = None

    def start(self, cancelled=lambda: False):
        """Start an owned server; return a ready client, or None on cancellation.

        Repeated start is rejected until close() releases the existing child.
        """
        if self._process is not None or self._socket_directory is not None:
            raise RuntimeUnavailable("Quality runtime is already started")
        if cancelled():
            return None
        model = self.root / MODEL_RELATIVE_PATH
        try:
            valid = model.is_file() and model.stat().st_size == MODEL_SIZE
        except OSError:
            valid = False
        if not valid:
            raise RuntimeUnavailable("Quality model is missing or invalid") from None
        try:
            digest = hashlib.sha256()
            with model.open("rb") as stream:
                while True:
                    if cancelled():
                        return None
                    block = stream.read(1048576)
                    if not block:
                        break
                    digest.update(block)
            if digest.hexdigest() != MODEL_SHA256:
                raise RuntimeUnavailable("Quality model is missing or invalid")
        except OSError:
            raise RuntimeUnavailable("Quality model is missing or invalid") from None

        backend = _find_backend()
        if cancelled():
            return None
        ready = False
        try:
            if not hasattr(socket, "SO_PEERCRED") or not hasattr(socket, "AF_UNIX"):
                raise RuntimeUnavailable("Quality runtime requires Linux private IPC")
            directory = Path(os.environ.get("TMPDIR", Path.home() / ".hermes/cache/scratch"))
            self._socket_directory = tempfile.TemporaryDirectory(prefix="tq-", dir=directory)
            socket_path = Path(self._socket_directory.name).resolve() / "api.sock"
            if len(os.fsencode(socket_path)) >= 108:
                raise RuntimeUnavailable("Quality runtime private IPC path is too long")
            self._process = subprocess.Popen([
                str(backend), "--model", str(model), "--host", str(socket_path),
                "--alias", "translitter-quality", "--ctx-size", "2048",
                "--parallel", "1", "--n-gpu-layers", "99", "--threads", "4",
                "--threads-batch", "4", "--offline", "--no-webui", "--no-jinja",
                "--chat-template", "gemma", "--log-disable",
                "--cors-origins", "", "--no-slots", "--no-agent", "--no-ui-mcp-proxy",
            ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, start_new_session=True,
                env={key: value for key, value in os.environ.items() if not key.startswith("LLAMA_")})
            opener = build_opener(ProxyHandler({}), _NoRedirects(),
                                  OwnedUnixHTTPHandler(socket_path, self._process))
            probe = LocalTranslator(timeout=1, opener=opener)
            deadline = time.monotonic() + 60
            while True:
                if cancelled():
                    return None
                if self._process.poll() is not None:
                    raise RuntimeUnavailable("Quality runtime exited before readiness")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeUnavailable("Quality runtime readiness timed out")
                probe.timeout = min(1, remaining)
                try:
                    available = probe.probe()
                except QualityUnavailable:
                    available = False
                if cancelled():
                    return None
                if self._process.poll() is not None:
                    raise RuntimeUnavailable("Quality runtime exited before readiness")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeUnavailable("Quality runtime readiness timed out")
                if available:
                    client = LocalTranslator(timeout=20, opener=opener)
                    ready = True
                    return client
                time.sleep(min(0.1, remaining))
        except RuntimeUnavailable:
            raise
        except Exception:
            raise RuntimeUnavailable("Quality runtime could not start") from None
        finally:
            if not ready:
                self.close()

    def close(self):
        """Release only resources owned by this instance; safe before start."""
        process = self._process
        if process is None:
            self._cleanup_socket_directory()
            return
        try:
            if process.poll() is None:
                try:
                    process.terminate()
                    process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired):
                    try:
                        process.kill()
                    except OSError:
                        # It may have exited between poll and signal; still reap it.
                        pass
                    process.wait(timeout=2)
            else:
                process.wait(timeout=2)
        except (OSError, subprocess.TimeoutExpired):
            # Retain ownership so a later close can retry a failed shutdown.
            raise RuntimeUnavailable("Quality runtime could not stop") from None
        self._cleanup_socket_directory()
        self._process = None

    def _cleanup_socket_directory(self):
        directory = self._socket_directory
        if directory is not None:
            try:
                directory.cleanup()
            except OSError:
                raise RuntimeUnavailable("Quality runtime could not stop") from None
            self._socket_directory = None
