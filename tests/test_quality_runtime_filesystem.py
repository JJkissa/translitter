"""Tiny local asset/discovery boundaries; never execute backend fixtures."""
import io
from pathlib import Path

import pytest

from translitter import quality_runtime as qr
from test_quality_runtime import small_model, stub_runtime  # noqa: F401


@pytest.mark.parametrize("boundary", ["eof", "close"])
def test_cancel_at_hash_completion_skips_backend_discovery_and_can_retry(stub_runtime, monkeypatch, boundary):
    state = stub_runtime
    contents = state.model.read_bytes()
    cancelled = []
    lookups = []
    original_open = Path.open
    original_backend = qr._find_backend

    class Stream(io.BytesIO):
        def read(self, size=-1):
            result = super().read(size)
            if not result and boundary == "eof":
                cancelled.append(True)
            return result

        def close(self):
            if boundary == "close":
                cancelled.append(True)
            super().close()

    stream = Stream(contents)
    monkeypatch.setattr(Path, "open", lambda *a, **k: stream)

    def unavailable_backend():
        lookups.append(True)
        raise qr.RuntimeUnavailable("Installed quality runtime was not found")

    monkeypatch.setattr(qr, "_find_backend", unavailable_backend)
    assert state.runtime.start(cancelled=lambda: bool(cancelled)) is None
    assert stream.closed
    assert lookups == state.spawns == state.probes == state.events == []
    assert state.runtime._process is state.runtime._socket_directory is None
    monkeypatch.setattr(Path, "open", original_open)
    monkeypatch.setattr(qr, "_find_backend", original_backend)
    assert state.runtime.start().timeout == 20
    assert len(state.spawns) == 1


@pytest.mark.parametrize("operation", ["stat", "open", "read"])
@pytest.mark.parametrize("failure", [PermissionError, FileNotFoundError, OSError])
def test_asset_io_failure_is_sanitized_without_discovery_and_recovers(stub_runtime, monkeypatch, operation, failure):
    state = stub_runtime
    original_stat, original_open = Path.stat, Path.open
    lookups = []
    original_backend = qr._find_backend

    def fail(*args, **kwargs):
        raise failure("/private/model/path: diagnostic")

    class BrokenStream(io.BytesIO):
        def read(self, size=-1):
            return fail()

    stream = BrokenStream(b"fixture")
    if operation == "stat":
        def stat(path, *args, **kwargs):
            if path == state.model:
                return fail()
            return original_stat(path, *args, **kwargs)
        monkeypatch.setattr(Path, "stat", stat)
    else:
        def open_file(path, *args, **kwargs):
            if path == state.model:
                return fail() if operation == "open" else stream
            return original_open(path, *args, **kwargs)
        monkeypatch.setattr(Path, "open", open_file)
    monkeypatch.setattr(qr, "_find_backend", lambda: lookups.append(True))
    with pytest.raises(qr.RuntimeUnavailable, match="^Quality model is missing or invalid$") as error:
        state.runtime.start()
    assert error.value.__cause__ is None
    assert lookups == state.spawns == state.probes == state.events == []
    assert state.runtime._process is state.runtime._socket_directory is None
    if operation == "read":
        assert stream.closed
    monkeypatch.setattr(Path, "stat", original_stat)
    monkeypatch.setattr(Path, "open", original_open)
    monkeypatch.setattr(qr, "_find_backend", original_backend)
    assert state.runtime.start().timeout == 20


def install_backend(home, version, mode=0o700, kind="file"):
    path = home / ".lmstudio/extensions/backends" / f"llama.cpp-linux-x86_64-vulkan-avx2-{version}" / "llama-server"
    path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "directory":
        path.mkdir()
    elif kind == "dangling":
        path.symlink_to(path.parent / "missing")
    else:
        path.write_bytes(b"never executed")
        path.chmod(mode)
    return path


@pytest.mark.parametrize("kind", ["directory", "dangling", "nonexecutable", "deleted"])
def test_discovery_ignores_unusable_higher_versions_without_caching(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    valid = install_backend(tmp_path, "2.9.10")
    newest = install_backend(tmp_path, "2.50.0", mode=0o600 if kind == "nonexecutable" else 0o700,
                             kind=kind if kind in ("directory", "dangling") else "file")
    if kind == "deleted":
        newest.unlink()
    assert qr._find_backend() == valid
    valid.unlink()
    with pytest.raises(qr.RuntimeUnavailable, match="^Installed quality runtime was not found$"):
        qr._find_backend()
    valid.write_bytes(b"reappeared")
    valid.chmod(0o700)
    assert qr._find_backend() == valid


@pytest.mark.parametrize("version", ["2.50", "2.50.0-beta", "2.50.0.extra", "latest", "-1.0.0"])
def test_discovery_ignores_non_release_version_names(tmp_path, monkeypatch, version):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    expected = install_backend(tmp_path, "2.10.0")
    install_backend(tmp_path, version)
    assert qr._find_backend() == expected


def test_discovery_scan_error_is_sanitized_and_next_scan_recovers(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    expected = install_backend(tmp_path, "2.10.0")
    original_glob = Path.glob

    def broken_glob(*args, **kwargs):
        raise PermissionError("/private/backend/path")

    monkeypatch.setattr(Path, "glob", broken_glob)
    with pytest.raises(qr.RuntimeUnavailable, match="^Installed quality runtime was not found$") as error:
        qr._find_backend()
    assert error.value.__cause__ is None
    monkeypatch.setattr(Path, "glob", original_glob)
    assert qr._find_backend() == expected
