"""The packaged owned process uses its own runtime and ordinary private IPC."""
from pathlib import Path
from test_quality_runtime import small_model, stub_runtime
from translitter import quality_runtime as qr


def test_owned_start_uses_packaged_backend_without_lm_studio(stub_runtime, monkeypatch):
    state = stub_runtime
    monkeypatch.setattr(qr, 'packaged_backend', lambda: (Path('/bundled/cpu/llama-server'), ('--device','none','--n-gpu-layers','0')), raising=False)
    state.runtime.start()
    arguments = state.spawns[0][0]
    assert arguments[0] == '/bundled/cpu/llama-server'
    assert arguments[-4:] == ['--device','none','--n-gpu-layers','0']


def test_owned_start_without_tmpdir_does_not_require_an_agent_home(stub_runtime, monkeypatch, tmp_path):
    monkeypatch.delenv('TMPDIR', raising=False)
    monkeypatch.setattr(qr.tempfile, 'gettempdir', lambda: str(tmp_path))
    stub_runtime.runtime.start()
    directory = Path(stub_runtime.runtime._socket_directory.name)
    assert directory.is_relative_to(tmp_path)
