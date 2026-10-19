"""Packaged runtime selection must not depend on LM Studio or agent files."""
import subprocess
from pathlib import Path
import pytest


def make_bundle(root):
    (root / 'appimage-build.json').write_text('{"layout":1}')
    for kind in ('cpu', 'vulkan'):
        path = root / 'runtime' / kind / 'llama-server'
        path.parent.mkdir(parents=True)
        path.write_text('placeholder for subprocess double')
        path.chmod(0o755)


def test_packaged_runtime_prefers_detected_nvidia_device(tmp_path, monkeypatch):
    from translitter.package_runtime import packaged_backend
    make_bundle(tmp_path)
    seen = []
    def probe(args, **kwargs):
        seen.append((args, kwargs))
        return subprocess.CompletedProcess(args, 0, 'Available devices:\n  Vulkan0: Intel Graphics\n  Vulkan1: NVIDIA GPU\n', '')
    monkeypatch.setattr(subprocess, 'run', probe)
    monkeypatch.setenv('LLAMA_MODEL_URL', 'must-not-be-inherited')
    path, flags = packaged_backend(tmp_path)
    assert path == tmp_path / 'runtime/vulkan/llama-server'
    assert flags == ('--device', 'Vulkan1')
    assert seen[0][0][-1] == '--list-devices'
    assert 'LLAMA_MODEL_URL' not in seen[0][1]['env']


def test_packaged_runtime_uses_cpu_when_gpu_probe_fails(tmp_path, monkeypatch):
    from translitter.package_runtime import packaged_backend
    make_bundle(tmp_path)
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: subprocess.CompletedProcess(a, 1, '', 'no driver'))
    path, flags = packaged_backend(tmp_path)
    assert path == tmp_path / 'runtime/cpu/llama-server'
    assert flags == ('--device', 'none', '--n-gpu-layers', '0')


def test_package_missing_cpu_backend_fails_instead_of_searching_lm_studio(tmp_path):
    from translitter.package_runtime import packaged_backend, PackageRuntimeError
    (tmp_path / 'appimage-build.json').write_text('{}')
    with pytest.raises(PackageRuntimeError):
        packaged_backend(tmp_path)


def test_source_checkout_keeps_its_existing_backend_discovery(tmp_path):
    from translitter.package_runtime import packaged_backend
    assert packaged_backend(tmp_path) is None
