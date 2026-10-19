"""Select a pinned application-bundled llama.cpp backend, with a CPU fallback."""
import os
from pathlib import Path
import re
import subprocess


class PackageRuntimeError(RuntimeError):
    pass


def packaged_backend(bundle_root=None):
    root = Path(__file__).resolve().parent.parent if bundle_root is None else Path(bundle_root)
    if not (root / 'appimage-build.json').is_file():
        return None
    cpu = root / 'runtime/cpu/llama-server'
    if not cpu.is_file() or not os.access(cpu, os.X_OK):
        raise PackageRuntimeError('The packaged local runtime is missing; download a fresh application package')
    vulkan = root / 'runtime/vulkan/llama-server'
    if vulkan.is_file() and os.access(vulkan, os.X_OK):
        try:
            result = subprocess.run([str(vulkan), '--list-devices'], capture_output=True, text=True,
                                    timeout=10, stdin=subprocess.DEVNULL,
                                    env={key:value for key,value in os.environ.items() if not key.startswith('LLAMA_')})
            devices = re.findall(r'^\s*(Vulkan\d+):\s*(.+)$', result.stdout, re.MULTILINE)
            if result.returncode == 0 and devices:
                device = next((identifier for identifier, name in devices if 'NVIDIA' in name.upper()), devices[0][0])
                return vulkan, ('--device', device)
        except (OSError, subprocess.TimeoutExpired, UnicodeError):
            pass
    return cpu, ('--device', 'none', '--n-gpu-layers', '0')
