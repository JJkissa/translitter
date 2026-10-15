"""Runtime regression: the project launcher must fix Qt without system edits."""
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_launcher_exposes_native_qt_and_gpu_libraries():
    launcher = ROOT / "run-python.sh"
    assert launcher.is_file(), "The project-local runtime launcher is missing"
    code = (
        "import ctypes; "
        "ctypes.CDLL('libcublas.so.12'); "
        "ctypes.CDLL('libcudnn.so.9'); "
        "from PySide6.QtWidgets import QApplication; "
        "a=QApplication([]); "
        "assert a.platformName() == 'xcb'; "
        "assert a.primaryScreen() is not None; "
        "print('NATIVE_QT_AND_GPU_LIBRARIES_READY')"
    )
    result = subprocess.run(
        ["bash", str(launcher), "-c", code], cwd=ROOT,
        text=True, capture_output=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "NATIVE_QT_AND_GPU_LIBRARIES_READY" in result.stdout
