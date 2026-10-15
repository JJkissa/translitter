"""Explicit opt-in only: opens both USB mics and plays translated fixtures."""
import os
from pathlib import Path
import subprocess
import pytest


@pytest.mark.skipif(os.environ.get("TRANSLITTER_HARDWARE_TESTS") != "1", reason="Opt-in real USB capture/playback")
def test_real_usb_pipeline_and_route_isolation():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([str(root / "run-python.sh"), str(root / "scripts/verify_usb.py")],
                            cwd=root, capture_output=True, text=True, timeout=75)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS_BOTH_TRANSLATED_PIPELINES_AND_ISOLATED_USB_PLAYBACK" in result.stdout


@pytest.mark.skipif(os.environ.get("TRANSLITTER_HARDWARE_TESTS") != "1", reason="Opt-in native GUI with real USB audio")
def test_real_native_gui_model_and_headset_interactions():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([str(root / "run-python.sh"), str(root / "scripts/verify_gui.py")],
                            cwd=root, capture_output=True, text=True, timeout=75)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "PASS_NATIVE_GUI_BOTH_PIPELINES_STOP_RESTART_AND_SILENCE" in result.stdout

