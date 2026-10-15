"""Repeatable launcher, including from outside the project directory."""
from pathlib import Path
import subprocess


def test_launcher_is_executable_and_resolves_its_project(tmp_path):
    root = Path(__file__).resolve().parents[1]
    launcher = root / "start.sh"
    assert launcher.is_file(), "Native application launcher is missing"
    assert launcher.stat().st_mode & 0o111
    result = subprocess.run([str(launcher), "--help"], cwd=tmp_path,
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Local two-headset Finnish" in result.stdout
