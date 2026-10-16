"""Desktop launcher resources; no GUI launch or machine-specific writes."""
import configparser
from pathlib import Path
import subprocess
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def test_desktop_template_uses_project_launcher_and_local_icon(tmp_path):
    template = ROOT / "assets/translitter.desktop.in"
    assert template.is_file(), "Desktop launcher template is missing"
    text = template.read_text().replace("@PROJECT_DIR@", str(ROOT))
    config = configparser.ConfigParser(interpolation=None)
    config.read_string(text)
    entry = config["Desktop Entry"]
    assert entry["Type"] == "Application"
    assert entry["Name"] == "Translitter"
    assert entry["Exec"] == f'"{ROOT}/start.sh"'
    assert entry["Path"] == str(ROOT)
    assert entry["Icon"] == str(ROOT / "assets/translitter.svg")
    assert entry["Terminal"] == "false"
    assert entry["StartupNotify"] == "true"
    assert "@PROJECT_DIR@" not in text
    desktop = tmp_path / "translitter.desktop"
    desktop.write_text(text)
    # Optional external validator; portability does not require GNOME tools.
    import shutil
    validator = shutil.which("desktop-file-validate")
    if validator:
        subprocess.run([validator, str(desktop)], check=True, capture_output=True)


def test_desktop_icon_is_self_contained_white_black_blue_svg():
    path = ROOT / "assets/translitter.svg"
    assert path.is_file(), "Desktop icon is missing"
    root = ET.fromstring(path.read_text())
    assert root.tag.endswith("svg")
    text = path.read_text().lower()
    assert all(color in text for color in ("#ffffff", "#111827", "#2563eb"))
    assert "<image" not in text and "<script" not in text and "href=" not in text
