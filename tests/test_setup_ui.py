"""Native first-run setup: explicit consent, real verified import and safe close."""
import hashlib
from pathlib import Path
import threading
import time

import pytest

from PySide6.QtCore import Qt


@pytest.mark.parametrize('error', [
    OSError('Settings storage is read-only'),
    ValueError('Settings location is unsafe'),
], ids=['oserror', 'valueerror'])
def test_open_recovers_from_settings_save_failure_and_allows_retry(qtbot, tmp_path, monkeypatch, error):
    ui, paths, source, catalog = fixture_setup(tmp_path, monkeypatch)
    dialog = ui.SetupDialog(paths, catalog=catalog, cuda_available=True)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.license_checkbox.setChecked(True)
    assert dialog.start_install(source_root=source, download=False)
    qtbot.waitUntil(lambda: dialog.worker is None and dialog.open_button.isEnabled(), timeout=5000)
    previous_bytes = paths.settings_file.read_bytes()
    previous_mode = paths.settings_file.stat().st_mode & 0o777
    assert previous_mode == 0o600
    assert paths.load_settings()['device'] == 'cpu'
    dialog.device.setCurrentIndex(dialog.device.findData('cuda'))
    accepted = []
    dialog.accepted.connect(lambda: accepted.append(True))
    save_settings = type(paths).save_settings
    attempts = []

    def fail_once_then_save(self, settings):
        attempts.append(settings)
        if len(attempts) == 1:
            raise error
        return save_settings(self, settings)

    monkeypatch.setattr(type(paths), 'save_settings', fail_once_then_save)
    assert dialog.open_button.isEnabled()
    with qtbot.captureExceptions() as exceptions:
        qtbot.mouseClick(dialog.open_button, Qt.LeftButton)
    assert not exceptions
    assert len(attempts) == 1
    assert dialog.isVisible()
    assert dialog.result() != dialog.DialogCode.Accepted
    assert not accepted
    assert 'Settings were not saved' in dialog.status.text()
    assert str(error) in dialog.status.text()
    assert 'try' in dialog.status.text().lower()
    assert paths.settings_file.read_bytes() == previous_bytes
    assert paths.settings_file.stat().st_mode & 0o777 == previous_mode
    assert paths.load_settings()['device'] == 'cpu'
    assert dialog.worker is None and dialog.open_button.isEnabled()

    with qtbot.captureExceptions() as retry_exceptions:
        qtbot.mouseClick(dialog.open_button, Qt.LeftButton)
    assert not retry_exceptions
    assert len(attempts) == 2
    assert accepted == [True]
    assert dialog.result() == dialog.DialogCode.Accepted
    assert not dialog.isVisible()
    assert paths.load_settings()['device'] == 'cuda'
    assert paths.settings_file.stat().st_mode & 0o777 == previous_mode


def fixture_setup(tmp_path, monkeypatch):
    from translitter import setup_ui
    from translitter.package_paths import AppPaths
    from translitter.model_setup import AssetFile
    paths = AppPaths.discover(bundle_root=tmp_path / 'bundle', data_root=tmp_path / 'data')
    content = b'verified fixture model'
    source = tmp_path / 'source'
    (source / 'models').mkdir(parents=True)
    (source / 'models/fixture.bin').write_bytes(content)
    catalog = {'fixture':(AssetFile('models/fixture.bin','https://publisher.example/fixture',len(content),hashlib.sha256(content).hexdigest()),)}
    monkeypatch.setattr(setup_ui,'required_assets',lambda *a,**k: ('fixture',))
    return setup_ui, paths, source, catalog


def test_setup_never_installs_or_opens_capture_on_construction(qtbot, tmp_path, monkeypatch):
    ui, paths, source, catalog = fixture_setup(tmp_path, monkeypatch)
    calls=[]
    monkeypatch.setattr(ui,'install_assets',lambda *a,**k:calls.append((a,k)))
    dialog=ui.SetupDialog(paths,catalog=catalog,cuda_available=False)
    qtbot.addWidget(dialog)
    dialog.show()
    assert not calls and dialog.worker is None
    assert not dialog.download_button.isEnabled() and not dialog.import_button.isEnabled()
    assert not dialog.open_button.isEnabled()
    assert not (paths.data_root / 'models').exists()
    dialog.license_checkbox.setChecked(True)
    assert dialog.download_button.isEnabled() and dialog.import_button.isEnabled()


def test_setup_imports_real_verified_files_and_saves_only_after_completion(qtbot, tmp_path, monkeypatch):
    ui, paths, source, catalog = fixture_setup(tmp_path, monkeypatch)
    dialog=ui.SetupDialog(paths,catalog=catalog,cuda_available=False)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.license_checkbox.setChecked(True)
    dialog.start_install(source_root=source, download=False)
    qtbot.waitUntil(lambda: dialog.worker is None and dialog.open_button.isEnabled(),timeout=5000)
    assert (paths.data_root / 'models/fixture.bin').read_bytes() == b'verified fixture model'
    assert paths.load_settings()['device']=='cpu'
    assert 'verified' in dialog.status.text().lower()
    qtbot.mouseClick(dialog.open_button,Qt.LeftButton)
    assert dialog.result()==dialog.DialogCode.Accepted


def test_closing_setup_cancels_and_waits_for_its_owned_worker(qtbot, tmp_path, monkeypatch):
    ui, paths, source, catalog = fixture_setup(tmp_path, monkeypatch)
    entered=threading.Event()
    def install(*args, cancelled, **kwargs):
        entered.set()
        while not cancelled():
            time.sleep(.005)
        raise ui.SetupCancelled('Cancelled')
    monkeypatch.setattr(ui,'install_assets',install)
    dialog=ui.SetupDialog(paths,catalog=catalog,cuda_available=False)
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.license_checkbox.setChecked(True)
    dialog.start_install(source_root=source, download=False)
    qtbot.waitUntil(entered.is_set)
    worker=dialog.worker
    finished=[]
    worker.finished.connect(lambda: finished.append(worker.isFinished()))
    dialog.close()
    qtbot.waitUntil(lambda: bool(finished) and not dialog.isVisible(),timeout=5000)
    assert finished == [True] and dialog.worker is None
    assert not paths.settings_file.exists()
