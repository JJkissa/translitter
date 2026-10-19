"""Packaged application state must never live inside the read-only AppImage."""
from pathlib import Path
import pytest


def test_user_data_is_separate_from_bundle_and_survives_bundle_change(tmp_path):
    from translitter.package_paths import AppPaths
    home = tmp_path / 'home'
    first = tmp_path / 'first.AppDir'
    second = tmp_path / 'updated.AppDir'
    paths = AppPaths.discover(bundle_root=first, home=home, environ={})
    assert paths.data_root == home / '.local/share/translitter'
    assert paths.bundle_root == first
    paths.prepare()
    marker = paths.data_root / 'models' / 'kept'
    marker.parent.mkdir(exist_ok=True)
    marker.write_text('model stays here')
    updated = AppPaths.discover(bundle_root=second, home=home, environ={})
    updated.prepare()
    assert (updated.data_root / 'models/kept').read_text() == 'model stays here'
    assert not first.exists() and not second.exists()


def test_xdg_location_and_explicit_data_directory(tmp_path):
    from translitter.package_paths import AppPaths
    env = {'XDG_DATA_HOME': str(tmp_path / 'data')}
    assert AppPaths.discover(home=tmp_path, environ=env).data_root == tmp_path / 'data/translitter'
    assert AppPaths.discover(data_root=tmp_path / 'chosen', environ=env).data_root == tmp_path / 'chosen'


def test_relative_xdg_value_is_not_used_as_an_installation_location(tmp_path):
    from translitter.package_paths import AppPaths
    assert AppPaths.discover(home=tmp_path, environ={'XDG_DATA_HOME':'relative'}).data_root == tmp_path / '.local/share/translitter'


def test_settings_are_validated_and_saved_privately_outside_bundle(tmp_path):
    from translitter.package_paths import AppPaths
    paths = AppPaths.discover(bundle_root=tmp_path / 'bundle', data_root=tmp_path / 'data')
    settings = {'speech_model':'whisper-small', 'device':'cpu', 'languages':['fi','en','ru'], 'quality':False}
    paths.save_settings(settings)
    assert paths.load_settings() == settings
    assert paths.settings_file.stat().st_mode & 0o777 == 0o600
    assert not paths.bundle_root.exists()
    for invalid in ({**settings, 'device':'remote'}, {**settings,'quality':'yes'},
                    {**settings, 'languages':['fi','xx']}, {**settings,'conversation':'secret'}):
        with pytest.raises(ValueError):
            paths.save_settings(invalid)
    assert paths.load_settings() == settings


def test_setup_refuses_symlink_data_or_settings(tmp_path):
    from translitter.package_paths import AppPaths
    other = tmp_path / 'other'
    other.mkdir()
    linked = tmp_path / 'linked'
    linked.symlink_to(other, target_is_directory=True)
    with pytest.raises(ValueError):
        AppPaths.discover(data_root=linked).prepare()
    paths = AppPaths.discover(data_root=tmp_path / 'data')
    paths.prepare()
    paths.settings_file.symlink_to(other / 'private')
    with pytest.raises(ValueError):
        paths.load_settings()
