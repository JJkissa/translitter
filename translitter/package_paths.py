"""Read-only application resources and update-safe, user-owned application data."""
from dataclasses import dataclass
import json
import os
from pathlib import Path
import stat
import tempfile


def validate_settings(settings):
    if not isinstance(settings, dict) or set(settings) != {'speech_model', 'device', 'languages', 'quality'}:
        raise ValueError('Invalid application settings')
    if settings['speech_model'] not in ('whisper-small', 'whisper-turbo') or settings['device'] not in ('cpu', 'cuda'):
        raise ValueError('Select an installed speech model and CPU or NVIDIA CUDA')
    languages = settings['languages']
    if (not isinstance(languages, list) or len(languages) < 2
            or any(not isinstance(code, str) or code not in ('fi', 'en', 'ru', 'uk') for code in languages)
            or len(set(languages)) != len(languages) or type(settings['quality']) is not bool):
        raise ValueError('Invalid language or translator selection')
    return dict(settings, languages=list(languages))


@dataclass(frozen=True)
class AppPaths:
    bundle_root: Path
    data_root: Path

    @classmethod
    def discover(cls, *, bundle_root=None, data_root=None, home=None, environ=None):
        environ = os.environ if environ is None else environ
        home = Path.home() if home is None else Path(home)
        if data_root is None:
            xdg = Path(environ.get('XDG_DATA_HOME', ''))
            base = xdg if xdg.is_absolute() else home / '.local/share'
            data_root = base / 'translitter'
        return cls(Path(bundle_root) if bundle_root is not None else Path(__file__).resolve().parent.parent,
                   Path(data_root).expanduser().absolute())

    @property
    def settings_file(self):
        return self.data_root / 'settings.json'

    def prepare(self):
        if self.data_root.is_symlink():
            raise ValueError('Application data location must not be a symbolic link')
        self.data_root.mkdir(mode=0o700, parents=True, exist_ok=True)
        info = self.data_root.stat()
        if info.st_uid != os.geteuid() or info.st_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise ValueError('Application data location must be owned and writable only by you')

    def load_settings(self):
        self.prepare()
        if self.settings_file.is_symlink():
            raise ValueError('Application settings must not be a symbolic link')
        if not self.settings_file.exists():
            return None
        if self.settings_file.stat().st_size > 16384:
            raise ValueError('Invalid application settings')
        try:
            return validate_settings(json.loads(self.settings_file.read_text()))
        except (OSError, UnicodeError, json.JSONDecodeError):
            raise ValueError('Application settings could not be read; open setup again') from None

    def save_settings(self, settings):
        settings = validate_settings(settings)
        self.prepare()
        if self.settings_file.is_symlink():
            raise ValueError('Application settings must not be a symbolic link')
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', dir=self.data_root,
                                             prefix='.settings-', delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(settings, stream, indent=2)
                stream.write('\n')
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.settings_file)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
