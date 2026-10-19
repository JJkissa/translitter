"""Trusted isolated entrypoint. Explicit model setup uses stdlib HTTPS, not HF."""
from pathlib import Path
import os
import runpy
import sys


def configure(appdir=None):
    root = Path(appdir).resolve() if appdir else Path(__file__).resolve().parents[3]
    library = root / 'usr/lib'
    python = library / 'python3.12'
    site = python / 'site-packages'
    app = library / 'translitter/app'
    # Keep only bundled standard library entries; never keep a development/CWD path.
    standard = []
    for entry in sys.path:
        if not entry:
            continue
        path = Path(entry).resolve()
        if (path == python or python in path.parents) and not (path == site or site in path.parents):
            standard.append(str(path))
    sys.path[:] = [str(app), str(site), *standard]
    for key in ('PYTHONPATH', 'PYTHONHOME', 'VIRTUAL_ENV', 'LD_PRELOAD',
                'QML2_IMPORT_PATH', 'QML_IMPORT_PATH'):
        os.environ.pop(key, None)
    directories = [library / 'native', site / 'torch/lib',
                   *sorted((site / 'nvidia').glob('*/lib')), site / 'PySide6/Qt/lib']
    os.environ['LD_LIBRARY_PATH'] = ':'.join(str(p) for p in directories if p.is_dir())
    plugins = site / 'PySide6/Qt/plugins'
    os.environ['QT_PLUGIN_PATH'] = str(plugins)
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(plugins / 'platforms')
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1',
                      ANONYMIZED_TELEMETRY='False', TOKENIZERS_PARALLELISM='false',
                      PYTHONDONTWRITEBYTECODE='1')
    certificate = root / 'usr/share/translitter/ca-certificates.crt'
    if certificate.is_file():
        os.environ['SSL_CERT_FILE'] = str(certificate)
    sys.dont_write_bytecode = True
    return app


def main():
    configure()
    runpy.run_module('translitter.packaged', run_name='__main__')


if __name__ == '__main__':
    main()
