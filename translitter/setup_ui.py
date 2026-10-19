"""Native, explicit first-run model setup. Never opens microphone/playback."""
from pathlib import Path
from threading import Event
import ctypes

from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QProgressBar, QPushButton, QVBoxLayout)

from .languages import LANGUAGE_NAMES
from .model_setup import load_catalog, required_assets, install_assets, SetupCancelled
from .theme import apply_theme


def cuda_is_available():
    try:
        ctypes.CDLL('libcublas.so.12')
        ctypes.CDLL('libcudnn.so.9')
        import ctranslate2
        return ctranslate2.get_cuda_device_count() > 0
    except (ImportError, OSError, RuntimeError):
        return False


def assets_present(paths, catalog, keys):
    for key in keys:
        for asset in catalog[key]:
            path = paths.data_root / asset.relative_path
            try:
                if (path.is_symlink() or not path.resolve().is_relative_to(paths.data_root.resolve())
                        or not path.is_file() or path.stat().st_size != asset.bytes):
                    return False
            except OSError:
                return False
    return True


class InstallWorker(QThread):
    progress = Signal(int, object, object, str)

    def __init__(self, revision, paths, catalog, keys, source_root, download, parent=None):
        super().__init__(parent)
        self.revision, self.paths, self.catalog, self.keys = revision, paths, catalog, keys
        self.source_root, self.download = source_root, download
        self.cancelled = Event()
        self.success = False
        self.problem = ''

    def run(self):
        try:
            install_assets(self.paths.data_root, self.catalog, self.keys,
                           source_root=self.source_root, download=self.download,
                           cancelled=self.cancelled.is_set,
                           progress=lambda done,total,message:self.progress.emit(self.revision,done,total,message))
            self.success = not self.cancelled.is_set()
            if not self.success:
                self.problem = 'Setup cancelled; completed verified files were kept'
        except SetupCancelled:
            self.problem = 'Setup cancelled; completed verified files were kept'
        except Exception as error:
            self.problem = f'Model setup failed: {error}'


class SetupDialog(QDialog):
    def __init__(self, paths, *, catalog=None, cuda_available=None, parent=None):
        super().__init__(parent)
        self.paths = paths
        self.catalog = load_catalog(paths.bundle_root) if catalog is None else catalog
        self.cuda_available = cuda_is_available() if cuda_available is None else cuda_available
        self.worker = None
        self._revision = 0
        self._close_pending = False
        self._verified_keys = set()
        self.setWindowTitle('Translitter — model setup')
        self.resize(780, 650)
        apply_theme(self)
        page = QVBoxLayout(self)
        page.setContentsMargins(24,24,24,24)
        page.setSpacing(12)
        location = str(paths.data_root)
        home = str(Path.home())
        if location.startswith(home + '/'):
            location = '~/' + location[len(home)+1:]
        label = QLabel('Model storage: ' + location)
        label.setWordWrap(True)
        page.addWidget(label)
        form = QFormLayout()
        self.speech_model = QComboBox()
        self.speech_model.addItem('Whisper small — smaller download, recommended for CPU', 'whisper-small')
        self.speech_model.addItem('Whisper turbo — larger download, recommended for NVIDIA', 'whisper-turbo')
        self.device = QComboBox()
        self.device.addItem('CPU', 'cpu')
        if self.cuda_available:
            self.device.addItem('NVIDIA CUDA', 'cuda')
        form.addRow('Speech recognition', self.speech_model)
        form.addRow('Processing', self.device)
        page.addLayout(form)
        hardware = QLabel('NVIDIA CUDA available; CPU is also supported.' if self.cuda_available else
                          'CPU available. NVIDIA CUDA libraries/driver were not detected.')
        hardware.setWordWrap(True)
        page.addWidget(hardware)
        group = QGroupBox('Installed languages')
        row = QHBoxLayout(group)
        self.languages = {}
        for code, name in LANGUAGE_NAMES.items():
            box = QCheckBox(name)
            box.setChecked(code in ('fi','en'))
            box.setEnabled(code not in ('fi','en'))
            self.languages[code] = box
            row.addWidget(box)
            box.toggled.connect(self._changed)
        page.addWidget(group)
        self.quality = QCheckBox('Install optional TranslateGemma 4B weights')
        self.quality.setToolTip('Additional large download. Uses the bundled local runtime; LM Studio is not required. Quality and latency vary.')
        page.addWidget(self.quality)
        terms = QLabel('Publisher terms: <a href="https://huggingface.co/Helsinki-NLP">Marian</a> · '
                       '<a href="https://huggingface.co/rhasspy/piper-voices">Piper voices</a> · '
                       '<a href="https://ai.google.dev/gemma/terms">Gemma</a>')
        terms.setOpenExternalLinks(True)
        page.addWidget(terms)
        warning = QLabel('Russian Irina voice: its publisher card lists the dataset license as unknown. Review the card before installing.')
        warning.setWordWrap(True)
        page.addWidget(warning)
        self.size_label = QLabel()
        self.size_label.setWordWrap(True)
        page.addWidget(self.size_label)
        self.license_checkbox = QCheckBox('I have reviewed the terms and approve installing the selected files.')
        page.addWidget(self.license_checkbox)
        buttons = QHBoxLayout()
        self.download_button = QPushButton('Download selected files')
        self.import_button = QPushButton('Import existing models…')
        buttons.addWidget(self.download_button)
        buttons.addWidget(self.import_button)
        page.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setRange(0,1000)
        self.progress.setValue(0)
        page.addWidget(self.progress)
        self.status = QLabel('Choose models, review the terms, then download or import. No microphone is opened during setup.')
        self.status.setWordWrap(True)
        page.addWidget(self.status)
        final = QHBoxLayout()
        self.cancel_button = QPushButton('Cancel installation')
        self.close_button = QPushButton('Close')
        self.open_button = QPushButton('Open Translitter')
        self.open_button.setObjectName('primary')
        final.addWidget(self.cancel_button)
        final.addStretch()
        final.addWidget(self.close_button)
        final.addWidget(self.open_button)
        page.addLayout(final)
        self.download_button.clicked.connect(lambda:self.start_install(download=True))
        self.import_button.clicked.connect(self._choose_import)
        self.cancel_button.clicked.connect(self._cancel)
        self.close_button.clicked.connect(self.reject)
        self.open_button.clicked.connect(self._open)
        self.license_checkbox.toggled.connect(self._changed)
        self.speech_model.currentIndexChanged.connect(self._changed)
        self.device.currentIndexChanged.connect(self._changed)
        self.quality.toggled.connect(self._changed)
        try:
            saved = self.paths.load_settings()
        except ValueError as error:
            saved = None
            self.status.setText(str(error))
        if saved:
            self.speech_model.setCurrentIndex(self.speech_model.findData(saved['speech_model']))
            index = self.device.findData(saved['device'])
            self.device.setCurrentIndex(index if index >= 0 else 0)
            for code, box in self.languages.items():
                if code not in ('fi','en'):
                    box.setChecked(code in saved['languages'])
            self.quality.setChecked(saved['quality'])
            keys = required_assets(saved['languages'],saved['speech_model'],saved['quality'])
            if assets_present(self.paths,self.catalog,keys):
                self._verified_keys = set(keys)
        self._changed()

    def settings(self):
        return {'speech_model':self.speech_model.currentData(), 'device':self.device.currentData(),
                'languages':[code for code,box in self.languages.items() if box.isChecked()],
                'quality':self.quality.isChecked()}

    def selected_keys(self):
        settings=self.settings()
        return required_assets(settings['languages'],settings['speech_model'],settings['quality'])

    def _changed(self, *_):
        if not hasattr(self,'open_button'):
            return
        busy = self.worker is not None
        keys=self.selected_keys()
        total=sum(asset.bytes for key in keys for asset in self.catalog[key])
        self.size_label.setText(f'Selected weights: {total / (1024**3):.2f} GiB ({total:,} bytes). Existing valid files are verified and reused.')
        consent=self.license_checkbox.isChecked()
        self.download_button.setEnabled(not busy and consent)
        self.import_button.setEnabled(not busy and consent)
        self.open_button.setEnabled(not busy and set(keys).issubset(self._verified_keys))
        self.cancel_button.setEnabled(busy)
        for widget in (self.speech_model,self.device,self.quality,self.license_checkbox):
            widget.setEnabled(not busy)
        for code, box in self.languages.items():
            box.setEnabled(not busy and code not in ('fi','en'))

    def _choose_import(self):
        directory=QFileDialog.getExistingDirectory(self,'Select the folder containing models/')
        if directory:
            self.start_install(source_root=Path(directory),download=False)

    def start_install(self, *, source_root=None, download=False):
        if self.worker is not None or not self.license_checkbox.isChecked():
            return False
        if not download and source_root is None:
            return False
        self._revision += 1
        self._pending_settings=self.settings()
        worker=InstallWorker(self._revision,self.paths,self.catalog,self.selected_keys(),source_root,download,self)
        self.worker=worker
        worker.progress.connect(self._on_progress)
        worker.finished.connect(self._installed)
        self.progress.setValue(0)
        self.status.setText('Verifying and installing selected model files…')
        self._changed()
        worker.start()
        return True

    def _on_progress(self, revision, done, total, message):
        if self.worker is not None and revision==self._revision:
            self.progress.setValue(min(1000,max(0,int(done*1000//total))) if total else 0)
            self.status.setText(message)

    def _cancel(self):
        if self.worker is not None:
            self.worker.cancelled.set()
            self.status.setText('Cancelling setup; waiting for the current file operation…')

    def _installed(self):
        worker=self.worker
        if worker is None or self.sender() is not worker:
            return
        self.worker=None
        if worker.success:
            try:
                self.paths.save_settings(self._pending_settings)
                self._verified_keys.update(worker.keys)
                self.progress.setValue(1000)
                self.status.setText('Ready · selected model files verified. Open Translitter to select your headsets.')
            except (OSError,ValueError) as error:
                self.status.setText(f'Settings were not saved: {error}')
        else:
            self.status.setText(worker.problem)
        worker.deleteLater()
        self._changed()
        if self._close_pending:
            super().reject()

    def _open(self):
        if self.open_button.isEnabled():
            try:
                self.paths.save_settings(self.settings())
            except (OSError, ValueError) as error:
                self.status.setText(f'Settings were not saved: {error}. Check your settings and storage permissions, then try Open Translitter again.')
                return
            self.accept()

    def reject(self):
        if self.worker is not None:
            self._close_pending=True
            self._cancel()
            return
        super().reject()

    def closeEvent(self,event):
        if self.worker is not None:
            self._close_pending=True
            self._cancel()
            event.ignore()
        else:
            super().closeEvent(event)
