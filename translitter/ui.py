"""Local native interpreter. Session generations gate every model response."""
from pathlib import Path
import numpy as np
from time import monotonic
from PySide6.QtCore import QObject, Signal, QTimer, Qt, QEvent
from PySide6.QtWidgets import (QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QComboBox, QCheckBox, QSlider, QPlainTextEdit, QProgressBar,
    QVBoxLayout, QHBoxLayout, QApplication)
from .core import MAX_TURN_SECONDS, MAX_TURN_SAMPLES, make_routes, Segmenter
from .theme import apply_theme
from .languages import LANGUAGE_NAMES, missing_assets, translation_steps

class _IdleAudio(QObject):
    """No-device backend for explicitly disabled startup (screenshots/tests)."""
    samples = Signal(int, object)
    level = Signal(int, float)
    failure = Signal(int, str)
    playing = Signal(int, bool)
    devices_changed = Signal()
    def headsets(self): return []
    def stop(self): pass
    def set_volume(self, volume): pass


def label(text, name=None):
    item = QLabel(text)
    item.setTextFormat(Qt.PlainText)
    if name:
        item.setObjectName(name)
    return item


class ParticipantCard(QFrame):
    def __init__(self, speaker):
        super().__init__()
        self.setObjectName("card")
        box = QVBoxLayout(self)
        box.setContentsMargins(20, 20, 20, 20)
        box.setSpacing(13)
        self.title = label(f"User {speaker + 1}", "participant")
        self.direction = label("", "section")
        box.addWidget(self.title)
        box.addWidget(self.direction)
        columns = QHBoxLayout()
        self.original, self.translated = QPlainTextEdit(), QPlainTextEdit()
        for title, editor in (
            ("ORIGINAL", self.original),
            ("TRANSLATED", self.translated)):
            column = QVBoxLayout()
            column.addWidget(label(title, "section"))
            editor.setReadOnly(True)
            editor.setMaximumBlockCount(100)

            column.addWidget(editor, 1)
            columns.addLayout(column, 1)
        box.addLayout(columns, 1)
        self.state = label("Waiting for conversation", "muted")
        box.addWidget(self.state)
        self.meter = QProgressBar()
        self.meter.setRange(0, 100)
        self.meter.setValue(0)
        self.meter.setTextVisible(False)
        self.meter.setFixedHeight(5)
        box.addWidget(self.meter)
        self.hold = QPushButton(f"Hold to talk  ·  F{speaker + 1}")
        self.hold.setObjectName("hold")
        self.hold.setToolTip(f"Hold while speaking; release to translate. Maximum {MAX_TURN_SECONDS} seconds; longer recordings are discarded.")
        self.hold.setEnabled(False)
        box.addWidget(self.hold)


class Window(QMainWindow):
    def __init__(self, root: Path, worker=None, audio=None, start_worker=True):
        super().__init__()
        self.root = Path(root)
        self.worker = worker
        if self.worker is None and start_worker:
            from .worker import ModelWorker
            self.worker = ModelWorker(self.root)
        if audio is None:
            if start_worker:
                from .audio import AudioHub
                audio = AudioHub(self)
            else:
                audio = _IdleAudio(self)
        self.audio = audio
        self.active = False
        self.generation = 0
        self.models_ready = False
        self._translation_backend = "marian"
        self._backend_supported = (hasattr(self.worker, "set_translation_backend")
                                   and hasattr(self.worker, "translation_ready"))
        self._translation_is_ready = True  # Base ready confirms unchanged Marian.
        self._translation_request = None
        self._translation_problem = ""
        self._closing = False
        self.segmenters = [Segmenter(), Segmenter()]
        self._held = [False, False]
        self._chunks = [[], []]
        self._sizes = [0, 0]
        self._capture_overflow = [False, False]
        self._busy = [0, 0]
        self._playback = [False, False]
        self._cooldown = [0.0, 0.0]
        self._deferred = [[], []]
        self._session_languages = ("fi", "en")
        self._session_mode = "ptt"
        self._display_languages = ("fi", "en")
        self.setWindowTitle("Translitter · Finnish ↔ English")
        self.resize(1280, 820)
        self.setMinimumSize(1000, 680)
        apply_theme(self)
        self._build()
        self.audio.devices_changed.connect(self._devices_changed)
        self.audio.failure.connect(self._audio_failure)
        self.audio.samples.connect(self._samples)
        self.audio.level.connect(self._level)
        self.audio.playing.connect(self._playing)
        self.text_only.toggled.connect(self._text_only_changed)
        for speaker, card in enumerate(self.cards):
            card.hold.pressed.connect(lambda n=speaker: self._press(n))
            card.hold.released.connect(lambda n=speaker: self._release(n))
        self.mode.currentIndexChanged.connect(self._selection_changed)
        QApplication.instance().installEventFilter(self)
        if self.worker is not None:
            self.worker.ready.connect(self._ready)
            if self._backend_supported:
                self.worker.translation_ready.connect(self._translation_ready)
            self.worker.loading.connect(self._loading)
            self.worker.result.connect(self._result)
            if hasattr(self.worker, "settled"):
                self.worker.settled.connect(self._settled)
            self.worker.progress.connect(self._progress)
            self.worker.error.connect(self._error)
        self.volume.valueChanged.connect(lambda value: self.audio.set_volume(value / 100))
        self.audio.set_volume(.45)
        self.refresh_devices()
        self._discovery = QTimer(self)
        self._discovery.setSingleShot(True)
        self._discovery.timeout.connect(self._reconcile_devices)
        self._discovery.start(800)
        self._close_poll = QTimer(self)
        self._close_poll.setInterval(50)
        self._close_poll.timeout.connect(self._poll_close)
        if start_worker and self.worker is not None:
            self.worker.start()

    def _build(self):
        central = QWidget()
        self.setCentralWidget(central)
        page = QVBoxLayout(central)
        page.setContentsMargins(28, 24, 28, 20)
        page.setSpacing(18)

        controls = QFrame()
        controls.setObjectName("controls")
        config = QVBoxLayout(controls)
        config.setContentsMargins(16, 14, 16, 14)
        select_row = QHBoxLayout()
        self.selectors = [QComboBox(), QComboBox()]
        for speaker, selector in enumerate(self.selectors):
            col = QVBoxLayout()
            col.addWidget(label(f"USER {speaker + 1} · HEADSET", "section"))
            selector.setMinimumWidth(290)
            selector.currentIndexChanged.connect(self._selection_changed)
            col.addWidget(selector)
            select_row.addLayout(col, 1)
        self.swap_button = QPushButton("Swap")
        self.swap_button.clicked.connect(self._swap)
        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_devices)
        select_row.addWidget(self.swap_button)
        select_row.addWidget(self.refresh_button)
        config.addLayout(select_row)
        language_row = QHBoxLayout()
        self.language_selectors = [QComboBox(), QComboBox()]
        for speaker, selector in enumerate(self.language_selectors):
            language_row.addWidget(label(f"User {speaker + 1} speaks", "section"))
            for code, name in LANGUAGE_NAMES.items():
                selector.addItem(name, code)
            selector.setCurrentIndex(selector.findData("fi" if speaker == 0 else "en"))
            selector.setToolTip("Spoken language; translations play in the other participant's language. Change while stopped.")
            language_row.addWidget(selector, 1)
            selector.currentIndexChanged.connect(self._languages_changed)
        config.addLayout(language_row)
        backend_row = QHBoxLayout()
        backend_row.addWidget(label("Translation", "section"))
        self.translation_backend = self.backend_selector = QComboBox()
        self.backend_selector.addItem("Marian (fast)", "marian")
        self.backend_selector.addItem("TranslateGemma 4B", "translategemma")
        self.backend_selector.setToolTip(
            "Local translation only. TranslateGemma translates directly without an English pivot; "
            "it is slower and does not guarantee better accuracy. Requires the optional installed "
            "runtime and model; no automatic downloads. Change while stopped.")
        self.backend_selector.currentIndexChanged.connect(self._translation_changed)
        backend_row.addWidget(self.backend_selector, 1)
        config.addLayout(backend_row)
        self.language_status = label("", "muted")
        self.language_status.setWordWrap(True)
        config.addWidget(self.language_status)
        transport = QHBoxLayout()
        self.start_button, self.stop_button = QPushButton("Start conversation"), QPushButton("Stop")
        self.start_button.setObjectName("primary")
        self.stop_button.setObjectName("danger")
        self.stop_button.setToolTip("Immediately stop microphones, headphone audio, and pending turns.")
        self.start_button.clicked.connect(self.start_session)
        self.stop_button.clicked.connect(self.stop_session)
        transport.addWidget(self.start_button)
        transport.addWidget(self.stop_button)
        self.mode = QComboBox()
        self.mode.addItem("Push to talk", "ptt")
        self.mode.addItem("Automatic", "auto")
        self.mode.setToolTip("Automatic mode may hear room bleed. Use closed headsets and take turns.")
        transport.addWidget(self.mode, 1)
        self.text_only = QCheckBox("Text only")
        self.text_only.setToolTip("Do not play synthesized translations through the headsets.")
        transport.addWidget(self.text_only)
        transport.addWidget(label("Volume", "muted"))
        self.volume = QSlider(Qt.Horizontal)
        self.volume.setRange(0, 80)
        self.volume.setValue(45)
        self.volume.setFixedWidth(100)
        self.volume.setToolTip("Headphone volume · default 45%, maximum 80%")
        transport.addWidget(self.volume)
        config.addLayout(transport)
        page.addWidget(controls)
        cards = QHBoxLayout()
        self.cards = [ParticipantCard(0), ParticipantCard(1)]
        for card in self.cards:
            cards.addWidget(card, 1)
        page.addLayout(cards, 1)
        self.status = label("Waiting for local models…" if self.worker else "Preview · inference and audio devices disabled", "badge")
        self.status.setWordWrap(True)
        page.addWidget(self.status)

        self._selection_changed()

    def _loading(self, message):
        if not self._closing and not self.models_ready:
            self.status.setText(message)

    def _ready(self):
        if self._closing or self.models_ready:
            return
        self.models_ready = True
        if self._translation_is_ready:
            self.status.setText("Ready · select two distinct USB headsets, then Start")
        self._selection_changed()

    def _translation_changed(self, *_):
        if self.active or self._closing or not self._backend_supported:
            self.backend_selector.blockSignals(True)
            self.backend_selector.setCurrentIndex(self.backend_selector.findData(self._translation_backend))
            self.backend_selector.blockSignals(False)
            return
        backend = self.backend_selector.currentData()
        if backend == self._translation_backend:
            return
        self._translation_backend = backend
        self._translation_is_ready = False
        self._translation_problem = ""
        self._translation_request = None
        try:
            self._translation_request = self.worker.set_translation_backend(backend)
            self.status.setText("Checking local translation backend · microphones remain off…")
        except Exception:
            self._translation_problem = "Local translation backend unavailable; select Marian or restart while stopped"
            self.status.setText(self._translation_problem)
        self._selection_changed()

    def _translation_ready(self, request, backend, ready, error):
        if (self._closing or self.active or request != self._translation_request
                or backend != self._translation_backend):
            return
        self._translation_is_ready = bool(ready)
        self._translation_problem = "" if ready else (error or "Local translation backend unavailable; select Marian")
        self.status.setText(self._translation_problem or "Local translation ready · press Start when ready")
        self._selection_changed()

    def _selected_routes(self):
        first, second = (selector.currentData() for selector in self.selectors)
        if first is None or second is None:
            raise ValueError("Connect and select two USB headsets")
        return make_routes(first, second, *(selector.currentData() for selector in self.language_selectors))

    def _languages_changed(self, *_):
        if self.active:
            # Session configuration is immutable, even for programmatic changes.
            for selector, code in zip(self.language_selectors, self._session_languages):
                selector.blockSignals(True)
                selector.setCurrentIndex(selector.findData(code))
                selector.blockSignals(False)
            return
        languages = tuple(selector.currentData() for selector in self.language_selectors)
        if languages != self._display_languages:
            for card in self.cards:
                card.original.clear()
                card.translated.clear()
            self._display_languages = languages
        self._selection_changed()

    def _language_problem(self):
        first, second = (selector.currentData() for selector in self.language_selectors)
        try:
            translation_steps(first, second)
        except ValueError as error:
            return str(error)
        missing = missing_assets(self.root, first, second, additional_only=True)
        if self._translation_backend == "translategemma":
            missing = tuple(path for path in missing if not path.startswith("translate-"))
        if missing:
            assets = sorted({path.split("/", 1)[0] for path in missing})
            return "Local language models not installed: " + ", ".join(assets) + ". Finnish ↔ English remains available; no automatic downloads."
        return ""

    def _selection_changed(self, *_):
        if self.active and self.mode.currentData() != self._session_mode:
            self.mode.blockSignals(True)
            self.mode.setCurrentIndex(self.mode.findData(self._session_mode))
            self.mode.blockSignals(False)
            self.status.setText("Stop the conversation before changing capture mode")
        for selector in self.selectors:
            headset = selector.currentData()
            selector.setToolTip(f"Microphone: {headset.microphone_id}\nHeadphones: {headset.output_id}" if headset else "No complete USB headset pair detected")
        try:
            self._selected_routes()
            valid = True
        except ValueError:
            valid = False
        problem = self._language_problem()
        valid = valid and not problem
        languages = tuple(selector.currentData() for selector in self.language_selectors)
        if all(code in LANGUAGE_NAMES for code in languages):
            first, second = languages
            self.setWindowTitle(f"Translitter · {LANGUAGE_NAMES[first]} ↔ {LANGUAGE_NAMES[second]}")

            for speaker, card in enumerate(self.cards):
                source, target = languages[speaker], languages[1 - speaker]
                card.title.setText(f"User {speaker + 1}  /  {LANGUAGE_NAMES[source]}")
                card.direction.setText(f"{LANGUAGE_NAMES[source].upper()} → {LANGUAGE_NAMES[target].upper()}")
        via_english = (self._translation_backend == "marian"
                       and languages[0] != languages[1] and "en" not in languages)
        note = ("Via English · accuracy may be lower." if via_english else
                "Direct local translation · no English pivot" if self._translation_backend == "translategemma"
                else "Direct translation")
        self.language_status.setText(f"{problem} {note}" if problem else note)
        self.start_button.setEnabled(self.models_ready and self._translation_is_ready
                                     and valid and not self.active and not self._closing)
        self.stop_button.setEnabled(self.active)
        for widget in (*self.selectors, *self.language_selectors, self.swap_button, self.refresh_button, self.mode):
            widget.setEnabled(not self.active and not self._closing)
        self.backend_selector.setEnabled(not self.active and not self._closing and self._backend_supported)
        for card in self.cards:
            card.hold.setEnabled(self.active and self.mode.currentData() == "ptt")

    def refresh_devices(self):
        if self.active or self._closing:
            return
        keys = [s.currentData().key if s.currentData() else None for s in self.selectors]
        pairs = self.audio.headsets()
        indices = [next((i for i, p in enumerate(pairs) if p.key == key), None)
                   for key in keys]
        # Reserve both surviving selections before assigning a missing device.
        # Independent positional defaults can otherwise duplicate the survivor.
        used = {i for i in indices if i is not None}
        for speaker, index in enumerate(indices):
            if index is None:
                default = min(speaker, len(pairs) - 1)
                index = (default if default not in used else
                         next((i for i in range(len(pairs)) if i not in used), default))
                indices[speaker] = index
                used.add(index)
        for speaker, selector in enumerate(self.selectors):
            selector.blockSignals(True)
            selector.clear()
            for pair in pairs:
                selector.addItem(pair.label, pair)
            selector.setCurrentIndex(indices[speaker])
            selector.blockSignals(False)
        self._selection_changed()

    def _swap(self):
        first, second = (s.currentIndex() for s in self.selectors)
        self.selectors[0].setCurrentIndex(second)
        self.selectors[1].setCurrentIndex(first)

    def start_session(self):
        if self.active or not self.models_ready or not self._translation_is_ready or self._closing:
            return
        try:
            routes = self._selected_routes()
            problem = self._language_problem()
            if problem:
                raise ValueError(problem)
        except ValueError as error:
            self.status.setText(str(error))
            return
        self._session_languages = tuple(route.source_language for route in routes)
        self._session_mode = self.mode.currentData()
        self.generation += 1
        generation = self.generation
        self._deferred = [[], []]
        self.active = True
        try:
            self.worker.set_session(self.generation, routes)
            self.audio.start(routes)
        except Exception as error:
            self.stop_session()
            self.status.setText(f"Cannot start: {error}")
            return
        # A synchronous audio failure can stop this session inside start().
        if not self.active or self.generation != generation:
            return
        self._reset_capture()
        self._playback = [False, False]
        self._cooldown = [0.0, 0.0]
        self.status.setText("Listening · automatic turns (take turns to avoid room bleed)" if self.mode.currentData() == "auto" else "Listening · hold a talk button or F1 / F2")
        self._busy = [0, 0]
        for speaker in (0, 1):
            self._idle_state(speaker)
        self._selection_changed()

    def stop_session(self):
        self.active = False
        self.generation += 1
        self._deferred = [[], []]
        self._reset_capture()
        self._busy = [0, 0]
        self.audio.stop()
        self._playback = [False, False]
        self._cooldown = [0.0, 0.0]
        if self.worker is not None:
            self.worker.stop_session(self.generation)
        for card in self.cards:
            card.state.setText("Stopped")
            card.meter.setValue(0)
        self.status.setText("Stopped · microphones and playback off")
        self._selection_changed()

    def _reset_capture(self, speaker=None):
        for n in (0, 1) if speaker is None else (speaker,):
            self.segmenters[n].reset()
            self._held[n] = False
            self._chunks[n] = []
            self._sizes[n] = 0
            self._capture_overflow[n] = False

    def _press(self, speaker):
        if not self.active or self._session_mode != "ptt" or self._held[speaker]:
            return
        self._reset_capture(speaker)
        # Claim capture before stop_playback's synchronous playing(False) signal.
        self._held[speaker] = True
        if self._playback[speaker]:
            self.audio.stop_playback(speaker)
        self._playback[speaker] = False
        self._cooldown[speaker] = 0.0
        self._idle_state(speaker)

    def _release(self, speaker):
        if not self._held[speaker]:
            return
        chunks = self._chunks[speaker]
        overflow = self._capture_overflow[speaker]
        self._reset_capture(speaker)
        if overflow:
            self.cards[speaker].state.setText(f"Recording discarded · {MAX_TURN_SECONDS}-second limit; try a shorter turn")
        elif chunks:
            self.submit_audio(speaker, np.concatenate(chunks))
        elif self.active:
            self._idle_state(speaker)
        self._flush_playback(speaker)

    def _samples(self, speaker, samples):
        if not self.active or speaker not in (0, 1):
            return
        if self._playback[speaker] or monotonic() < self._cooldown[speaker]:
            self.segmenters[speaker].reset()
            return
        try:
            samples = np.asarray(samples)
            if samples.ndim != 1 or samples.dtype.kind not in "buif":
                return
            if not np.isfinite(samples).all():
                return
            with np.errstate(over="raise", invalid="raise"):
                samples = np.asarray(samples, dtype=np.float32)
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            return
        if self._session_mode == "auto":
            was_talking = bool(self.segmenters[speaker].active)
            turns = self.segmenters[speaker].push(samples)
            for turn in turns:
                self.submit_audio(speaker, turn)
            if self.segmenters[speaker].active:
                self.cards[speaker].state.setText("Talking · automatic turn, pause to translate")
            elif was_talking and not turns:
                # A discarded short turn becomes idle. Completed submissions
                # keep their success/rejection feedback through idle callbacks.
                self._idle_state(speaker)
        elif self._held[speaker]:
            if self._capture_overflow[speaker]:
                return
            if self._sizes[speaker] + len(samples) > MAX_TURN_SAMPLES:
                self._capture_overflow[speaker] = True
                self._chunks[speaker] = []
                self._sizes[speaker] = 0
                self.status.setText(f"Recording discarded · {MAX_TURN_SECONDS}-second PTT limit; release and try a shorter turn")
                self._idle_state(speaker)
                return
            if len(samples):
                self._chunks[speaker].append(samples.copy())
                self._sizes[speaker] += len(samples)

    def _reject_turn(self, speaker, message):
        self.status.setText(message)
        if (self._held[speaker] or self._playback[speaker]
                or (self._session_mode == "auto" and self.segmenters[speaker].active)):
            # An invalid explicit turn must not hide concurrent live activity.
            self._idle_state(speaker)
        else:
            self.cards[speaker].state.setText(message)
        return False

    def submit_audio(self, speaker, samples):
        """Submit an in-memory mono 16 kHz turn; usable by explicit local demos."""
        if (not self.active or type(speaker) is not int
                or speaker not in (0, 1) or self.worker is None):
            return False
        try:
            samples = np.asarray(samples)
            if (samples.ndim != 1 or samples.dtype.kind not in "buif"
                    or not np.isfinite(samples).all() or len(samples) < 4000):
                raise ValueError("Invalid mono numeric turn")
            with np.errstate(over="raise", invalid="raise"):
                samples = np.asarray(samples, dtype=np.float32)
        except (TypeError, ValueError, OverflowError, FloatingPointError):
            return self._reject_turn(speaker, "Turn too short or invalid · speak for at least ¼ second")
        if len(samples) > MAX_TURN_SAMPLES:
            return self._reject_turn(speaker, f"Turn discarded · maximum {MAX_TURN_SECONDS} seconds")
        if not self.worker.submit(self.generation, speaker, samples):
            self.status.setText("Worker busy · turn not queued; wait, then try again")
            if (self._held[speaker] or self._playback[speaker]
                    or (self._session_mode == "auto" and self.segmenters[speaker].active)):
                # Rejection concerns this explicit turn, not live capture/playback.
                self._idle_state(speaker)
            else:
                self.cards[speaker].state.setText("Busy · queue full, please wait")
            return False
        self._busy[speaker] += 1
        if (self._held[speaker] or self._playback[speaker]
                or (self._session_mode == "auto" and self.segmenters[speaker].active)):
            # Queued inference does not supersede live capture/playback feedback.
            self._idle_state(speaker)
        else:
            self.cards[speaker].state.setText("Busy · transcribing and translating locally…")
        self.status.setText(f"User {speaker + 1} · interpreting a local turn")
        return True

    def _level(self, speaker, rms):
        if self.active and speaker in (0, 1) and np.isfinite(rms) and not self._playback[speaker]:
            self.cards[speaker].meter.setValue(min(100, max(0, int(rms * 500))))

    def eventFilter(self, watched, event):
        if event.type() in (QEvent.KeyPress, QEvent.KeyRelease) and event.key() in (Qt.Key_F1, Qt.Key_F2):
            # Application filter catches keys even when a read-only editor has focus.
            if isinstance(watched, QWidget) and watched.window() is self and self.active:
                if not event.isAutoRepeat():
                    speaker = 0 if event.key() == Qt.Key_F1 else 1
                    if event.type() == QEvent.KeyPress:
                        self._press(speaker)
                    else:
                        self._release(speaker)
                return True
        if (watched is self and event.type() == QEvent.WindowDeactivate
                and self._session_mode == "ptt"):
            # Only held-key capture needs focus-loss cancellation. Automatic
            # conversation stays active in the background and must retain audio.
            cancelled = any(self._held)
            self._reset_capture()
            if cancelled and self.active and not self._closing:
                self.status.setText("Talk cancelled · window lost focus; unsubmitted microphone audio discarded")
                for speaker in (0, 1):
                    self._idle_state(speaker)
                    self._flush_playback(speaker)
        return super().eventFilter(watched, event)


    def _progress(self, generation, message):
        if self.active and generation == self.generation:
            self.status.setText(message)

    def _settled(self, generation, speaker, produced):
        if not self.active or generation != self.generation or speaker not in (0, 1) or produced:
            return
        self._busy[speaker] = max(0, self._busy[speaker] - 1)
        self._idle_state(speaker)
        if (not self._busy[speaker] and not self._playback[speaker]
                and not self._held[speaker]
                and not (self._session_mode == "auto" and self.segmenters[speaker].active)):
            self.cards[speaker].state.setText("Waiting · turn produced no transcript")

    def _error(self, generation, message):
        if self._closing:
            return
        if generation == -1:
            self.models_ready = False
            if self.active:
                self.stop_session()
            self.status.setText(f"Model loading failed: {message}")
            self._selection_changed()
        elif self.active and generation == self.generation:
            self.status.setText(f"Interpretation error: {message}")

    def _result(self, turn):
        # Never trust worker cancellation alone: queued Qt signals can arrive late.
        if not self.active or turn.generation != self.generation:
            return
        if turn.speaker not in (0, 1) or turn.listener != 1 - turn.speaker:
            return
        if getattr(turn, "capture_id", None) is not None:
            # Retired incremental captures are never accepted as released turns.
            return
        source, target = self._session_languages[turn.speaker], self._session_languages[turn.listener]
        if (turn.source_language, turn.target_language) != (source, target):
            return
        card = self.cards[turn.speaker]
        card.original.appendPlainText(str(turn.original))
        card.translated.appendPlainText(str(turn.translation))
        self._busy[turn.speaker] = max(0, self._busy[turn.speaker] - 1)
        self._idle_state(turn.speaker)
        self.status.setText(f"User {turn.speaker + 1} → User {turn.listener + 1} · translated locally ({turn.elapsed:.1f}s)")
        if not self.text_only.isChecked() and turn.audio is not None and len(turn.audio):
            pending = self._deferred[turn.listener]
            if len(pending) >= 2:
                self.status.setText(f"Deferred playback queue full for User {turn.listener + 1} · translation kept as text; spoken input preserved")
                return
            pending.append((turn.audio, turn.rate))
            if self._held[turn.listener]:
                self.status.setText(f"Translation ready · playback deferred until User {turn.listener + 1} releases talk")
            self._flush_playback(turn.listener)

    def _flush_playback(self, listener):
        while (self.active and not self.text_only.isChecked() and not self._held[listener]
               and not self._playback[listener] and self._deferred[listener]):
            samples, rate = self._deferred[listener].pop(0)
            if self.audio.play(listener, samples, rate):
                return
            # Rejected output produces no completion signal to advance the FIFO.
            # Recheck lifecycle guards before attempting the next translation.
            self.status.setText("Translation ready · headphone playback unavailable")

    def _playing(self, listener, playing):
        if listener not in (0, 1) or not self.active:
            return
        # Late playback signals must never clear a held microphone's voice.
        if self._held[listener]:
            self._playback[listener] = False
            self._cooldown[listener] = 0.0
            return
        if not playing and not self._playback[listener]:
            # stop_playback also notifies idle endpoints (e.g. Text only).
            # Only a real playback completion may reset capture/cooldown.
            return
        if playing and self.text_only.isChecked():
            return
        self._playback[listener] = bool(playing)
        self.segmenters[listener].reset()
        self._chunks[listener] = []
        self._sizes[listener] = 0
        self.cards[listener].meter.setValue(0)
        if playing:
            self.cards[listener].state.setText("Paused · translated audio in your headphones")
        else:
            self._cooldown[listener] = monotonic() + .3
            self._idle_state(listener)
            self._flush_playback(listener)

    def _text_only_changed(self, enabled):
        if enabled:
            self._deferred = [[], []]
            if self.active:
                for listener in (0, 1):
                    self.audio.stop_playback(listener)

    def _audio_failure(self, speaker, message):
        if self.active:
            self.stop_session()
            self.status.setText(f"Audio stopped: {message}")

    def _devices_changed(self):
        if self._closing:
            return
        self._reconcile_devices()
        # Enumeration can trail Qt's device change notification. Recheck active
        # routes too, rather than calling refresh_devices (a no-op when active).
        self._discovery.start(800)

    def _reconcile_devices(self):
        if self._closing:
            return
        if self.active:
            present = {(h.key, h.microphone_id, h.output_id) for h in self.audio.headsets()}
            selected = [s.currentData() for s in self.selectors]
            if any((h.key, h.microphone_id, h.output_id) not in present for h in selected):
                self.stop_session()
                self.status.setText("A selected headset disconnected · reconnect and select again")
        self.refresh_devices()

    def closeEvent(self, event):
        if not self._closing:
            self._closing = True
            self._discovery.stop()
            self.stop_session()
            self.centralWidget().setEnabled(False)
            if self.worker is not None:
                self.worker.shutdown()
        if self.worker is None or self.worker.wait(0):
            self._close_poll.stop()
            QApplication.instance().removeEventFilter(self)
            event.accept()
        else:
            # Keep the object/thread alive, but never block the GUI during inference.
            self.status.setText("Closing · capture stopped; waiting for local model thread to finish…")
            self._close_poll.start()
            event.ignore()

    def _poll_close(self):
        if self.worker is None or self.worker.wait(0):
            self._close_poll.stop()
            self.close()

    def _idle_state(self, speaker):
        if self._held[speaker] and self._capture_overflow[speaker]:
            state = f"Recording discarded · {MAX_TURN_SECONDS}-second limit; release to try again"
        elif self._held[speaker]:
            state = "Talking · release to translate"
        elif self._playback[speaker]:
            state = "Paused · translated audio in your headphones"
        elif self._session_mode == "auto" and self.segmenters[speaker].active:
            state = "Talking · automatic turn, pause to translate"
        elif self._busy[speaker]:
            state = "Busy · interpreting a queued local turn…"
        else:
            state = "Waiting · hold to speak" if self.mode.currentData() == "ptt" else "Waiting · automatic conversation"
        self.cards[speaker].state.setText(state)
