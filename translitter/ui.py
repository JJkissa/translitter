"""Local native interpreter. Session generations gate every model response."""
from pathlib import Path
import numpy as np
from time import monotonic
from PySide6.QtCore import QObject, Signal, QTimer, Qt, QEvent
from PySide6.QtWidgets import (QMainWindow, QWidget, QFrame, QLabel, QPushButton,
    QComboBox, QCheckBox, QSlider, QPlainTextEdit, QProgressBar,
    QVBoxLayout, QHBoxLayout, QApplication)
from .core import make_routes, Segmenter
from .theme import apply_theme


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
        box.addWidget(label(f"User {speaker + 1}  /  {'Finnish' if speaker == 0 else 'English'}", "participant"))
        box.addWidget(label("SUOMI → ENGLISH" if speaker == 0 else "ENGLISH → SUOMI", "section"))
        columns = QHBoxLayout()
        self.original, self.translated = QPlainTextEdit(), QPlainTextEdit()
        for title, editor, placeholder in (
            ("ORIGINAL", self.original, "Your words appear here…"),
            ("TRANSLATED", self.translated, "Translation for the other headset…")):
            column = QVBoxLayout()
            column.addWidget(label(title, "section"))
            editor.setReadOnly(True)
            editor.setMaximumBlockCount(100)
            editor.setPlaceholderText(placeholder)
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
        self.hold.setToolTip("Hold while speaking; release to translate. Maximum turn: 8 seconds.")
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
        self._closing = False
        self.segmenters = [Segmenter(), Segmenter()]
        self._held = [False, False]
        self._chunks = [[], []]
        self._sizes = [0, 0]
        self._busy = [0, 0]
        self._playback = [False, False]
        self._cooldown = [0.0, 0.0]
        self._deferred = [[], []]
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
        self._discovery.timeout.connect(self.refresh_devices)
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
        header = QHBoxLayout()
        titles = QVBoxLayout()
        titles.addWidget(label("Translitter", "brand"))
        titles.addWidget(label("Two people. Two headsets. One conversation.", "muted"))
        header.addLayout(titles)
        header.addStretch()
        header.addWidget(label("LOCAL ONLY  ·  FI ↔ EN", "badge"))
        page.addLayout(header)
        controls = QFrame()
        controls.setObjectName("controls")
        config = QVBoxLayout(controls)
        config.setContentsMargins(16, 14, 16, 14)
        select_row = QHBoxLayout()
        self.selectors = [QComboBox(), QComboBox()]
        for speaker, selector in enumerate(self.selectors):
            col = QVBoxLayout()
            col.addWidget(label(f"USER {speaker + 1} · {'FINNISH' if speaker == 0 else 'ENGLISH'} HEADSET", "section"))
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
        self.mode.addItem("Push to talk · recommended", "ptt")
        self.mode.addItem("Automatic conversation · opt-in", "auto")
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
        page.addWidget(label("Hold F1 / F2 to speak · release to translate · audio and conversation stay in memory", "muted"))
        self._selection_changed()

    def _loading(self, message):
        if not self._closing:
            self.status.setText(message)

    def _ready(self):
        if self._closing:
            return
        self.models_ready = True
        self.status.setText("Ready · select two distinct USB headsets, then Start")
        self._selection_changed()

    def _selected_routes(self):
        first, second = (selector.currentData() for selector in self.selectors)
        if first is None or second is None:
            raise ValueError("Connect and select two USB headsets")
        return make_routes(first, second)

    def _selection_changed(self, *_):
        for selector in self.selectors:
            headset = selector.currentData()
            selector.setToolTip(f"Microphone: {headset.microphone_id}\nHeadphones: {headset.output_id}" if headset else "No complete USB headset pair detected")
        try:
            self._selected_routes()
            valid = True
        except ValueError:
            valid = False
        self.start_button.setEnabled(self.models_ready and valid and not self.active and not self._closing)
        self.stop_button.setEnabled(self.active)
        for widget in (*self.selectors, self.swap_button, self.refresh_button, self.mode):
            widget.setEnabled(not self.active and not self._closing)
        for card in self.cards:
            card.hold.setEnabled(self.active and self.mode.currentData() == "ptt")

    def refresh_devices(self):
        if self.active or self._closing:
            return
        keys = [s.currentData().key if s.currentData() else None for s in self.selectors]
        pairs = self.audio.headsets()
        for speaker, selector in enumerate(self.selectors):
            selector.blockSignals(True)
            selector.clear()
            for pair in pairs:
                selector.addItem(pair.label, pair)
            index = next((i for i, p in enumerate(pairs) if p.key == keys[speaker]), min(speaker, len(pairs) - 1))
            selector.setCurrentIndex(index)
            selector.blockSignals(False)
        self._selection_changed()

    def _swap(self):
        first, second = (s.currentIndex() for s in self.selectors)
        self.selectors[0].setCurrentIndex(second)
        self.selectors[1].setCurrentIndex(first)

    def start_session(self):
        if self.active or not self.models_ready or self._closing:
            return
        try:
            routes = self._selected_routes()
        except ValueError as error:
            self.status.setText(str(error))
            return
        self.generation += 1
        self._deferred = [[], []]
        self.active = True
        try:
            self.worker.set_session(self.generation, routes)
            self.audio.start(routes)
        except Exception as error:
            self.stop_session()
            self.status.setText(f"Cannot start: {error}")
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

    def _press(self, speaker):
        if not self.active or self.mode.currentData() != "ptt" or self._held[speaker]:
            return
        self._reset_capture(speaker)
        # Claim capture before stop_playback's synchronous playing(False) signal.
        self._held[speaker] = True
        if self._playback[speaker]:
            self.audio.stop_playback(speaker)
        self._playback[speaker] = False
        self._cooldown[speaker] = 0.0
        self.cards[speaker].state.setText("Talking · release to translate")

    def _release(self, speaker):
        if not self._held[speaker]:
            return
        chunks = self._chunks[speaker]
        self._reset_capture(speaker)
        if chunks:
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
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            return
        if self.mode.currentData() == "auto":
            for turn in self.segmenters[speaker].push(samples):
                self.submit_audio(speaker, turn)
            if self.segmenters[speaker].active:
                self.cards[speaker].state.setText("Talking · automatic turn, pause to translate")
            else:
                self._idle_state(speaker)
        elif self._held[speaker]:
            while len(samples):
                take = min(len(samples), 128000 - self._sizes[speaker])
                self._chunks[speaker].append(samples[:take].copy())
                self._sizes[speaker] += take
                samples = samples[take:]
                if self._sizes[speaker] == 128000:
                    self.submit_audio(speaker, np.concatenate(self._chunks[speaker]))
                    self._chunks[speaker] = []
                    self._sizes[speaker] = 0

    def submit_audio(self, speaker, samples):
        """Submit an in-memory mono 16 kHz turn; usable by explicit local demos."""
        if not self.active or speaker not in (0, 1) or self.worker is None:
            return False
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all() or len(samples) < 4000:
            self.cards[speaker].state.setText("Turn too short or invalid · speak for at least ¼ second")
            return False
        if not self.worker.submit(self.generation, speaker, samples[:128000]):
            self.status.setText("Worker busy · turn not queued; wait, then try again")
            self.cards[speaker].state.setText("Busy · queue full, please wait")
            return False
        self._busy[speaker] += 1
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
        if watched is self and event.type() == QEvent.WindowDeactivate:
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
        if not self._busy[speaker] and not self._playback[speaker]:
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
        source, target = ("fi", "en") if turn.speaker == 0 else ("en", "fi")
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
        if (not self.active or self.text_only.isChecked() or self._held[listener]
                or self._playback[listener] or not self._deferred[listener]):
            return
        samples, rate = self._deferred[listener].pop(0)
        if not self.audio.play(listener, samples, rate):
            self.status.setText("Translation ready · headphone playback unavailable")

    def _playing(self, listener, playing):
        if listener not in (0, 1) or not self.active:
            return
        # Late playback signals must never clear a held microphone's voice.
        if self._held[listener]:
            self._playback[listener] = False
            self._cooldown[listener] = 0.0
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
        if self.active:
            present = {(h.key, h.microphone_id, h.output_id) for h in self.audio.headsets()}
            selected = [s.currentData() for s in self.selectors]
            if any((h.key, h.microphone_id, h.output_id) not in present for h in selected):
                self.stop_session()
                self.status.setText("A selected headset disconnected · reconnect and select again")
        self.refresh_devices()
        # Enumeration can trail Qt's device change notification.
        self._discovery.start(800)

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
        if self._held[speaker]:
            state = "Talking · release to translate"
        elif self._playback[speaker]:
            state = "Paused · translated audio in your headphones"
        elif self._busy[speaker]:
            state = "Busy · interpreting a queued local turn…"
        else:
            state = "Waiting · hold to speak" if self.mode.currentData() == "ptt" else "Waiting · automatic conversation"
        self.cards[speaker].state.setText(state)
