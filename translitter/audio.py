"""Explicitly selected USB pairs; never open a default audio endpoint."""
from .core import Headset
from collections import deque
import math
import numpy as np
from scipy.signal import resample_poly
from PySide6.QtCore import QObject, Signal, QByteArray, QBuffer, QIODevice, QTimer
from PySide6.QtMultimedia import QtAudio as QAudio, QAudioFormat, QAudioSource, QAudioSink, QMediaDevices


def prepare_pcm(samples, source_rate):
    if not isinstance(source_rate, int) or source_rate <= 0:
        raise ValueError("Invalid synthesized sample rate")
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not np.isfinite(samples).all():
        raise ValueError("Audio must be finite, mono PCM")
    divisor = math.gcd(source_rate, 16000)
    if source_rate != 16000:
        samples = resample_poly(samples, 16000 // divisor, source_rate // divisor)
    return (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()


def mono_format():
    fmt = QAudioFormat()
    fmt.setSampleRate(16000)
    fmt.setChannelCount(1)
    fmt.setSampleFormat(QAudioFormat.Int16)
    return fmt


class AudioHub(QObject):
    samples = Signal(int, object)
    level = Signal(int, float)
    failure = Signal(int, str)
    playing = Signal(int, bool)
    devices_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.manager = QMediaDevices(self)
        self.sources = {}
        self.sinks = {}
        self.inputs = {}
        self.output_ids = {}
        self.capture_bytes = [0, 0]
        self.played_bytes = [0, 0]
        self.buffers = {}
        self.queues = {0: deque(), 1: deque()}
        self.tails = {0: b"", 1: b""}
        self.running = False
        self.volume = 0.45
        self.routes = ()
        self.manager.audioInputsChanged.connect(self._devices_changed)
        self.manager.audioOutputsChanged.connect(self._devices_changed)
        # Qt 6.11 exposes stateChanged(QAudio::State) but returns QtAudio.State;
        # its Python signal conversion fails. Query the working enum API instead.
        self.monitor = QTimer(self)
        self.monitor.setInterval(40)
        self.monitor.timeout.connect(self._poll_states)
        self.monitor.start()

    def _poll_states(self):
        if not self.running:
            return
        for speaker, source in list(self.sources.items()):
            self._input_state(speaker, source.state())
        for listener, sink in list(self.sinks.items()):
            self._output_state(listener, sink.state())

    def headsets(self):
        return pair_devices(
            [(bytes(d.id()).decode(), d.description()) for d in self.manager.audioInputs()],
            [(bytes(d.id()).decode(), d.description()) for d in self.manager.audioOutputs()],
        )

    def _devices_changed(self):
        self.devices_changed.emit()
        if self.running:
            inputs = {bytes(d.id()).decode() for d in self.manager.audioInputs()}
            outputs = {bytes(d.id()).decode() for d in self.manager.audioOutputs()}
            if any(r.microphone_id not in inputs or r.output_id not in outputs for r in self.routes):
                self.failure.emit(-1, "A selected headset disconnected; conversation stopped")
                self.stop()

    def start(self, routes):
        from .core import make_routes, Headset
        if len(routes) != 2:
            raise ValueError("Exactly two routes are required")
        # Validate direction and uniqueness at the audio boundary too.
        first = Headset("first", routes[0].microphone_id, routes[1].output_id, "first")
        second = Headset("second", routes[1].microphone_id, routes[0].output_id, "second")
        if tuple(routes) != make_routes(first, second):
            raise ValueError("Invalid cross-headset routing")
        inputs = {bytes(d.id()).decode(): d for d in self.manager.audioInputs()}
        outputs = {bytes(d.id()).decode(): d for d in self.manager.audioOutputs()}
        fmt = mono_format()
        for route in routes:
            if route.microphone_id not in inputs or route.output_id not in outputs:
                raise ValueError("A selected headset is missing; no default-device fallback")
            if not inputs[route.microphone_id].isFormatSupported(fmt) or not outputs[route.output_id].isFormatSupported(fmt):
                raise ValueError("Selected headset does not support mono 16 kHz PCM")
        self.stop()
        self.routes = tuple(routes)
        self.running = True
        self.capture_bytes = [0, 0]
        self.played_bytes = [0, 0]
        try:
            for route in routes:
                sink = QAudioSink(outputs[route.output_id], fmt, self)
                sink.setVolume(self.volume)

                self.sinks[route.listener] = sink
                self.output_ids[route.listener] = route.output_id
                source = QAudioSource(inputs[route.microphone_id], fmt, self)
                source.setBufferSize(6400)

                self.sources[route.speaker] = source
                io = source.start()
                if io is None or source.error() != QAudio.Error.NoError:
                    raise RuntimeError(f"Cannot open User {route.speaker + 1}'s microphone")
                self.inputs[route.speaker] = io
                io.readyRead.connect(lambda n=route.speaker: self._read(n))
        except Exception:
            self.stop()
            raise

    def _read(self, speaker):
        if not self.running or speaker not in self.inputs:
            return
        data = self.tails[speaker] + bytes(self.inputs[speaker].readAll())
        count = len(data) - len(data) % 2
        self.tails[speaker] = data[count:]
        if not count:
            return
        samples = np.frombuffer(data[:count], dtype="<i2").astype(np.float32) / 32768
        self.capture_bytes[speaker] += count
        self.level.emit(speaker, float(np.sqrt(np.mean(samples * samples))))
        self.samples.emit(speaker, samples)

    def _input_state(self, speaker, state):
        source = self.sources.get(speaker)
        if self.running and source and source.error() != QAudio.Error.NoError:
            self.failure.emit(speaker, "Microphone stream failed; conversation stopped")

    def play(self, listener, samples, rate):
        if not self.running or listener not in self.sinks:
            return False
        if len(self.queues[listener]) >= 2:
            self.failure.emit(listener, "Playback queue full; please take shorter turns")
            return False
        self.queues[listener].append(prepare_pcm(samples, rate))
        if listener not in self.buffers:
            self._next(listener)
        return True

    def _next(self, listener):
        if not self.running or listener not in self.sinks or not self.queues[listener]:
            self.playing.emit(listener, False)
            return
        data = self.queues[listener].popleft()
        buffer = QBuffer(self)
        buffer.setData(QByteArray(data))
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self.buffers[listener] = buffer
        self.playing.emit(listener, True)
        self.played_bytes[listener] += len(data)
        self.sinks[listener].start(buffer)

    def _output_state(self, listener, state):
        sink = self.sinks.get(listener)
        if not self.running or not sink:
            return
        if sink.error() not in (QAudio.Error.NoError, QAudio.Error.UnderrunError):
            self.stop_playback(listener)
            self.failure.emit(listener, "Headphone playback failed; verify USB connection")
        elif state == QAudio.State.IdleState and listener in self.buffers:
            buffer = self.buffers.pop(listener)
            sink.reset()
            buffer.close()
            buffer.deleteLater()
            if self.queues[listener]:
                self._next(listener)
            else:
                self.playing.emit(listener, False)

    def set_volume(self, volume):
        self.volume = max(0.0, min(0.8, float(volume)))
        for sink in self.sinks.values():
            sink.setVolume(self.volume)

    def stop_playback(self, listener):
        self.queues[listener].clear()
        buffer = self.buffers.pop(listener, None)
        sink = self.sinks.get(listener)
        if sink:
            sink.reset()
        if buffer:
            buffer.close()
            buffer.deleteLater()
        self.playing.emit(listener, False)

    def stop(self):
        self.running = False
        for source in self.sources.values():
            source.stop()
            source.deleteLater()
        self.sources.clear()
        self.inputs.clear()
        for listener in (0, 1):
            self.stop_playback(listener)
        for sink in self.sinks.values():
            sink.deleteLater()
        self.sinks.clear()
        self.output_ids.clear()
        self.routes = ()
        self.tails = {0: b"", 1: b""}


def pair_devices(inputs, outputs):
    def key(identifier, direction):
        prefix = f"alsa_{direction}."
        if not identifier.startswith(prefix + "usb-"):
            return None
        return identifier[len(prefix):].rsplit(".", 1)[0]

    output_by_key = {}
    for identifier, label in outputs:
        identity = key(identifier, "output")
        if identity:
            output_by_key.setdefault(identity, []).append(identifier)
    result = []
    seen = set()
    for identifier, label in inputs:
        identity = key(identifier, "input")
        matches = output_by_key.get(identity, [])
        if identity and len(matches) == 1 and identity not in seen:
            suffix = identity.rsplit("_", 1)[-1].removesuffix("-00")[-6:]
            result.append(Headset(identity, identifier, matches[0], f"{label} · USB {suffix}"))
            seen.add(identity)
    return sorted(result, key=lambda pair: pair.key)
