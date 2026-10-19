"""Explicitly selected USB pairs; never open a default audio endpoint."""
from .core import Headset
from collections import deque
import math
import numpy as np
from scipy.signal import resample_poly
from PySide6.QtCore import QObject, Signal, QByteArray, QBuffer, QIODevice, QTimer
from PySide6.QtMultimedia import QtAudio as QAudio, QAudioFormat, QAudioSource, QAudioSink, QMediaDevices


def prepare_pcm(samples, source_rate):
    if isinstance(source_rate, bool) or not isinstance(source_rate, int) or source_rate <= 0:
        raise ValueError("Invalid synthesized sample rate")
    samples = np.asarray(samples)
    if np.iscomplexobj(samples):
        raise ValueError("Audio must be real, mono PCM")
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
        self._lifecycle = 0
        self._output_lifecycle = [0, 0]
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
            # A direct failure slot may stop or replace the whole session.
            if self.running and self.sources.get(speaker) is source:
                self._input_state(speaker, source.state())
        for listener, sink in list(self.sinks.items()):
            if self.running and self.sinks.get(listener) is sink:
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
                lifecycle = self._lifecycle
                self.failure.emit(-1, "A selected headset disconnected; conversation stopped")
                # A direct failure slot may already stop or replace this session.
                if self._lifecycle == lifecycle:
                    self.stop()

    def start(self, routes):
        from .core import make_routes, Headset
        if len(routes) != 2:
            raise ValueError("Exactly two routes are required")
        # Validate direction and uniqueness at the audio boundary too.
        first = Headset("first", routes[0].microphone_id, routes[1].output_id, "first")
        second = Headset("second", routes[1].microphone_id, routes[0].output_id, "second")
        if tuple(routes) != make_routes(first, second,
                                        routes[0].source_language,
                                        routes[1].source_language):
            raise ValueError("Invalid cross-headset routing")
        inputs = {bytes(d.id()).decode(): d for d in self.manager.audioInputs()}
        outputs = {bytes(d.id()).decode(): d for d in self.manager.audioOutputs()}
        fmt = mono_format()
        for route in routes:
            if route.microphone_id not in inputs or route.output_id not in outputs:
                raise ValueError("A selected headset is missing; no default-device fallback")
            if not inputs[route.microphone_id].isFormatSupported(fmt) or not outputs[route.output_id].isFormatSupported(fmt):
                raise ValueError("Selected headset does not support mono 16 kHz PCM")
        stop_lifecycle = self._lifecycle + 1
        self.stop()
        # Settlement slots may supersede this start with a newer Stop or Start.
        if self._lifecycle != stop_lifecycle:
            return
        self.routes = tuple(routes)
        self.running = True
        self.capture_bytes = [0, 0]
        self.played_bytes = [0, 0]
        try:
            for route in routes:
                sink = QAudioSink(outputs[route.output_id], fmt, self)
                # A constructor can return after a synchronous Stop or restart.
                # This endpoint was never registered, so retire it locally only.
                if self._lifecycle != stop_lifecycle:
                    sink.reset()
                    sink.deleteLater()
                    return
                # Track ownership before configuration so rollback sees every endpoint.
                self.sinks[route.listener] = sink
                sink.setVolume(self.volume)
                # Configuration may synchronously retire this startup too.
                if self._lifecycle != stop_lifecycle:
                    return
                self.output_ids[route.listener] = route.output_id
                source = QAudioSource(inputs[route.microphone_id], fmt, self)
                if self._lifecycle != stop_lifecycle:
                    source.stop()
                    source.deleteLater()
                    return
                self.sources[route.speaker] = source
                source.setBufferSize(6400)
                if self._lifecycle != stop_lifecycle:
                    return
                io = source.start()
                # Activation can synchronously stop or replace this startup.
                if self._lifecycle != stop_lifecycle:
                    return
                if io is None or source.error() != QAudio.Error.NoError:
                    raise RuntimeError(f"Cannot open User {route.speaker + 1}'s microphone")
                self.inputs[route.speaker] = io
                io.readyRead.connect(lambda n=route.speaker, origin=io: self._read(n, origin))
        except Exception:
            # A retired activation failure must not tear down its replacement.
            if self._lifecycle != stop_lifecycle:
                return
            self.stop()
            raise

    def _read(self, speaker, origin=None):
        if not self.running or speaker not in self.inputs:
            return
        io = self.inputs[speaker]
        # Deferred readyRead signals from retired devices must not drain new ones.
        if origin is not None and origin is not io:
            return
        data = self.tails[speaker] + bytes(io.readAll())
        count = len(data) - len(data) % 2
        self.tails[speaker] = data[count:]
        if not count:
            return
        samples = np.frombuffer(data[:count], dtype="<i2").astype(np.float32) / 32768
        self.capture_bytes[speaker] += count
        self.level.emit(speaker, float(np.sqrt(np.mean(samples * samples))))
        # Direct level slots may stop or synchronously replace this session.
        if self.running and self.inputs.get(speaker) is io:
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
        try:
            data = prepare_pcm(samples, rate)
        except (TypeError, ValueError, OverflowError):
            # Reject malformed output without interrupting healthy queued audio.
            return False
        if not data:
            return False
        self.queues[listener].append(data)
        if listener not in self.buffers:
            self._next(listener)
        return True

    def _next(self, listener):
        if not self.running or listener not in self.sinks or not self.queues[listener]:
            self.playing.emit(listener, False)
            return
        self._output_lifecycle[listener] += 1
        data = self.queues[listener].popleft()
        buffer = QBuffer(self)
        buffer.setData(QByteArray(data))
        buffer.open(QIODevice.OpenModeFlag.ReadOnly)
        self.buffers[listener] = buffer
        sink = self.sinks[listener]
        self.playing.emit(listener, True)
        # Direct slots can stop, replace the session, or cancel just this output.
        if (not self.running or self.sinks.get(listener) is not sink
                or self.buffers.get(listener) is not buffer):
            return
        self.played_bytes[listener] += len(data)
        sink.start(buffer)

    def _output_state(self, listener, state):
        sink = self.sinks.get(listener)
        if not self.running or not sink:
            return
        if sink.error() not in (QAudio.Error.NoError, QAudio.Error.UnderrunError):
            lifecycle = self._lifecycle
            # stop_playback advances once; anything else supersedes this failure.
            output_lifecycle = self._output_lifecycle[listener] + 1
            self.stop_playback(listener)
            if (self.running and self._lifecycle == lifecycle
                    and self.sinks.get(listener) is sink
                    and self._output_lifecycle[listener] == output_lifecycle):
                self.failure.emit(listener, "Headphone playback failed; verify USB connection")
        elif state == QAudio.State.IdleState and listener in self.buffers:
            lifecycle = self._lifecycle
            output_lifecycle = self._output_lifecycle[listener]
            buffer = self.buffers.pop(listener)
            sink.reset()
            buffer.close()
            buffer.deleteLater()
            # Reset can synchronously cancel output or replace the session.
            # Always retire our detached buffer, but never settle newer output.
            if (not self.running or self._lifecycle != lifecycle
                    or self.sinks.get(listener) is not sink
                    or self._output_lifecycle[listener] != output_lifecycle):
                return
            if self.queues[listener]:
                self._next(listener)
            else:
                self.playing.emit(listener, False)

    def set_volume(self, volume):
        self.volume = max(0.0, min(0.8, float(volume)))
        lifecycle = self._lifecycle
        for listener, sink in list(self.sinks.items()):
            # A direct callback may stop or replace the session during propagation.
            if self._lifecycle != lifecycle:
                return
            if self.sinks.get(listener) is sink:
                sink.setVolume(self.volume)

    def stop_playback(self, listener):
        self._output_lifecycle[listener] += 1
        lifecycle = self._lifecycle
        output_lifecycle = self._output_lifecycle[listener]
        self.queues[listener].clear()
        buffer = self.buffers.pop(listener, None)
        sink = self.sinks.get(listener)
        if sink:
            sink.reset()
        if buffer:
            buffer.close()
            buffer.deleteLater()
        # Reset may synchronously supersede this cancellation on the same sink.
        if (self._lifecycle != lifecycle or self.sinks.get(listener) is not sink
                or self._output_lifecycle[listener] != output_lifecycle):
            return
        self.playing.emit(listener, False)

    def stop(self):
        self._lifecycle += 1
        self.running = False
        # Retire ownership before publishing any synchronous settlement signals.
        sources, sinks, buffers = self.sources, self.sinks, self.buffers
        self.sources = {}
        self.inputs = {}
        self.sinks = {}
        stopped_sinks = self.sinks
        self.buffers = {}
        self.queues = {0: deque(), 1: deque()}
        self.output_ids = {}
        self.routes = ()
        self.tails = {0: b"", 1: b""}
        for source in sources.values():
            source.stop()
            source.deleteLater()
        for sink in sinks.values():
            sink.reset()
        for buffer in buffers.values():
            buffer.close()
            buffer.deleteLater()
        for sink in sinks.values():
            sink.deleteLater()
        for listener in (0, 1):
            if self.running or self.sinks is not stopped_sinks:
                break
            self.playing.emit(listener, False)


def pair_devices(inputs, outputs):
    def key(identifier, direction):
        prefix = f"alsa_{direction}."
        if not identifier.startswith(prefix + "usb-"):
            return None
        identity, separator, profile = identifier[len(prefix):].rpartition(".")
        if not separator or not profile or identity == "usb-":
            return None
        return identity

    output_by_key = {}
    for identifier, label in outputs:
        identity = key(identifier, "output")
        if identity:
            output_by_key.setdefault(identity, []).append(identifier)
    input_counts = {}
    for identifier, _ in inputs:
        identity = key(identifier, "input")
        if identity:
            input_counts[identity] = input_counts.get(identity, 0) + 1
    result = []
    seen = set()
    for identifier, label in inputs:
        identity = key(identifier, "input")
        matches = output_by_key.get(identity, [])
        if (identity and input_counts[identity] == 1
                and len(matches) == 1 and identity not in seen):
            suffix = identity.rsplit("_", 1)[-1].removesuffix("-00")[-6:]
            result.append(Headset(identity, identifier, matches[0], f"{label} · USB {suffix}"))
            seen.add(identity)
    return sorted(result, key=lambda pair: pair.key)
