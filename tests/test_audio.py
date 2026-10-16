import importlib.util
from itertools import permutations

import pytest


@pytest.fixture
def explicit_audio(qtbot, monkeypatch):
    """Real AudioHub startup with fake Qt endpoints; never opens hardware."""
    from types import SimpleNamespace
    from PySide6.QtCore import QObject, Signal, QBuffer, QIODevice
    from PySide6.QtMultimedia import QtAudio
    from translitter import audio
    from translitter.core import Headset

    pairs = (Headset("a", "mic-a", "out-a", "USB A"),
             Headset("b", "mic-b", "out-b", "USB B"))
    opened = []

    def device(identifier):
        return SimpleNamespace(id=lambda: identifier.encode(), description=lambda: identifier,
                               isFormatSupported=lambda fmt: True)

    class Manager(QObject):
        audioInputsChanged = Signal()
        audioOutputsChanged = Signal()
        def audioInputs(self): return [device(pair.microphone_id) for pair in pairs]
        def audioOutputs(self): return [device(pair.output_id) for pair in pairs]

    class Source:
        def __init__(self, endpoint, fmt, parent):
            self.identifier = bytes(endpoint.id()).decode()
            self.buffer = QBuffer(parent)
            self.buffer.open(QIODevice.ReadOnly)
        def setBufferSize(self, size): pass
        def start(self):
            opened.append(("microphone", self.identifier))
            return self.buffer
        def error(self): return QtAudio.Error.NoError
        def state(self): return QtAudio.State.ActiveState
        def stop(self): self.buffer.close()
        def deleteLater(self): self.buffer.deleteLater()

    class Sink:
        def __init__(self, endpoint, fmt, parent):
            self.identifier = bytes(endpoint.id()).decode()
            self._state = QtAudio.State.StoppedState
            opened.append(("headphones", self.identifier))
        def setVolume(self, volume): pass
        def error(self): return QtAudio.Error.NoError
        def state(self): return self._state
        def start(self, buffer):
            assert len(buffer.data()) > 0
            opened.append(("playback", self.identifier))
            self._state = QtAudio.State.IdleState
        def reset(self): self._state = QtAudio.State.StoppedState
        def deleteLater(self): pass

    monkeypatch.setattr(audio, "QMediaDevices", Manager)
    monkeypatch.setattr(audio, "QAudioSource", Source)
    monkeypatch.setattr(audio, "QAudioSink", Sink)
    hub = audio.AudioHub()
    yield hub, pairs, opened
    hub.stop()
    hub.monitor.stop()
    hub.deleteLater()


@pytest.mark.parametrize("first,second", tuple(permutations(("fi", "en", "ru", "uk"), 2)))
def test_audio_boundary_accepts_selected_language_pair(explicit_audio, first, second):
    from translitter.core import make_routes
    hub, pairs, opened = explicit_audio
    routes = make_routes(*pairs, first, second)
    hub.start(routes)
    assert hub.running and hub.routes == routes
    assert set(hub.sources) == {0, 1}
    assert hub.output_ids == {1: "out-b", 0: "out-a"}
    assert opened == [("headphones", "out-b"), ("microphone", "mic-a"),
                      ("headphones", "out-a"), ("microphone", "mic-b")]
    hub.stop()
    assert not hub.running and not hub.sources and not hub.sinks


@pytest.mark.parametrize("corruption", ["speaker", "listener", "target", "unsupported", "same_language", "microphone", "output", "missing", "order"])
def test_multilingual_audio_boundary_still_rejects_invalid_routes_before_open(explicit_audio, corruption):
    from dataclasses import replace
    from translitter.core import make_routes
    hub, pairs, opened = explicit_audio
    routes = list(make_routes(*pairs, "ru", "en"))
    if corruption == "speaker": routes[0] = replace(routes[0], speaker=1)
    elif corruption == "listener": routes[0] = replace(routes[0], listener=0)
    elif corruption == "target": routes[0] = replace(routes[0], target_language="fi")
    elif corruption == "unsupported": routes[0] = replace(routes[0], source_language="xx")
    elif corruption == "same_language": routes[0] = replace(routes[0], source_language="en")
    elif corruption == "microphone": routes[1] = replace(routes[1], microphone_id=routes[0].microphone_id)
    elif corruption == "output": routes[1] = replace(routes[1], output_id=routes[0].output_id)
    elif corruption == "missing": routes[0] = replace(routes[0], microphone_id="not-connected")
    elif corruption == "order": routes.reverse()
    with pytest.raises(ValueError):
        hub.start(routes)
    assert not hub.running and not opened and not hub.sources and not hub.sinks


@pytest.mark.parametrize("first,second", tuple(permutations(("fi", "en", "ru", "uk"), 2)))
def test_window_start_reaches_real_worker_and_audio_validation(explicit_audio, qtbot, tmp_path, first, second):
    """Exercise the production boundaries, not the former FakeAudio shortcut."""
    from translitter.languages import missing_assets
    from translitter.ui import Window
    from translitter.worker import ModelWorker
    hub, pairs, opened = explicit_audio
    for relative in missing_assets(tmp_path, first, second, additional_only=True):
        path = tmp_path / "models" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    worker = ModelWorker(tmp_path)
    window = Window(tmp_path, worker=worker, audio=hub, start_worker=False)
    qtbot.addWidget(window)
    for selector, pair in zip(window.selectors, pairs):
        selector.addItem(pair.label, pair)
        selector.setCurrentIndex(0)
    for selector, language in zip(window.language_selectors, (first, second)):
        selector.setCurrentIndex(selector.findData(language))
    worker.ready.emit()
    assert window.start_button.isEnabled()
    window.start_session()
    try:
        assert window.active, window.status.text()
        assert hub.running and opened
        assert worker._routes == hub.routes
        assert [(r.source_language, r.target_language) for r in hub.routes] == [(first, second), (second, first)]
        assert "Invalid cross-headset" not in window.status.text()
    finally:
        window.stop_session()
        window.close()
    assert not hub.running and not hub.sources and not hub.sinks


def test_identical_headset_names_are_paired_by_device_id_not_by_position():
    assert importlib.util.find_spec("translitter.audio") is not None, "USB audio adapter is missing"
    from translitter.audio import pair_devices
    inputs = [("alsa_input.usb-headset_A.analog-stereo", "USB Headset"),
              ("alsa_input.usb-headset_B.analog-stereo", "USB Headset"),
              ("alsa_output.usb-headset_A.monitor", "Monitor")]
    outputs = [("alsa_output.usb-headset_B.analog-stereo", "USB Headset"),
               ("alsa_output.usb-headset_A.analog-stereo", "USB Headset")]
    pairs = pair_devices(inputs, outputs)
    assert len(pairs) == 2
    assert {p.microphone_id for p in pairs} == {row[0] for row in inputs[:2]}
    for pair in pairs:
        assert pair.microphone_id.partition(".")[2] == pair.output_id.partition(".")[2]
    assert pair_devices(inputs[:1], outputs[:1]) == []


@pytest.mark.parametrize("reverse", [False, True])
def test_ambiguous_microphone_profiles_are_not_paired_by_enumeration_order(reverse):
    from translitter.audio import pair_devices
    inputs = [("alsa_input.usb-headset_A.analog-stereo", "USB A"),
              ("alsa_input.usb-headset_A.mono-fallback", "USB A"),
              ("alsa_input.usb-headset_B.analog-stereo", "USB B")]
    outputs = [("alsa_output.usb-headset_A.analog-stereo", "USB A"),
               ("alsa_output.usb-headset_B.analog-stereo", "USB B")]
    if reverse:
        inputs.reverse()
    pairs = pair_devices(inputs, outputs)
    assert [pair.key for pair in pairs] == ["usb-headset_B"]
    assert pairs[0].microphone_id == "alsa_input.usb-headset_B.analog-stereo"


def test_pcm_preparation_resamples_and_clips_audio():
    import numpy as np
    from translitter import audio
    assert hasattr(audio, "prepare_pcm"), "Output PCM conversion is missing"
    data = audio.prepare_pcm(np.full(22050, 2.0, dtype=np.float32), 22050)
    pcm = np.frombuffer(data, dtype="<i2")
    assert len(pcm) == 16000
    assert np.max(pcm) == 32767 and np.min(pcm) >= 0


def test_missing_routes_cannot_fall_back_to_default_microphones(qtbot):
    from translitter import audio
    from translitter.core import Headset, make_routes
    assert hasattr(audio, "AudioHub"), "Explicit Qt audio streams are missing"
    hub = audio.AudioHub()
    routes = make_routes(Headset("A", "missing-a", "missing-out-a", "A"),
                         Headset("B", "missing-b", "missing-out-b", "B"))
    import pytest
    with pytest.raises(ValueError):
        hub.start(routes)
    assert hub.sources == {} and not hub.running
    hub.stop()


def test_qt_audio_no_error_is_not_misclassified_as_failure(qtbot):
    from PySide6.QtMultimedia import QAudioSource, QtAudio
    from translitter.audio import AudioHub
    hub = AudioHub()
    source = QAudioSource(parent=hub)
    hub.sources[0] = source
    hub.running = True
    failures = []
    hub.failure.connect(lambda n, message: failures.append(message))
    assert source.error() == QtAudio.Error.NoError
    hub._input_state(0, QtAudio.State.IdleState)
    assert failures == []
    hub.stop()


def test_stream_monitor_uses_callable_state_queries(qtbot):
    from translitter.audio import AudioHub
    hub = AudioHub()
    assert hasattr(hub, "monitor"), "Qt state monitor is missing"
    assert hub.monitor.isActive()
    hub._poll_states()
    assert not hub.buffers
    hub.stop()


def test_panic_playback_reset_discards_buffer_instead_of_draining(qtbot):
    from types import SimpleNamespace
    from translitter.audio import AudioHub
    hub = AudioHub()
    calls = []
    hub.sinks[0] = SimpleNamespace(reset=lambda: calls.append("reset"), stop=lambda: calls.append("stop"))
    hub.stop_playback(0)
    assert calls == ["reset"]
    hub.sinks.clear()
    hub.stop()


@pytest.mark.parametrize("rate", [16000, 22050, 24000, 48000])
@pytest.mark.parametrize("already_playing", [False, True])
def test_empty_pcm_does_not_start_or_queue_playback(explicit_audio, rate, already_playing):
    import numpy as np
    from translitter.core import make_routes
    hub, pairs, opened = explicit_audio
    hub.start(make_routes(*pairs))
    events = []
    hub.playing.connect(lambda listener, active: events.append((listener, active)))
    if already_playing:
        assert hub.play(0, np.ones(160, dtype=np.float32), 16000)
    before = (list(opened), dict(hub.buffers), list(events), list(hub.played_bytes))

    assert hub.play(0, np.array([], dtype=np.float32), rate) is False

    assert not hub.queues[0]
    assert (opened, hub.buffers, events, hub.played_bytes) == before


def test_playback_queue_advances_and_stop_discards_pending_audio(explicit_audio):
    import numpy as np
    from PySide6.QtMultimedia import QtAudio
    from translitter.core import make_routes
    hub, pairs, opened = explicit_audio
    hub.start(make_routes(*pairs))
    events = []
    failures = []
    hub.playing.connect(lambda listener, active: events.append((listener, active)))
    hub.failure.connect(lambda listener, message: failures.append((listener, message)))
    for size in (160, 320, 480):
        assert hub.play(0, np.ones(size, dtype=np.float32), 16000)
    assert not hub.play(0, np.ones(640, dtype=np.float32), 16000)
    assert len(hub.queues[0]) == 2
    assert len(failures) == 1 and "queue full" in failures[0][1]
    first = hub.buffers[0]
    hub._output_state(0, QtAudio.State.IdleState)
    assert not first.isOpen()
    assert len(hub.queues[0]) == 1
    assert hub.played_bytes[0] == (160 + 320) * 2
    current = hub.buffers[0]
    hub.stop_playback(0)
    assert not current.isOpen()
    assert not hub.queues[0] and not hub.buffers
    assert events[-1] == (0, False)
    hub._poll_states()
    assert sum(kind == "playback" for kind, _ in opened) == 2
    assert hub.play(0, np.ones(80, dtype=np.float32), 16000)
    hub._poll_states()
    assert not hub.buffers and events[-1] == (0, False)


def _set_capture_chunk(io, data):
    from PySide6.QtCore import QIODevice
    io.close()
    io.setData(data)
    assert io.open(QIODevice.ReadOnly)


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("chunk_sizes", [(12,), (1,) * 12, (3, 0, 1, 5, 3), (0, 5, 0, 7, 0)])
def test_capture_reassembles_pcm_across_byte_boundaries(explicit_audio, speaker, chunk_sizes):
    import numpy as np
    from translitter.core import make_routes
    hub, pairs, _ = explicit_audio
    hub.start(make_routes(*pairs))
    pcm = np.array([-32768, -12345, -1, 0, 12345, 32767], dtype="<i2")
    raw = pcm.tobytes()
    received, levels = [], []
    hub.samples.connect(lambda who, values: received.append((who, values.copy())))
    hub.level.connect(lambda who, value: levels.append((who, value)))
    offset = 0
    for size in chunk_sizes:
        _set_capture_chunk(hub.inputs[speaker], raw[offset:offset + size])
        before = len(received)
        expected_bytes = (offset + size) // 2 * 2
        hub._read(speaker)
        offset += size
        assert hub.capture_bytes[speaker] == expected_bytes
        assert hub.capture_bytes[1 - speaker] == 0
        assert hub.tails[speaker] == raw[expected_bytes:offset]
        assert hub.tails[1 - speaker] == b""
        assert len(received) - before in (0, 1)
    assert offset == len(raw)
    assert all(who == speaker for who, _ in received + levels)
    np.testing.assert_array_equal(np.concatenate([values for _, values in received]),
                                  pcm.astype(np.float32) / 32768)
    assert len(levels) == len(received)
    for (_, values), (_, rms) in zip(received, levels):
        assert values.dtype == np.float32
        assert rms == pytest.approx(float(np.sqrt(np.mean(values * values))))
    before = len(received), len(levels), list(hub.capture_bytes)
    hub._read(speaker)
    assert (len(received), len(levels), hub.capture_bytes) == before


@pytest.mark.parametrize("speaker", [0, 1])
def test_capture_stop_restart_discards_incomplete_sample(explicit_audio, speaker):
    import numpy as np
    from translitter.core import make_routes
    hub, pairs, _ = explicit_audio
    routes = make_routes(*pairs)
    hub.start(routes)
    received = []
    hub.samples.connect(lambda who, values: received.append((who, values.copy())))
    _set_capture_chunk(hub.inputs[speaker], b"\x01\x00\xff")
    hub._read(speaker)
    assert hub.capture_bytes[speaker] == 2
    assert hub.tails[speaker] == b"\xff"
    hub.stop()
    assert hub.tails == {0: b"", 1: b""}
    hub._read(speaker)
    assert len(received) == 1
    hub.start(routes)
    assert hub.capture_bytes == [0, 0]
    _set_capture_chunk(hub.inputs[speaker], b"\x02\x00")
    hub._read(speaker)
    assert len(received) == 2
    assert hub.capture_bytes[speaker] == 2
    assert hub.tails == {0: b"", 1: b""}
    np.testing.assert_array_equal(received[-1][1], np.array([2 / 32768], dtype=np.float32))


@pytest.mark.parametrize("speaker", [0, 1])
def test_capture_ignores_stopped_or_missing_inputs_without_consuming_bytes(explicit_audio, speaker):
    from translitter.core import make_routes
    hub, pairs, _ = explicit_audio
    hub.start(make_routes(*pairs))
    received, levels = [], []
    hub.samples.connect(lambda *args: received.append(args))
    hub.level.connect(lambda *args: levels.append(args))
    io = hub.inputs[speaker]
    _set_capture_chunk(io, b"\x01\x00")
    hub.running = False
    hub._read(speaker)
    assert io.pos() == 0
    hub.running = True
    del hub.inputs[speaker]
    hub._read(speaker)
    assert io.pos() == 0
    hub.inputs[speaker] = io
    assert not received and not levels
    assert hub.capture_bytes == [0, 0]
    assert hub.tails == {0: b"", 1: b""}


@pytest.mark.parametrize("route_index", [0, 1])
@pytest.mark.parametrize("stage", ["sink_constructor", "sink_volume", "source_constructor",
                                  "source_buffer", "source_start", "source_none", "source_error"])
def test_startup_failure_releases_all_endpoints_and_allows_retry(explicit_audio, monkeypatch,
                                                               route_index, stage):
    from PySide6.QtMultimedia import QtAudio
    from translitter import audio
    from translitter.core import make_routes

    hub, pairs, _ = explicit_audio
    routes = make_routes(*pairs)
    source_type, sink_type = audio.QAudioSource, audio.QAudioSink
    created = []
    failing = True

    def fails(endpoint, expected):
        identifier = routes[route_index].microphone_id if expected.startswith("source") else routes[route_index].output_id
        return failing and endpoint.identifier == identifier and stage == expected

    class Source(source_type):
        def __init__(self, endpoint, fmt, parent):
            self.identifier = bytes(endpoint.id()).decode()
            if fails(self, "source_constructor"):
                raise RuntimeError("injected startup failure")
            super().__init__(endpoint, fmt, parent)
            self.deleted = False
            self.stopped = False
            created.append(self)

        def setBufferSize(self, size):
            if fails(self, "source_buffer"):
                raise RuntimeError("injected startup failure")
            super().setBufferSize(size)

        def start(self):
            if fails(self, "source_start"):
                raise RuntimeError("injected startup failure")
            if fails(self, "source_none"):
                return None
            return super().start()

        def error(self):
            return QtAudio.Error.OpenError if fails(self, "source_error") else super().error()

        def stop(self):
            self.stopped = True
            super().stop()

        def deleteLater(self):
            self.deleted = True
            super().deleteLater()

    class Sink(sink_type):
        def __init__(self, endpoint, fmt, parent):
            self.identifier = bytes(endpoint.id()).decode()
            if fails(self, "sink_constructor"):
                raise RuntimeError("injected startup failure")
            super().__init__(endpoint, fmt, parent)
            self.deleted = False
            self.stopped = False
            created.append(self)

        def setVolume(self, volume):
            if fails(self, "sink_volume"):
                raise RuntimeError("injected startup failure")
            super().setVolume(volume)

        def reset(self):
            self.stopped = True
            super().reset()

        def deleteLater(self):
            self.deleted = True
            super().deleteLater()

    monkeypatch.setattr(audio, "QAudioSource", Source)
    monkeypatch.setattr(audio, "QAudioSink", Sink)
    with pytest.raises(RuntimeError, match="injected startup failure|Cannot open User"):
        hub.start(routes)
    assert not hub.running and not hub.routes
    assert not hub.sources and not hub.sinks and not hub.inputs and not hub.output_ids
    assert not hub.buffers and not any(hub.queues.values())
    assert hub.tails == {0: b"", 1: b""}
    assert all(endpoint.deleted and endpoint.stopped for endpoint in created)

    failing = False
    hub.start(routes)
    assert hub.running and hub.routes == routes
    assert set(hub.inputs) == {0, 1}
    assert hub.capture_bytes == hub.played_bytes == [0, 0]
    hub.stop()
    assert all(endpoint.deleted and endpoint.stopped for endpoint in created)


def test_completed_playback_closes_its_system_stream(qtbot):
    from types import SimpleNamespace
    from PySide6.QtCore import QBuffer
    from PySide6.QtMultimedia import QtAudio
    from translitter.audio import AudioHub
    hub = AudioHub()
    hub.running = True
    calls = []
    hub.sinks[0] = SimpleNamespace(error=lambda: QtAudio.Error.NoError, reset=lambda: calls.append("reset"))
    hub.buffers[0] = QBuffer(hub)
    hub._output_state(0, QtAudio.State.IdleState)
    assert calls == ["reset"]
    assert not hub.buffers
    hub.sinks.clear()
    hub.stop()
