import importlib.util


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
