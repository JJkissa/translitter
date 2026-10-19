"""Device notifications cannot tear down a synchronously recovered session."""
import numpy as np
import pytest
from PySide6.QtMultimedia import QtAudio

from test_audio import explicit_audio
from test_audio_stop_lifecycle import feed
from translitter.audio import prepare_pcm
from translitter.core import make_routes


@pytest.mark.parametrize("endpoint", ["audioInputs", "audioOutputs"])
@pytest.mark.parametrize("signal", ["devices_changed", "failure"])
@pytest.mark.parametrize("action", ["restart", "stop"])
def test_disconnect_callback_preserves_newer_lifecycle(explicit_audio, endpoint, signal, action):
    hub, pairs, opened = explicit_audio
    hub.start(make_routes(*pairs))
    old_sources = dict(hub.sources)
    for listener in (0, 1):
        assert hub.play(listener, np.full(8, .5, np.float32), 16000)
        assert hub.play(listener, np.full(4, .25, np.float32), 16000)
    old_buffers = dict(hub.buffers)
    enumerate_original = getattr(hub.manager, endpoint)
    setattr(hub.manager, endpoint, lambda: [])
    failures = []
    hub.failure.connect(lambda *args: failures.append(args))
    fresh = {}
    replacement_routes = make_routes(*pairs, "ru", "uk")
    events = []
    hub.playing.connect(lambda *args: events.append(args))
    notification = getattr(hub, signal)

    def supersede(*args):
        notification.disconnect(supersede)
        hub.stop()
        setattr(hub.manager, endpoint, enumerate_original)
        if action == "restart":
            hub.start(replacement_routes)
            for speaker in (0, 1):
                feed(hub, speaker, b"\x00\x40\xab")
                assert hub.play(speaker, np.full(20, .125, np.float32), 16000)
                assert hub.play(speaker, np.full(10, -.25, np.float32), 16000)
        fresh.update(sources=dict(hub.sources), inputs=dict(hub.inputs),
                     sinks=dict(hub.sinks), buffers=dict(hub.buffers),
                     queues={n: list(q) for n, q in hub.queues.items()},
                     opened=list(opened), events=list(events), lifecycle=hub._lifecycle)

    notification.connect(supersede)
    hub._devices_changed()
    assert hub.running is (action == "restart")
    assert hub._lifecycle == fresh["lifecycle"]
    assert hub.sources == fresh["sources"]
    assert hub.inputs == fresh["inputs"]
    assert hub.sinks == fresh["sinks"]
    assert hub.buffers == fresh["buffers"]
    assert {n: list(q) for n, q in hub.queues.items()} == fresh["queues"]
    assert opened == fresh["opened"] and events == fresh["events"]
    assert len(failures) == (1 if signal == "failure" else 0)
    if failures:
        assert failures[0] == (-1, "A selected headset disconnected; conversation stopped")
    assert all(not source.buffer.isOpen() for source in old_sources.values())
    assert all(not buffer.isOpen() for buffer in old_buffers.values())
    if action == "restart":
        assert hub.routes == replacement_routes
        assert hub.output_ids == {1: "out-b", 0: "out-a"}
        assert hub.capture_bytes == [2, 2] and hub.played_bytes == [40, 40]
        assert hub.tails == {0: b"\xab", 1: b"\xab"}
        captured = []
        hub.samples.connect(lambda who, pcm: captured.append((who, pcm.copy())))
        for speaker in (0, 1):
            feed(hub, speaker, b"\x01")
            assert captured[-1][0] == speaker
            np.testing.assert_array_equal(captured[-1][1], np.array([427 / 32768], np.float32))
            hub._output_state(speaker, QtAudio.State.IdleState)
            assert bytes(hub.buffers[speaker].data()) == prepare_pcm(np.full(10, -.25, np.float32), 16000)
            hub._output_state(speaker, QtAudio.State.IdleState)
        assert hub.capture_bytes == [4, 4] and hub.played_bytes == [60, 60]
        assert not hub.buffers and not any(hub.queues.values())
    else:
        assert not hub.routes and not hub.output_ids
        assert hub.tails == {0: b"", 1: b""}
        hub.start(replacement_routes)
        assert hub.running and hub.routes == replacement_routes


@pytest.mark.parametrize("endpoint", ["audioInputs", "audioOutputs"])
@pytest.mark.parametrize("participant", [0, 1])
def test_current_disconnect_still_cleans_owned_audio(explicit_audio, endpoint, participant):
    hub, pairs, _ = explicit_audio
    routes = make_routes(*pairs)
    hub.start(routes)
    old_sources = dict(hub.sources)
    for speaker in (0, 1):
        feed(hub, speaker, b"\x00\x40\xab")
        assert hub.play(speaker, np.full(20, .125, np.float32), 16000)
        assert hub.play(speaker, np.full(10, -.25, np.float32), 16000)
    old_buffers = dict(hub.buffers)
    original = getattr(hub.manager, endpoint)
    devices = original()
    setattr(hub.manager, endpoint, lambda: [d for n, d in enumerate(devices) if n != participant])
    failures = []
    hub.failure.connect(lambda *args: failures.append(args))
    hub._devices_changed()
    assert len(failures) == 1 and failures[0][0] == -1
    assert not hub.running and not hub.routes and not hub.output_ids
    assert not hub.sources and not hub.sinks and not hub.inputs and not hub.buffers
    assert not any(hub.queues.values()) and hub.tails == {0: b"", 1: b""}
    assert all(not source.buffer.isOpen() for source in old_sources.values())
    assert all(not buffer.isOpen() for buffer in old_buffers.values())
    hub._devices_changed()
    assert len(failures) == 1
    setattr(hub.manager, endpoint, original)
    hub.start(routes)
    assert hub.running
    for speaker in (0, 1):
        assert hub.sources[speaker] is not old_sources[speaker]
        feed(hub, speaker, b"\x00\x20")
        assert hub.play(speaker, np.full(10, .25, np.float32), 16000)
        hub._output_state(speaker, QtAudio.State.IdleState)
    assert hub.capture_bytes == [2, 2] and hub.played_bytes == [20, 20]


@pytest.mark.parametrize("endpoint", ["audioInputs", "audioOutputs"])
def test_unrelated_change_preserves_capture_and_playback(explicit_audio, endpoint):
    from types import SimpleNamespace
    hub, pairs, _ = explicit_audio
    hub.start(make_routes(*pairs))
    for speaker in (0, 1):
        feed(hub, speaker, b"\x00\x40\xab")
        assert hub.play(speaker, np.full(20, .125, np.float32), 16000)
        assert hub.play(speaker, np.full(10, -.25, np.float32), 16000)
    sources, sinks, buffers = dict(hub.sources), dict(hub.sinks), dict(hub.buffers)
    queues = {n: list(q) for n, q in hub.queues.items()}
    original = getattr(hub.manager, endpoint)
    setattr(hub.manager, endpoint, lambda: original() + [SimpleNamespace(id=lambda: b"unrelated")])
    failures = []
    hub.failure.connect(lambda *args: failures.append(args))
    lifecycle = hub._lifecycle
    hub._devices_changed()
    assert hub.running and not failures and hub._lifecycle == lifecycle
    assert hub.sources == sources and hub.sinks == sinks and hub.buffers == buffers
    assert {n: list(q) for n, q in hub.queues.items()} == queues
    assert hub.tails == {0: b"\xab", 1: b"\xab"}
    assert hub.capture_bytes == [2, 2] and hub.played_bytes == [40, 40]
    for speaker in (0, 1):
        feed(hub, speaker, b"\x01")
        hub._output_state(speaker, QtAudio.State.IdleState)
        assert bytes(hub.buffers[speaker].data()) == queues[speaker][0]
        hub._output_state(speaker, QtAudio.State.IdleState)
    assert hub.capture_bytes == [4, 4] and hub.played_bytes == [60, 60]
    assert not hub.buffers and not any(hub.queues.values())
