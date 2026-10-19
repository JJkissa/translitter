"""Runtime volume propagation must not configure retired headphone sinks."""
import numpy as np
import pytest
from PySide6.QtMultimedia import QtAudio

from test_audio import explicit_audio
from test_audio_stop_lifecycle import feed
from translitter import audio
from translitter.audio import prepare_pcm
from translitter.core import make_routes


@pytest.mark.parametrize("listener", [0, 1])
@pytest.mark.parametrize("action", ["stop", "restart", "restart_volume"])
def test_volume_update_preserves_superseding_session(explicit_audio, monkeypatch, listener, action):
    hub, pairs, opened = explicit_audio
    base = audio.QAudioSink
    calls = []

    class Sink(base):
        def setVolume(self, volume):
            calls.append((self, volume))
            self.volume = volume
            super().setVolume(volume)

    monkeypatch.setattr(audio, "QAudioSink", Sink)
    hub.start(make_routes(*pairs))
    old_sinks = dict(hub.sinks)
    target = old_sinks[listener]
    original = target.setVolume
    fresh = {}
    replacement = make_routes(*pairs, "ru", "uk")

    def supersede(volume):
        original(volume)
        target.setVolume = original
        if action == "stop":
            hub.stop()
        else:
            hub.start(replacement)
            if action == "restart_volume":
                hub.set_volume(.2)
            for who in (0, 1):
                feed(hub, who, b"\x00\x40\xab")
                assert hub.play(who, np.full(20, .125, np.float32), 16000)
                assert hub.play(who, np.full(10, -.25, np.float32), 16000)
        fresh.update(calls=list(calls), sinks=dict(hub.sinks), sources=dict(hub.sources),
                     inputs=dict(hub.inputs), buffers=dict(hub.buffers), opened=list(opened),
                     queues={n: list(q) for n, q in hub.queues.items()})

    target.setVolume = supersede
    hub.set_volume(.7)
    # No calls at all after the synchronous replacement/cancellation returned.
    assert calls == fresh["calls"]
    assert hub.sinks == fresh["sinks"] and hub.sources == fresh["sources"]
    assert hub.inputs == fresh["inputs"] and hub.buffers == fresh["buffers"]
    assert opened == fresh["opened"]
    assert {n: list(q) for n, q in hub.queues.items()} == fresh["queues"]
    if action == "stop":
        assert not hub.running and not hub.routes and not hub.sinks
        hub.start(replacement)
        assert all(sink.volume == .7 for sink in hub.sinks.values())
    else:
        assert hub.running and hub.routes == replacement
        assert hub.volume == (.2 if action == "restart_volume" else .7)
        assert all(sink.volume == hub.volume for sink in hub.sinks.values())
        assert hub.output_ids == {1: "out-b", 0: "out-a"}
        assert hub.capture_bytes == [2, 2] and hub.played_bytes == [40, 40]
        assert hub.tails == {0: b"\xab", 1: b"\xab"}
        captured = []
        hub.samples.connect(lambda who, pcm: captured.append((who, pcm.copy())))
        for who in (0, 1):
            feed(hub, who, b"\x01")
            assert captured[-1][0] == who
            np.testing.assert_array_equal(captured[-1][1], np.array([427 / 32768], np.float32))
            hub._output_state(who, QtAudio.State.IdleState)
            assert bytes(hub.buffers[who].data()) == prepare_pcm(np.full(10, -.25, np.float32), 16000)
            hub._output_state(who, QtAudio.State.IdleState)
        assert hub.capture_bytes == [4, 4] and hub.played_bytes == [60, 60]
        assert not hub.buffers and not any(hub.queues.values())


@pytest.mark.parametrize("value,expected", [(-1, 0), (0, 0), (.3, .3), (.8, .8), (2, .8), ("0.6", .6)])
def test_volume_clips_and_applies_to_both_listeners(explicit_audio, monkeypatch, value, expected):
    hub, pairs, opened = explicit_audio
    base = audio.QAudioSink

    class Sink(base):
        def setVolume(self, volume):
            self.volume = volume

    monkeypatch.setattr(audio, "QAudioSink", Sink)
    hub.set_volume(value)
    assert hub.volume == expected and not hub.sinks
    hub.start(make_routes(*pairs))
    assert all(s.volume == expected for s in hub.sinks.values())
    hub.set_volume(.4)
    assert all(s.volume == .4 for s in hub.sinks.values())
    hub.set_volume(value)
    assert all(s.volume == expected for s in hub.sinks.values())
    hub.stop()
    hub.set_volume(value)
    assert hub.volume == expected and not hub.sinks


@pytest.mark.parametrize("value,error", [(None, TypeError), (object(), TypeError), ("bad", ValueError)])
def test_volume_conversion_error_preserves_current_setting(explicit_audio, value, error):
    hub, pairs, opened = explicit_audio
    hub.start(make_routes(*pairs))
    calls = []
    for who, sink in hub.sinks.items():
        sink.setVolume = lambda volume, n=who: calls.append((n, volume))
    with pytest.raises(error):
        hub.set_volume(value)
    assert hub.volume == .45 and not calls and hub.running
