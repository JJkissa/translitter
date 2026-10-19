"""Endpoint constructors returning after retirement cannot acquire ownership."""
import numpy as np
import pytest
from PySide6.QtMultimedia import QtAudio

from test_audio import explicit_audio
from test_audio_stop_lifecycle import feed
from translitter import audio
from translitter.audio import prepare_pcm
from translitter.core import make_routes


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("stage", ["sink", "source"])
@pytest.mark.parametrize("action", ["stop", "restart"])
def test_constructor_return_respects_superseding_lifecycle(
        explicit_audio, monkeypatch, speaker, stage, action):
    hub, pairs, opened = explicit_audio
    replacement = make_routes(*pairs, "ru", "uk")
    source_base, sink_base = audio.QAudioSource, audio.QAudioSink
    sources, sinks, retired, fresh = [], [], [], {}
    armed = True
    events = []
    hub.playing.connect(lambda who, active: events.append((who, active)))

    def supersede():
        nonlocal armed
        armed = False
        retired.extend(sources + sinks)
        if action == "restart":
            hub.start(replacement)
            for who in (0, 1):
                feed(hub, who, b"\x00\x40\xab")
                assert hub.play(who, np.full(20, .125, np.float32), 16000)
                assert hub.play(who, np.full(10, -.25, np.float32), 16000)
        else:
            hub.stop()
        fresh.update(sources=dict(hub.sources), sinks=dict(hub.sinks),
                     inputs=dict(hub.inputs), buffers=dict(hub.buffers),
                     opened=list(opened), events=list(events),
                     constructed=(len(sources), len(sinks)),
                     queues={n: list(q) for n, q in hub.queues.items()})

    class Source(source_base):
        def __init__(self, *args):
            super().__init__(*args)
            self.deleted = False
            self.configurations = self.starts = self.stops = 0
            sources.append(self)
            if armed and stage == "source" and self.identifier == pairs[speaker].microphone_id:
                supersede()

        def setBufferSize(self, size):
            self.configurations += 1
            super().setBufferSize(size)

        def start(self):
            self.starts += 1
            return super().start()

        def stop(self):
            self.stops += 1
            super().stop()

        def deleteLater(self):
            self.deleted = True
            super().deleteLater()

    class Sink(sink_base):
        def __init__(self, *args):
            super().__init__(*args)
            self.deleted = False
            self.configurations = self.resets = 0
            sinks.append(self)
            if armed and stage == "sink" and self.identifier == replacement[speaker].output_id:
                supersede()

        def setVolume(self, volume):
            self.configurations += 1
            super().setVolume(volume)

        def reset(self):
            self.resets += 1
            super().reset()

        def deleteLater(self):
            self.deleted = True
            super().deleteLater()

    monkeypatch.setattr(audio, "QAudioSource", Source)
    monkeypatch.setattr(audio, "QAudioSink", Sink)
    hub.start(make_routes(*pairs))
    assert not armed
    assert (len(sources), len(sinks)) == fresh["constructed"]
    assert opened == fresh["opened"]
    assert hub.sources == fresh["sources"]
    assert hub.sinks == fresh["sinks"]
    assert hub.inputs == fresh["inputs"]
    assert hub.buffers == fresh["buffers"]
    assert events == fresh["events"]
    assert {n: list(q) for n, q in hub.queues.items()} == fresh["queues"]
    assert all(endpoint.deleted for endpoint in retired)
    assert all(endpoint.stops == 1 and not endpoint.buffer.isOpen()
               for endpoint in retired if isinstance(endpoint, Source))
    assert all(endpoint.resets == 1 for endpoint in retired if isinstance(endpoint, Sink))
    target = sources[speaker] if stage == "source" else sinks[speaker]
    assert target.configurations == 0
    if stage == "source":
        assert target.starts == 0
    if action == "restart":
        assert hub.running and hub.routes == replacement
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
    else:
        assert not hub.running and not hub.routes and not hub.output_ids
        assert not hub.sources and not hub.sinks and not hub.inputs
        assert hub.tails == {0: b"", 1: b""}
        hub.start(replacement)
        assert hub.running and hub.routes == replacement
