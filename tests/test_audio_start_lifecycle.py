"""Initial teardown callbacks may supersede an outer audio start."""
import numpy as np
import pytest
from PySide6.QtMultimedia import QtAudio

from test_audio import explicit_audio
from test_audio_stop_lifecycle import feed
from translitter.audio import prepare_pcm
from translitter.core import make_routes


@pytest.mark.parametrize("trigger", [0, 1])
@pytest.mark.parametrize("action", ["restart", "stop"])
def test_start_initial_settlement_respects_newer_lifecycle(explicit_audio, trigger, action):
    hub, pairs, opened = explicit_audio
    routes = make_routes(*pairs)
    replacement_routes = make_routes(*pairs, "ru", "uk")
    hub.start(routes)
    old_sources = dict(hub.sources)
    old_sinks = dict(hub.sinks)
    deleted = []
    for listener, sink in old_sinks.items():
        sink.deleteLater = lambda n=listener: deleted.append(n)
        assert hub.play(listener, np.full(8, .25, np.float32), 16000)
    old_buffers = dict(hub.buffers)
    fresh = {}
    events = []
    hub.playing.connect(lambda who, active: events.append((who, active)))

    def supersede(who, active):
        if who != trigger or active:
            return
        hub.playing.disconnect(supersede)
        if action == "restart":
            hub.start(replacement_routes)
            for speaker in (0, 1):
                feed(hub, speaker, b"\x00\x40\xab")
                assert hub.play(speaker, np.full(20, .125, np.float32), 16000)
                assert hub.play(speaker, np.full(10, -.25, np.float32), 16000)
        else:
            hub.stop()
        fresh.update(sources=dict(hub.sources), sinks=dict(hub.sinks),
                     inputs=dict(hub.inputs), buffers=dict(hub.buffers),
                     opened=list(opened), events=list(events),
                     queues={n: list(q) for n, q in hub.queues.items()})

    hub.playing.connect(supersede)
    hub.start(make_routes(*pairs, "en", "fi"))
    # The obsolete outer call must not create/overwrite any endpoints or state.
    assert opened == fresh["opened"]
    assert hub.sources == fresh["sources"]
    assert hub.sinks == fresh["sinks"]
    assert hub.inputs == fresh["inputs"]
    assert hub.buffers == fresh["buffers"]
    assert events == fresh["events"]
    assert {n: list(q) for n, q in hub.queues.items()} == fresh["queues"]
    assert sorted(deleted) == [0, 1]
    assert all(not source.buffer.isOpen() for source in old_sources.values())
    assert all(not buffer.isOpen() for buffer in old_buffers.values())
    if action == "restart":
        assert hub.running and hub.routes == replacement_routes
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
        assert not hub.running and not hub.routes and not hub.output_ids
        assert not hub.sources and not hub.sinks and not hub.inputs
        assert hub.tails == {0: b"", 1: b""}
        # Cancellation does not poison a subsequent explicit start.
        hub.start(replacement_routes)
        assert hub.running and hub.routes == replacement_routes


def test_start_replaces_active_session_without_reentrant_callbacks(explicit_audio):
    hub, pairs, opened = explicit_audio
    hub.start(make_routes(*pairs))
    sources = dict(hub.sources)
    for speaker in (0, 1):
        feed(hub, speaker, b"\x00\x40\xab")
        assert hub.play(speaker, np.full(10, .5, np.float32), 16000)
        assert hub.play(speaker, np.full(5, .25, np.float32), 16000)
    buffers = dict(hub.buffers)
    routes = make_routes(*pairs, "ru", "uk")
    hub.start(routes)
    assert hub.running and hub.routes == routes
    assert hub.capture_bytes == [0, 0] and hub.played_bytes == [0, 0]
    assert hub.tails == {0: b"", 1: b""}
    assert not hub.buffers and not any(hub.queues.values())
    assert all(not source.buffer.isOpen() for source in sources.values())
    assert all(not buffer.isOpen() for buffer in buffers.values())
    assert all(hub.sources[n] is not sources[n] for n in (0, 1))
    assert hub.play(0, np.ones(5, np.float32), 16000)
    hub._output_state(0, QtAudio.State.IdleState)
    assert not hub.buffers
