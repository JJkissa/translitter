"""Full AudioHub teardown must not retire a synchronously restarted session."""
import numpy as np
import pytest
from PySide6.QtCore import QByteArray, QIODevice
from PySide6.QtMultimedia import QtAudio

from test_audio import explicit_audio
from translitter.audio import prepare_pcm
from translitter.core import make_routes


def feed(hub, speaker, data):
    io = hub.inputs[speaker]
    io.close()
    io.setData(QByteArray(data))
    io.open(QIODevice.ReadOnly)
    io.readyRead.emit()


@pytest.mark.parametrize("trigger", [0, 1])
@pytest.mark.parametrize("nested", [False, True])
def test_stop_settlement_preserves_replacement_session(explicit_audio, trigger, nested):
    hub, pairs, opened = explicit_audio
    routes = make_routes(*pairs)
    hub.start(routes)
    old_sources = dict(hub.sources)
    old_sinks = dict(hub.sinks)
    retired = []
    for listener in (0, 1):
        assert hub.play(listener, np.full(16, .25, np.float32), 16000)
        assert hub.play(listener, np.full(8, .5, np.float32), 16000)
        retired.append(hub.buffers[listener])
    deleted = []
    for listener, sink in old_sinks.items():
        sink.deleteLater = lambda n=listener: deleted.append(n)
    fresh = {}
    notifications = []
    captured = []
    hub.samples.connect(lambda who, pcm: captured.append((who, pcm.copy())))
    hub.playing.connect(lambda who, active: notifications.append((who, active)))

    def restart(who, active):
        if who != trigger or active:
            return
        hub.playing.disconnect(restart)
        assert not hub.running and not hub.sinks and not hub.sources
        assert not hub.inputs and not hub.buffers and not hub.output_ids
        assert not hub.routes and not any(hub.queues.values())
        assert hub.tails == {0: b"", 1: b""}
        assert sorted(deleted) == [0, 1]
        assert all(not source.buffer.isOpen() for source in old_sources.values())
        assert all(not buffer.isOpen() for buffer in retired)
        if nested:
            hub.stop()
        hub.start(routes)
        for listener in (0, 1):
            assert hub.play(listener, np.full(20, .125, np.float32), 16000)
            assert hub.play(listener, np.full(10, -.25, np.float32), 16000)
            feed(hub, listener, b"\x00\x40\xab")
        fresh.update(sinks=dict(hub.sinks), sources=dict(hub.sources),
                     inputs=dict(hub.inputs), buffers=dict(hub.buffers),
                     queues={n: list(q) for n, q in hub.queues.items()},
                     notifications=len(notifications))

    hub.playing.connect(restart)
    hub.stop()
    assert hub.running
    assert hub.sinks == fresh["sinks"]
    assert hub.sources == fresh["sources"]
    assert hub.inputs == fresh["inputs"]
    assert hub.buffers == fresh["buffers"]
    assert hub.routes == routes
    assert hub.output_ids == {1: "out-b", 0: "out-a"}
    assert hub.tails == {0: b"\xab", 1: b"\xab"}
    assert hub.capture_bytes == [2, 2]
    assert hub.played_bytes == [40, 40]
    assert {n: list(q) for n, q in hub.queues.items()} == fresh["queues"]
    assert len(notifications) == fresh["notifications"]
    assert sorted(deleted) == [0, 1]
    assert all(not source.buffer.isOpen() for source in old_sources.values())
    assert all(not buffer.isOpen() for buffer in retired)
    assert all(buffer.isOpen() for buffer in fresh["buffers"].values())
    for speaker in (0, 1):
        feed(hub, speaker, b"\x01")
        assert captured[-1][0] == speaker
        np.testing.assert_array_equal(captured[-1][1], np.array([427 / 32768], np.float32))
        assert hub.tails[speaker] == b""
        hub._output_state(speaker, QtAudio.State.IdleState)
        assert bytes(hub.buffers[speaker].data()) == prepare_pcm(np.full(10, -.25, np.float32), 16000)
        hub._output_state(speaker, QtAudio.State.IdleState)
    assert hub.capture_bytes == [4, 4]
    assert hub.played_bytes == [60, 60]
    assert not hub.buffers and not any(hub.queues.values())


@pytest.mark.parametrize("nested", [False, True])
def test_stop_cleans_both_participants_and_is_reusable(explicit_audio, nested):
    hub, pairs, opened = explicit_audio
    routes = make_routes(*pairs)
    hub.start(routes)
    sources = dict(hub.sources)
    buffers = []
    for speaker in (0, 1):
        feed(hub, speaker, b"\xff")
        assert hub.play(speaker, np.ones(10, np.float32), 16000)
        assert hub.play(speaker, np.ones(5, np.float32), 16000)
        buffers.append(hub.buffers[speaker])
    if nested:
        def stop_again(who, active):
            if not active:
                hub.playing.disconnect(stop_again)
                hub.stop()
        hub.playing.connect(stop_again)
    hub.stop()
    hub.stop()
    assert not hub.running
    assert not hub.sources and not hub.sinks and not hub.inputs
    assert not hub.buffers and not hub.output_ids and not hub.routes
    assert hub.tails == {0: b"", 1: b""}
    assert not any(hub.queues.values())
    assert all(not source.buffer.isOpen() for source in sources.values())
    assert all(not buffer.isOpen() for buffer in buffers)
    hub.start(routes)
    assert hub.running and hub.capture_bytes == [0, 0] and hub.played_bytes == [0, 0]
    assert hub.play(0, np.ones(5, np.float32), 16000)
    hub._output_state(0, QtAudio.State.IdleState)
    assert not hub.buffers
