"""Playback completion may be superseded synchronously by sink.reset."""
import numpy as np
import pytest
from PySide6.QtMultimedia import QtAudio

from test_audio import explicit_audio
from test_audio_stop_lifecycle import feed
from translitter.audio import prepare_pcm
from translitter.core import make_routes


@pytest.mark.parametrize("listener", [0, 1])
@pytest.mark.parametrize("queued", [False, True])
@pytest.mark.parametrize("action", ["none", "stop", "restart", "restart_empty", "cancel", "replace"])
def test_completion_reset_preserves_newer_output(explicit_audio, listener, queued, action):
    hub, pairs, opened = explicit_audio
    routes = make_routes(*pairs)
    replacement_routes = make_routes(*pairs, "ru", "uk")
    hub.start(routes)
    events, failures = [], []
    hub.playing.connect(lambda who, active: events.append((who, active)))
    hub.failure.connect(lambda *args: failures.append(args))
    for who in (0, 1):
        feed(hub, who, b"\x00\x40\xab")
        assert hub.play(who, np.full(16, .25, np.float32), 16000)
    if queued:
        assert hub.play(listener, np.full(8, .5, np.float32), 16000)
    retired = hub.buffers[listener]
    sink = hub.sinks[listener]
    original_reset = sink.reset
    fresh = {}

    def supersede():
        # Disarm before nested Stop calls reset again.
        sink.reset = original_reset
        original_reset()
        if action == "stop":
            hub.stop()
        elif action.startswith("restart"):
            hub.start(replacement_routes)
            for who in (0, 1):
                feed(hub, who, b"\x00\x20\xcd")
            if action == "restart":
                for who in (0, 1):
                    assert hub.play(who, np.full(20, .125, np.float32), 16000)
                    assert hub.play(who, np.full(10, -.25, np.float32), 16000)
        elif action in ("cancel", "replace"):
            hub.stop_playback(listener)
            if action == "replace":
                assert hub.play(listener, np.full(20, .125, np.float32), 16000)
                assert hub.play(listener, np.full(10, -.25, np.float32), 16000)
        fresh.update(sinks=dict(hub.sinks), sources=dict(hub.sources), inputs=dict(hub.inputs),
                     buffers=dict(hub.buffers), queues={n: list(q) for n, q in hub.queues.items()},
                     events=list(events), opened=list(opened), capture=list(hub.capture_bytes),
                     played=list(hub.played_bytes), tails=dict(hub.tails), routes=hub.routes)

    sink.reset = supersede
    hub._output_state(listener, QtAudio.State.IdleState)
    assert not retired.isOpen()
    assert not failures
    if action == "none":
        if queued:
            assert bytes(hub.buffers[listener].data()) == prepare_pcm(np.full(8, .5, np.float32), 16000)
            assert hub.played_bytes[listener] == 48
            assert events[-1] == (listener, True)
            hub._output_state(listener, QtAudio.State.IdleState)
        assert listener not in hub.buffers
        assert events[-1] == (listener, False)
    else:
        assert events == fresh["events"], "retired completion published a stale settlement/start"
        assert opened == fresh["opened"]
        assert hub.buffers == fresh["buffers"]
        assert hub.sinks == fresh["sinks"] and hub.sources == fresh["sources"]
        assert hub.inputs == fresh["inputs"] and hub.routes == fresh["routes"]
        assert hub.capture_bytes == fresh["capture"] and hub.played_bytes == fresh["played"]
        assert hub.tails == fresh["tails"]
        assert {n: list(q) for n, q in hub.queues.items()} == fresh["queues"]
        assert all(buffer.isOpen() for buffer in fresh["buffers"].values())
        if action == "stop":
            assert not hub.running and not hub.buffers and not hub.sinks
            hub.start(replacement_routes)
        elif action in ("restart", "replace"):
            assert bytes(hub.buffers[listener].data()) == prepare_pcm(np.full(20, .125, np.float32), 16000)
            hub._output_state(listener, QtAudio.State.IdleState)
            assert bytes(hub.buffers[listener].data()) == prepare_pcm(np.full(10, -.25, np.float32), 16000)
            hub._output_state(listener, QtAudio.State.IdleState)
    # Fresh microphone and FIFO recovery after every branch.
    captured = []
    hub.samples.connect(lambda who, pcm: captured.append((who, pcm.copy())))
    for who in (0, 1):
        tail = hub.tails[who]
        feed(hub, who, b"\x01" if tail else b"\x01\x00")
        expected = int.from_bytes(tail + b"\x01", "little", signed=True) if tail else 1
        assert captured[-1][0] == who
        np.testing.assert_array_equal(captured[-1][1], np.array([expected / 32768], np.float32))
        while who in hub.buffers:
            hub._output_state(who, QtAudio.State.IdleState)
        assert hub.play(who, np.ones(4, np.float32), 16000)
        hub._output_state(who, QtAudio.State.IdleState)
    assert not hub.buffers and not any(hub.queues.values())


@pytest.mark.parametrize("listener", [0, 1])
@pytest.mark.parametrize("other_only", [False, True])
def test_completion_reset_scopes_output_identity(explicit_audio, listener, other_only):
    hub, pairs, _ = explicit_audio
    hub.start(make_routes(*pairs))
    for who in (0, 1):
        assert hub.play(who, np.ones(16, np.float32), 16000)
    if other_only:
        assert hub.play(listener, np.full(8, .5, np.float32), 16000)
    sink = hub.sinks[listener]
    original_reset = sink.reset
    fresh = {}

    def reset():
        sink.reset = original_reset
        original_reset()
        if other_only:
            hub.stop_playback(1 - listener)
        else:
            # The old buffer is detached; new dispatch need not call Stop first.
            assert hub.play(listener, np.full(20, .125, np.float32), 16000)
            assert hub.play(listener, np.full(10, -.25, np.float32), 16000)
            fresh["buffer"] = hub.buffers[listener]

    sink.reset = reset
    hub._output_state(listener, QtAudio.State.IdleState)
    if other_only:
        assert bytes(hub.buffers[listener].data()) == prepare_pcm(np.full(8, .5, np.float32), 16000)
        assert listener not in hub.queues or not hub.queues[listener]
    else:
        assert hub.buffers[listener] is fresh["buffer"] and fresh["buffer"].isOpen()
        assert len(hub.queues[listener]) == 1
        assert hub.played_bytes[listener] == 72
        hub._output_state(listener, QtAudio.State.IdleState)
        assert bytes(hub.buffers[listener].data()) == prepare_pcm(np.full(10, -.25, np.float32), 16000)
    hub._output_state(listener, QtAudio.State.IdleState)
    assert listener not in hub.buffers and not hub.queues[listener]
