"""Explicit output cancellation must not settle replacement playback."""
import numpy as np
import pytest
from PySide6.QtMultimedia import QtAudio

from test_audio import explicit_audio
from test_audio_stop_lifecycle import feed
from translitter.audio import prepare_pcm
from translitter.core import make_routes


@pytest.mark.parametrize("listener", [0, 1])
@pytest.mark.parametrize("error", [False, True])
@pytest.mark.parametrize("action", ["none", "stop", "restart", "cancel", "replace", "dispatch", "other"])
def test_cancel_reset_cannot_settle_newer_output(explicit_audio, listener, error, action):
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
        assert hub.play(who, np.full(8, .5, np.float32), 16000)
    old = hub.buffers[listener]
    sink = hub.sinks[listener]
    original_reset = sink.reset
    fresh = {}

    def reset():
        sink.reset = original_reset
        original_reset()
        # Real reset clears the error; only the pre-reset query is failing.
        sink.error = lambda: QtAudio.Error.NoError
        if action == "stop":
            hub.stop()
        elif action == "restart":
            hub.start(replacement_routes)
            for who in (0, 1):
                feed(hub, who, b"\x00\x20\xcd")
                assert hub.play(who, np.full(20, .125, np.float32), 16000)
                assert hub.play(who, np.full(10, -.25, np.float32), 16000)
        elif action in ("cancel", "replace"):
            hub.stop_playback(listener)
        elif action == "other":
            hub.stop_playback(1 - listener)
        if action in ("replace", "dispatch"):
            assert hub.play(listener, np.full(20, .125, np.float32), 16000)
            assert hub.play(listener, np.full(10, -.25, np.float32), 16000)
        fresh.update(sinks=dict(hub.sinks), sources=dict(hub.sources), inputs=dict(hub.inputs),
                     buffers=dict(hub.buffers), queues={n: list(q) for n, q in hub.queues.items()},
                     events=list(events), opened=list(opened), capture=list(hub.capture_bytes),
                     played=list(hub.played_bytes), tails=dict(hub.tails), routes=hub.routes,
                     output_ids=dict(hub.output_ids))

    sink.reset = reset
    if error:
        sink.error = lambda: QtAudio.Error.IOError
        hub._output_state(listener, QtAudio.State.StoppedState)
    else:
        hub.stop_playback(listener)
    assert not old.isOpen()
    superseded = action not in ("none", "other")
    expected_events = fresh["events"] + ([] if superseded else [(listener, False)])
    assert events == expected_events, "obsolete cancellation published playing(False)"
    assert len(failures) == int(error and not superseded), "obsolete failure reached newer output"
    assert opened == fresh["opened"]
    assert hub.sinks == fresh["sinks"] and hub.sources == fresh["sources"]
    assert hub.inputs == fresh["inputs"] and hub.buffers == fresh["buffers"]
    assert hub.routes == fresh["routes"] and hub.output_ids == fresh["output_ids"]
    assert hub.capture_bytes == fresh["capture"] and hub.played_bytes == fresh["played"]
    assert hub.tails == fresh["tails"]
    assert {n: list(q) for n, q in hub.queues.items()} == fresh["queues"]
    assert all(buffer.isOpen() for buffer in hub.buffers.values())
    if action == "stop":
        assert not hub.running and not hub.sinks
        hub.start(replacement_routes)
    if action in ("restart", "replace", "dispatch"):
        assert bytes(hub.buffers[listener].data()) == prepare_pcm(np.full(20, .125, np.float32), 16000)
        hub._output_state(listener, QtAudio.State.IdleState)
        assert bytes(hub.buffers[listener].data()) == prepare_pcm(np.full(10, -.25, np.float32), 16000)
    # Both microphones and both output FIFOs remain usable after each branch.
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
