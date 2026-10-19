"""Synchronous playback signals may retire output; all endpoints are fake."""
import numpy as np
import pytest
from PySide6.QtMultimedia import QtAudio

from test_audio import explicit_audio
from translitter.core import make_routes


@pytest.mark.parametrize("listener", [0, 1])
@pytest.mark.parametrize("action", ["stop", "restart", "cancel", "replace"])
def test_playing_slot_cannot_dispatch_retired_buffer(explicit_audio, listener, action):
    hub, pairs, opened = explicit_audio
    routes = make_routes(*pairs)
    hub.start(routes)
    old_sink = hub.sinks[listener]
    old_starts = []
    original_start = old_sink.start

    def track_start(buffer):
        old_starts.append(buffer)
        original_start(buffer)

    old_sink.start = track_start
    retired = []
    replacement = []

    def retire(who, active):
        if who != listener or not active:
            return
        hub.playing.disconnect(retire)
        retired.append(hub.buffers[listener])
        if action in ("stop", "restart"):
            hub.stop()
            if action == "restart":
                hub.start(routes)
        else:
            hub.stop_playback(listener)
        if action in ("restart", "replace"):
            assert hub.play(listener, np.ones(80, np.float32), 16000)
            replacement.append(hub.buffers[listener])

    hub.playing.connect(retire)
    hub.play(listener, np.ones(160, np.float32), 16000)
    assert not retired[0].isOpen()
    assert hub.played_bytes[listener] == (160 if replacement else 0)
    assert len(old_starts) == (1 if action == "replace" else 0)
    assert all(buffer is not retired[0] for buffer in old_starts)
    if replacement:
        assert hub.buffers[listener] is replacement[0]
        assert replacement[0].isOpen()
        hub._output_state(listener, QtAudio.State.IdleState)
        assert listener not in hub.buffers
    else:
        assert listener not in hub.buffers
    # Recovery after cancellation or a complete session retirement.
    if not hub.running:
        hub.start(routes)
    before = hub.played_bytes[listener]
    assert hub.play(listener, np.ones(40, np.float32), 16000)
    assert hub.played_bytes[listener] == before + 80
    hub._output_state(listener, QtAudio.State.IdleState)
    assert not hub.queues[listener] and listener not in hub.buffers


@pytest.mark.parametrize("listener", [0, 1])
@pytest.mark.parametrize("error", [False, True])
@pytest.mark.parametrize("action", ["none", "stop", "restart"])
def test_output_settlement_does_not_publish_retired_failure(explicit_audio, listener, error, action):
    hub, pairs, opened = explicit_audio
    routes = make_routes(*pairs)
    hub.start(routes)
    assert hub.play(listener, np.ones(160, np.float32), 16000)
    old_buffer = hub.buffers[listener]
    old_sink = hub.sinks[listener]
    failures, replacements = [], []
    hub.failure.connect(lambda who, message: failures.append((who, message)))

    def settle(who, active):
        if who != listener or active:
            return
        hub.playing.disconnect(settle)
        if action != "none":
            hub.stop()
            if action == "restart":
                hub.start(routes)
                assert hub.play(listener, np.ones(80, np.float32), 16000)
                replacements.append(hub.buffers[listener])

    hub.playing.connect(settle)
    if error:
        old_sink.error = lambda: QtAudio.Error.IOError
    hub._output_state(listener, QtAudio.State.IdleState)
    assert not old_buffer.isOpen()
    assert len(failures) == (1 if error and action == "none" else 0)
    if action == "restart":
        assert hub.running
        assert hub.buffers[listener] is replacements[0]
        assert replacements[0].isOpen()
        assert hub.played_bytes[listener] == 160
        assert not hub.queues[listener]
        hub._output_state(listener, QtAudio.State.IdleState)
        assert listener not in hub.buffers
    elif action == "stop":
        assert not hub.running and not hub.sinks and not hub.buffers
    else:
        old_sink.error = lambda: QtAudio.Error.NoError
        assert hub.play(listener, np.ones(40, np.float32), 16000)
        hub._output_state(listener, QtAudio.State.IdleState)
        assert listener not in hub.buffers
