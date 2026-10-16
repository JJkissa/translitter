"""Runtime monitor regressions with fake endpoints, never physical audio."""
from types import SimpleNamespace

import pytest
from PySide6.QtMultimedia import QtAudio

from test_audio import explicit_audio
from translitter.core import make_routes


@pytest.mark.parametrize("kind", ["input", "output"])
@pytest.mark.parametrize("restart", [False, True])
def test_poll_does_not_query_retired_endpoints_after_failure(explicit_audio, kind, restart):
    hub, pairs, _ = explicit_audio
    routes = make_routes(*pairs)
    hub.start(routes)
    sources = list(hub.sources.items())
    sinks = list(hub.sinks.items())
    retired_queries = []
    retired = False
    failures = []

    def state_query(label, state):
        if retired:
            retired_queries.append(label)
        return state

    for speaker, source in sources:
        source.state = lambda n=speaker: state_query(("input", n), QtAudio.State.ActiveState)
    for listener, sink in sinks:
        sink.state = lambda n=listener: state_query(("output", n), QtAudio.State.StoppedState)
    failed = sources[0][1] if kind == "input" else sinks[0][1]
    failed.error = lambda: QtAudio.Error.IOError

    def on_failure(speaker, message):
        nonlocal retired
        failures.append((speaker, message))
        hub.stop()
        retired = True
        if restart:
            hub.start(routes)

    hub.failure.connect(on_failure)
    hub._poll_states()

    assert len(failures) == 1
    assert retired_queries == [], "The monitor queried an endpoint retired by synchronous failure handling"
    assert hub.running is restart
    if restart:
        assert all(hub.sources[n] is not source for n, source in sources)
        assert all(hub.sinks[n] is not sink for n, sink in sinks)


@pytest.mark.parametrize("endpoint", ["microphone", "headphones"])
def test_selected_device_disconnect_stops_and_clears_audio(explicit_audio, endpoint):
    import numpy as np
    hub, pairs, _ = explicit_audio
    hub.start(make_routes(*pairs))
    assert hub.play(0, np.ones(160, dtype=np.float32), 16000)
    assert hub.play(0, np.ones(160, dtype=np.float32), 16000)
    events = []
    failures = []
    hub.devices_changed.connect(lambda: events.append("changed"))
    hub.failure.connect(lambda speaker, message: failures.append((speaker, message)))
    method = "audioInputs" if endpoint == "microphone" else "audioOutputs"
    setattr(hub.manager, method, lambda: [])
    hub._devices_changed()
    assert events == ["changed"]
    assert len(failures) == 1 and failures[0][0] == -1
    assert "disconnected" in failures[0][1]
    assert not hub.running
    assert not hub.sources and not hub.sinks and not hub.inputs and not hub.buffers
    assert not any(hub.queues.values())
    hub._poll_states()
    assert len(failures) == 1


def test_unselected_device_change_does_not_stop_conversation(explicit_audio):
    hub, pairs, _ = explicit_audio
    hub.start(make_routes(*pairs))
    original = hub.manager.audioInputs
    hub.manager.audioInputs = lambda: original() + [SimpleNamespace(id=lambda: b"unselected")]
    failures = []
    hub.failure.connect(lambda *args: failures.append(args))
    hub._devices_changed()
    assert hub.running and not failures


@pytest.mark.parametrize("error", [QtAudio.Error.NoError, QtAudio.Error.UnderrunError])
def test_recoverable_output_state_is_not_reported_as_failure(explicit_audio, error):
    hub, pairs, _ = explicit_audio
    hub.start(make_routes(*pairs))
    failures = []
    hub.failure.connect(lambda *args: failures.append(args))
    for sink in hub.sinks.values():
        sink.error = lambda: error
    hub._poll_states()
    assert hub.running and not failures
