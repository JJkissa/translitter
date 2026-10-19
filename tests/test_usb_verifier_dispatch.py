"""Synchronous USB startup/playback failures must retire outer callbacks."""
from types import SimpleNamespace

import pytest

from test_usb_verifier_lifecycle import verifier


def prepare_start(v):
    pairs = [object(), object()]
    routes = (SimpleNamespace(speaker=0), SimpleNamespace(speaker=1))
    v.ns["hub"].headsets = lambda: pairs
    v.ns["make_routes"] = lambda *actual: routes if list(actual) == pairs else ()
    v.ns["hub"].set_volume = lambda value: v.events.append(("volume", value))
    v.ns["worker"].set_session = lambda *args: v.events.append(("session", args))

    def submit(*args):
        v.events.append(("submit", args))
        return True

    v.ns["worker"].submit = submit
    return routes


@pytest.mark.parametrize("raises", [False, True])
def test_start_failure_signal_prevents_session_and_submission(verifier, raises):
    v = verifier
    routes = prepare_start(v)

    def start(actual):
        assert actual == routes
        v.events.append("start")
        v.ns["fail"]("synchronous startup failure")
        if raises:
            raise RuntimeError("retired startup exception")

    v.ns["hub"].start = start
    v.ns["begin"]()
    assert v.state["failed"] and not v.state["done"]
    assert v.events == [("volume", 0.30), "start", "stop", "shutdown", "quit"]
    assert v.evidence["errors"] == ["synchronous startup failure"]
    assert v.pending == []


def test_live_start_still_submits_first_fixture(verifier):
    v = verifier
    routes = prepare_start(v)
    v.ns["hub"].start = lambda actual: v.events.append(("start", actual))
    v.ns["begin"]()
    assert v.events == [("volume", 0.30), ("start", routes),
                        ("session", (1, routes)),
                        ("submit", (1, 0, v.ns["fixtures"][0]))]
    assert not v.state["failed"]
    assert v.pending == []


def make_turn(speaker):
    return SimpleNamespace(generation=1, speaker=speaker, listener=1 - speaker,
                           original="Hello", translation="Hei", source_language="en",
                           target_language="fi", elapsed=0.5, rate=16000, audio=[0.25, 0.5])


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("outcome", ["accepted", "rejected", "exception"])
def test_play_failure_signal_does_not_schedule_graph(verifier, speaker, outcome):
    v = verifier
    v.evidence["turns"].clear()
    turn = make_turn(speaker)

    def play(listener, audio, rate):
        assert (listener, audio, rate) == (turn.listener, turn.audio, turn.rate)
        v.events.append("play")
        v.ns["fail"]("synchronous playback failure")
        if outcome == "exception":
            raise RuntimeError("retired playback exception")
        return outcome == "accepted"

    v.ns["hub"].play = play
    v.ns["receive"](turn)
    assert v.state["failed"] and not v.state["done"]
    assert v.events == ["play", "stop", "shutdown", "quit"]
    assert v.evidence["errors"] == ["synchronous playback failure"]
    assert len(v.evidence["turns"]) == 1
    assert v.pending == []


@pytest.mark.parametrize("speaker", [0, 1])
def test_live_result_still_schedules_listener_graph(verifier, speaker):
    v = verifier
    v.evidence["turns"].clear()
    turn = make_turn(speaker)
    v.ns["hub"].play = lambda *args: v.events.append(("play", args)) or True
    v.ns["receive"](turn)
    assert not v.state["failed"]
    assert v.state["playing"] == turn.listener
    assert v.events == [("play", (turn.listener, turn.audio, turn.rate))]
    assert len(v.pending) == 1
    listeners = []
    v.ns["inspect_graph"] = listeners.append
    v.pending.pop()()
    assert listeners == [turn.listener]
