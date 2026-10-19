"""Later GUI verifier steps must stop after synchronous failure signals."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from test_gui_verifier_lifecycle import verifier


def prepare_step(v, step, failure=None):
    w = v.window
    after_abort = []
    snapshot = []

    def operation(name):
        v.events.append(name)
        if v.state["failed"]:
            after_abort.append(name)
        if name == failure:
            v.ns["abort"]("synchronous " + name)
            snapshot.append(deepcopy(v.evidence))

    v.state["step"] = step
    v.ns["fixtures"] = [[1, 2], [3, 4, 5]]
    v.ns["np"] = np
    w.active = True
    w._held = [step in (2, 3), step == 1]
    w._sizes = [2 if step in (2, 3) else 0, 3 if step == 1 else 0]
    w._busy = [0, 0]
    w._deferred = [[object()] if step == 3 else [], [object()] if step == 1 else []]
    w._playback = [False, False]
    w.audio.buffers = {}
    w.audio.played_bytes = [10, 0 if step == 1 else 20]
    w.audio.capture_bytes = [30, 40]
    w.audio.sources = []
    w.audio.sinks = []
    for index, card in enumerate(w.cards):
        card.original = SimpleNamespace(toPlainText=lambda: "original")
        card.translated.toPlainText = lambda: "translation"
        card.state = SimpleNamespace(text=lambda: "Waiting")
    w.status = SimpleNamespace(text=lambda: "Input cancelled")
    w.grab = lambda: SimpleNamespace(save=lambda path: operation("screenshot") or True)
    w.stop_button, w.start_button = object(), object()

    def click(button, modifier):
        w.active = button is w.start_button
        operation("start" if w.active else "stop")

    def key_event(speaker, pressed):
        w._held[speaker] = pressed
        if not pressed and step == 1:
            w._busy[speaker] = 1
        operation(f"{'press' if pressed else 'release'}.{speaker}")

    def samples(speaker, data):
        w._sizes[speaker] = len(data)
        operation(f"samples.{speaker}")

    def speak(speaker):
        operation(f"speak.{speaker}")

    def send_event(target, event):
        if event == "deactivate":
            w._held[0] = False
            w._sizes[0] = 0
            w._deferred[0] = []
        operation(event)

    def close():
        if v.state["failed"]:
            v.events.append("abort.close")
            return
        operation("close")

    w.close = close
    w.audio.samples = SimpleNamespace(emit=samples)
    w.text_only = SimpleNamespace(setChecked=lambda value: operation("text_only"))
    w.submit_audio = lambda speaker, data: operation("submit") or True
    v.ns.update(key_event=key_event, speak=speak,
                Qt=SimpleNamespace(LeftButton=1), QTest=SimpleNamespace(mouseClick=click),
                QApplication=SimpleNamespace(sendEvent=send_event),
                QEvent=type("Event", (), {"WindowDeactivate": "deactivate",
                    "WindowActivate": "activate", "__new__": lambda cls, kind: kind}))
    v.evidence["focus_loss"] = {}
    return after_abort, snapshot


@pytest.mark.parametrize("step,stage", [
    (1, "release.1"),
    (2, "screenshot"), (2, "stop"), (2, "start"), (2, "press.0"),
    (2, "samples.0"), (2, "speak.1"),
    (3, "deactivate"), (3, "activate"), (3, "release.0"),
    (4, "text_only"), (4, "submit"), (5, "stop"), (5, "close"),
])
def test_abort_stops_later_step_continuation(verifier, step, stage):
    v = verifier
    after_abort, snapshot = prepare_step(v, step, stage)
    path = v.root / ".local/gui-integration.json"
    path.write_bytes(b"previous evidence")
    v.ns["check"]()
    assert v.state["failed"] and not v.state["done"]
    assert after_abort == []
    assert v.state["step"] == step
    assert v.evidence == snapshot[0]
    assert v.evidence["errors"] == ["synchronous " + stage]
    assert path.read_bytes() == b"previous evidence"
    v.events.clear()
    v.ns["check"]()
    assert v.events == []


@pytest.mark.parametrize("step,expected", [
    (1, ["release.1"]),
    (2, ["screenshot", "stop", "start", "press.0", "samples.0", "speak.1"]),
    (3, ["deactivate", "activate", "release.0"]),
    (4, ["text_only", "submit"]), (5, ["stop", "close"]),
])
def test_live_later_step_still_advances(verifier, step, expected):
    v = verifier
    prepare_step(v, step)
    v.ns["check"]()
    assert not v.state["failed"]
    assert v.state["step"] == step + 1
    assert v.events == expected
