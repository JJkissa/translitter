"""Real GUI startup/PTT callbacks isolated from models and physical audio."""
from copy import deepcopy
from types import SimpleNamespace

import pytest

from test_gui_verifier_lifecycle import verifier


def prepare_capture(v, failure=None):
    """Keep valid UI state after abort to expose obsolete continuation."""
    after_abort = []

    def operation(name):
        v.events.append(name)
        if v.state["failed"]:
            after_abort.append(name)
        if name == failure:
            v.ns["abort"]("synchronous " + name)

    w = v.window
    w.active = True
    w._held = [False, False]
    w._sizes = [0, 0]
    w._busy = [0, 0]
    w.audio.sources = [object(), object()]
    w.audio.headsets = lambda: [object(), object()]
    w.refresh_devices = lambda: operation("refresh")
    w.start_button = SimpleNamespace(isEnabled=lambda: True)
    w.selectors = [SimpleNamespace(currentText=lambda: operation("selector") or "USB")
                   for _ in range(2)]
    w.audio.set_volume = lambda value: operation("volume")
    fixtures = [[0.1, 0.2], [0.3, 0.4, 0.5]]
    v.ns["fixtures"] = fixtures
    v.ns["Qt"] = SimpleNamespace(Key_F1=1, Key_F2=2, NoModifier=0, LeftButton=1)
    v.ns["QEvent"] = SimpleNamespace(KeyPress="press", KeyRelease="release")
    v.ns["QKeyEvent"] = lambda kind, key, modifier: (kind, key)
    v.ns["QTest"] = SimpleNamespace(mouseClick=lambda *args: operation("click"))
    for speaker, card in enumerate(w.cards):
        card.original = speaker

    def send_event(speaker, event):
        kind, key = event
        assert key == speaker + 1
        w._held[speaker] = kind == "press"
        if kind == "release":
            w._busy[speaker] += 1
        operation(f"{kind}.{speaker}")

    def emit(speaker, samples):
        assert samples is fixtures[speaker]
        w._sizes[speaker] = len(samples)
        operation(f"samples.{speaker}")

    v.ns["QApplication"] = SimpleNamespace(sendEvent=send_event)
    w.audio.samples = SimpleNamespace(emit=emit)
    return after_abort


@pytest.mark.parametrize("stage", ["refresh", "click", "volume", "press.1", "samples.1",
                                   "press.0", "samples.0", "release.0"])
def test_startup_abort_stops_outer_continuation(verifier, stage):
    v = verifier
    after_abort = prepare_capture(v, stage)
    path = v.root / ".local/gui-integration.json"
    path.write_bytes(b"previous evidence")
    v.ns["check"]()
    assert v.state["failed"] and not v.state["done"]
    assert after_abort == []
    assert v.state["step"] == 0
    assert v.evidence["errors"] == ["synchronous " + stage]
    if stage in ("refresh", "click", "volume"):
        assert "headsets" not in v.evidence
    assert path.read_bytes() == b"previous evidence"
    before = deepcopy(v.evidence)
    v.events.clear()
    v.ns["check"]()
    assert v.events == [] and v.evidence == before


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("stage", ["press", "samples", "release"])
def test_speak_abort_does_not_publish_further_input(verifier, speaker, stage):
    v = verifier
    after_abort = prepare_capture(v, f"{stage}.{speaker}")
    v.ns["speak"](speaker)
    assert v.state["failed"]
    assert after_abort == []
    assert v.evidence["errors"] == [f"synchronous {stage}.{speaker}"]


@pytest.mark.parametrize("terminal", ["failed", "done"])
def test_retired_ptt_callbacks_are_inert(verifier, terminal):
    v = verifier
    prepare_capture(v)
    v.state[terminal] = True
    for speaker in (0, 1):
        v.ns["key_event"](speaker, True)
        v.ns["key_event"](speaker, False)
        v.ns["speak"](speaker)
    assert v.events == []
    assert v.window._held == [False, False]
    assert v.window._sizes == [0, 0]
    assert v.window._busy == [0, 0]


def test_live_startup_preserves_overlap_and_submits_other_speaker(verifier):
    v = verifier
    prepare_capture(v)
    v.ns["check"]()
    assert not v.state["failed"] and v.state["step"] == 1
    assert v.events == ["refresh", "click", "volume", "selector", "selector",
                        "press.1", "samples.1", "press.0", "samples.0", "release.0"]
    assert v.window._held == [False, True]
    assert v.window._sizes == [2, 3]
    assert v.window._busy == [1, 0]


@pytest.mark.parametrize("speaker", [0, 1])
def test_live_speak_still_captures_and_submits(verifier, speaker):
    v = verifier
    prepare_capture(v)
    v.ns["speak"](speaker)
    assert v.events == [f"press.{speaker}", f"samples.{speaker}", f"release.{speaker}"]
    assert not v.window._held[speaker] and v.window._busy[speaker] == 1
    assert v.window._sizes[speaker] == len(v.ns["fixtures"][speaker])
    assert not v.state["failed"]
