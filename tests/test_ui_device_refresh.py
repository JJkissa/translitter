"""Stopped-window discovery continuity using native widgets and fake devices."""
from dataclasses import replace

import pytest

from test_ui import gui
from translitter.core import make_routes


@pytest.mark.parametrize("missing_speaker", [0, 1])
def test_refresh_preserves_survivor_without_duplicate_fallback(gui, missing_speaker):
    window, audio, worker = gui
    original = list(audio.pairs)
    window._swap()
    selected = [s.currentData() for s in window.selectors]
    survivor = selected[1 - missing_speaker]
    replacement = replace(original[0], key="c", microphone_id="mic-c",
                          output_id="out-c", label="USB Gamma")
    # Put the survivor at the missing participant's positional default index.
    audio.pairs = [survivor, replacement] if missing_speaker == 0 else [replacement, survivor]
    worker.ready.emit()
    window.refresh_devices()
    actual = [s.currentData() for s in window.selectors]
    assert actual[1 - missing_speaker] == survivor
    assert actual[missing_speaker] == replacement
    assert window.start_button.isEnabled()
    expected = make_routes(*actual)
    assert window._selected_routes() == expected
    window.start_session()
    assert audio.starts == [expected]
    assert worker.sessions[-1] == (window.generation, expected)


@pytest.mark.parametrize("swapped", [False, True])
@pytest.mark.parametrize("same_label", [False, True])
def test_refresh_preserves_identity_across_reorder_and_profile_change(gui, swapped, same_label):
    window, audio, worker = gui
    if swapped:
        window._swap()
    selected = [s.currentData().key for s in window.selectors]
    audio.pairs = [replace(p, microphone_id=p.microphone_id + "-new",
                           output_id=p.output_id + "-new",
                           label="Identical USB" if same_label else p.label)
                   for p in reversed(audio.pairs)]
    worker.ready.emit()
    window.refresh_devices()
    assert [s.currentData().key for s in window.selectors] == selected
    for selector in window.selectors:
        current = selector.currentData()
        assert current == next(p for p in audio.pairs if p.key == current.key)
        assert current.microphone_id in selector.toolTip()
        assert current.output_id in selector.toolTip()
    assert window.start_button.isEnabled()
    assert not audio.starts


@pytest.mark.parametrize("available", [0, 1])
def test_refresh_partial_availability_and_reappearance(gui, available):
    window, audio, worker = gui
    original = list(audio.pairs)
    worker.ready.emit()
    audio.pairs = original[:available]
    window.refresh_devices()
    assert all(s.count() == available for s in window.selectors)
    assert not window.start_button.isEnabled()
    if not available:
        assert all(s.currentData() is None for s in window.selectors)
    else:
        assert all(s.currentData() == original[0] for s in window.selectors)
    audio.pairs = original
    window.refresh_devices()
    # An explicit duplicate is preserved, including the single-device selection.
    if available:
        assert [s.currentData().key for s in window.selectors] == ["a", "a"]
        assert not window.start_button.isEnabled()
        window.selectors[1].setCurrentIndex(1)
    assert window.start_button.isEnabled()
    assert window._selected_routes() == make_routes(*original)
    assert not audio.starts


@pytest.mark.parametrize("speaker", [0, 1])
def test_refresh_does_not_undo_explicit_duplicate_selection(gui, speaker):
    window, audio, worker = gui
    window.selectors[1 - speaker].setCurrentIndex(speaker)
    audio.pairs.reverse()
    worker.ready.emit()
    window.refresh_devices()
    assert [s.currentData().key for s in window.selectors] == ["ab"[speaker]] * 2
    assert not window.start_button.isEnabled()
    window.start_session()
    assert not window.active and not audio.starts


@pytest.mark.parametrize("guard", ["active", "closing"])
def test_refresh_ignores_active_or_closing_window(gui, guard):
    window, audio, worker = gui
    if guard == "active":
        worker.ready.emit()
        window.start_session()
    else:
        window._closing = True
    previous = [s.currentData() for s in window.selectors]
    try:
        def forbidden_discovery():
            pytest.fail("guarded refresh must not enumerate devices")
        audio.headsets = forbidden_discovery
        window.refresh_devices()
        assert [s.currentData() for s in window.selectors] == previous
    finally:
        if guard == "closing":
            window._closing = False
