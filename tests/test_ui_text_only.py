"""Text-only transitions with real widgets and fake physical endpoints."""
import numpy as np
import pytest

from test_ui import gui, start, turn
from test_ui_automatic import start_automatic, voice, silence


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("prefix_size", [160, 8000])
@pytest.mark.parametrize("other_playing", [False, True])
def test_text_only_preserves_nonplaying_automatic_capture(
        gui, qtbot, monkeypatch, speaker, prefix_size, other_playing):
    import translitter.ui as ui
    monkeypatch.setattr(ui, "monotonic", lambda: 100.0)
    window, audio, worker = gui
    start_automatic(window, worker, qtbot)
    if other_playing:
        worker.result.emit(turn(window, speaker=speaker))
        assert window._playback[1 - speaker]
    generation = window.generation
    audio.samples.emit(speaker, np.full(prefix_size, .12, dtype=np.float32))
    window.text_only.setChecked(True)
    assert window.active and window.generation == generation
    assert window._cooldown[speaker] == 0.0
    audio.samples.emit(speaker, voice(.3))
    audio.samples.emit(speaker, np.zeros(9920, dtype=np.float32))
    assert len(worker.submissions) == 1
    samples = worker.submissions[0][2]
    assert np.count_nonzero(samples == np.float32(.12)) == prefix_size
    assert np.count_nonzero(samples == np.float32(.3)) == 8000


@pytest.mark.parametrize("speaker", [0, 1])
def test_duplicate_playback_stop_does_not_erase_new_automatic_speech(
        gui, qtbot, monkeypatch, speaker):
    import translitter.ui as ui
    now = [100.0]
    monkeypatch.setattr(ui, "monotonic", lambda: now[0])
    window, audio, worker = gui
    start_automatic(window, worker, qtbot)
    worker.result.emit(turn(window, speaker=1 - speaker))
    window.text_only.setChecked(True)
    assert not window._playback[speaker]
    assert window._cooldown[speaker] == pytest.approx(100.3)
    now[0] = 101.0
    audio.samples.emit(speaker, voice(.12))
    state = window.cards[speaker].state.text()
    audio.playing.emit(speaker, False)
    assert window.cards[speaker].state.text() == state
    assert window._cooldown[speaker] == pytest.approx(100.3)
    audio.samples.emit(speaker, voice(.3))
    audio.samples.emit(speaker, silence())
    assert len(worker.submissions) == 1
    samples = worker.submissions[0][2]
    assert np.count_nonzero(samples == np.float32(.12)) == 8000
    assert np.count_nonzero(samples == np.float32(.3)) == 8000


@pytest.mark.parametrize("listener", [0, 1])
def test_text_only_discards_deferred_playback_without_losing_held_input(gui, qtbot, listener):
    window, audio, worker = gui
    start(window, worker, qtbot)
    window._press(listener)
    audio.samples.emit(listener, voice())
    for _ in range(2):
        worker.result.emit(turn(window, speaker=1 - listener))
    assert len(window._deferred[listener]) == 2
    window.text_only.setChecked(True)
    assert window._deferred == [[], []]
    window.text_only.setChecked(False)
    window._release(listener)
    assert len(worker.submissions) == 1
    np.testing.assert_array_equal(worker.submissions[0][2], voice())
    assert not audio.plays
    worker.result.emit(turn(window, speaker=1 - listener))
    assert len(audio.plays) == 1


@pytest.mark.parametrize("listener", [0, 1])
def test_text_only_stops_active_and_deferred_output_but_keeps_real_cooldown(
        gui, qtbot, monkeypatch, listener):
    import translitter.ui as ui
    now = [100.0]
    monkeypatch.setattr(ui, "monotonic", lambda: now[0])
    window, audio, worker = gui
    start_automatic(window, worker, qtbot)
    for _ in range(3):
        worker.result.emit(turn(window, speaker=1 - listener))
    assert len(audio.plays) == 1 and len(window._deferred[listener]) == 2
    window.text_only.setChecked(True)
    assert not window._playback[listener]
    assert window._deferred == [[], []]
    audio.samples.emit(listener, voice(.2))
    assert not window.segmenters[listener].active
    window.text_only.setChecked(False)
    now[0] = 100.301
    audio.samples.emit(listener, voice(.3))
    audio.samples.emit(listener, silence())
    assert len(worker.submissions) == 1
    assert np.count_nonzero(worker.submissions[0][2] == np.float32(.3)) == 8000
    assert not np.any(worker.submissions[0][2] == np.float32(.2))
    assert len(audio.plays) == 1
