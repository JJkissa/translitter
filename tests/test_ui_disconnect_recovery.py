"""Selected endpoint removal at the Window boundary; no physical devices."""
from dataclasses import replace

import numpy as np
import pytest

from test_ui import gui, start, turn


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("automatic", [False, True])
@pytest.mark.parametrize("notification", ["immediate", "delayed", "endpoint_changed"])
def test_disconnect_clears_capture_and_recovers_after_reselection(
        gui, qtbot, speaker, automatic, notification):
    window, audio, worker = gui
    if automatic:
        window.mode.setCurrentIndex(1)
    start(window, worker, qtbot)
    generation = window.generation
    original_pairs = list(audio.pairs)
    voice = np.full(8161, .17, dtype=np.float32)
    assert window.submit_audio(speaker, voice)
    stale = turn(window, speaker=speaker)
    if not automatic:
        window._press(speaker)
    audio.samples.emit(speaker, voice)
    audio.level.emit(speaker, .1)
    window._deferred[speaker].append((voice, 16000))
    window._playback[1 - speaker] = True
    if notification == "delayed":
        # The first Qt notification still sees the old enumeration. No second
        # notification is guaranteed: the scheduled discovery must recheck it.
        audio.devices_changed.emit()
        assert window.active
    if notification == "endpoint_changed":
        audio.pairs[speaker] = replace(original_pairs[speaker],
                                       output_id="replacement-output")
    else:
        audio.pairs = [pair for pair in original_pairs
                       if pair.key != original_pairs[speaker].key]
    if notification == "delayed":
        qtbot.waitUntil(lambda: not window.active, timeout=1800)
    else:
        audio.devices_changed.emit()
    assert not window.active and window.generation > generation
    assert "disconnected" in window.status.text()
    assert worker.stops[-1] == window.generation and audio.stops >= 1
    assert window._held == [False, False] and window._chunks == [[], []]
    assert window._sizes == [0, 0] and window._capture_overflow == [False, False]
    assert window._busy == [0, 0] and window._deferred == [[], []]
    assert window._playback == [False, False] and window._cooldown == [0., 0.]
    assert all(not s.active and not len(s.pending) and not s.preroll
               for s in window.segmenters)
    assert all(card.state.text() == "Stopped" and card.meter.value() == 0
               and not card.hold.isEnabled() for card in window.cards)
    assert not window.stop_button.isEnabled()
    assert all(selector.isEnabled() for selector in window.selectors)
    submitted = len(worker.submissions)
    window._release(speaker)
    audio.samples.emit(speaker, voice)
    worker.result.emit(stale)
    worker.settled.emit(generation, speaker, False)
    assert len(worker.submissions) == submitted and not audio.plays
    assert not window.cards[speaker].original.toPlainText()

    audio.pairs = original_pairs
    audio.devices_changed.emit()
    for selector, pair in zip(window.selectors, original_pairs):
        selector.setCurrentIndex(next(i for i in range(selector.count())
                                      if selector.itemData(i).key == pair.key))
    assert window.start_button.isEnabled()
    start(window, worker, qtbot)
    fresh = np.full(8000, .29, dtype=np.float32)
    if not automatic:
        window._press(speaker)
    audio.samples.emit(speaker, fresh)
    if automatic:
        audio.samples.emit(speaker, np.zeros(9920, dtype=np.float32))
    else:
        window._release(speaker)
    assert len(worker.submissions) == submitted + 1
    assert worker.submissions[-1][:2] == (window.generation, speaker)
    samples = worker.submissions[-1][2]
    assert np.count_nonzero(samples == np.float32(.29)) == len(fresh)
    assert not np.any(samples == np.float32(.17))


@pytest.mark.parametrize("automatic", [False, True])
def test_unselected_device_change_preserves_current_capture(gui, qtbot, automatic):
    window, audio, worker = gui
    if automatic:
        window.mode.setCurrentIndex(1)
    start(window, worker, qtbot)
    generation = window.generation
    voice = np.full(8161, .19, dtype=np.float32)
    if not automatic:
        window._press(0)
    audio.samples.emit(0, voice)
    audio.pairs.append(replace(audio.pairs[0], key="extra", microphone_id="extra-mic",
                               output_id="extra-output"))
    audio.devices_changed.emit()
    # Exercise the real single-shot timer, not a manually invoked helper.
    qtbot.waitUntil(lambda: not window._discovery.isActive(), timeout=1800)
    assert window.active and window.generation == generation
    assert not audio.stops and not worker.submissions
    if automatic:
        audio.samples.emit(0, np.zeros(9920, dtype=np.float32))
    else:
        window._release(0)
    assert len(worker.submissions) == 1
    assert np.count_nonzero(worker.submissions[0][2] == np.float32(.19)) == len(voice)
