"""PTT buffers the complete turn; no pause or elapsed time can submit it."""
import numpy as np
import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QCheckBox
from test_ui import gui, start


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("control", ["button", "keyboard"])
def test_ptt_waits_for_release_across_pauses_and_eight_seconds(gui, qtbot, speaker, control):
    window, audio, worker = gui
    start(window, worker, qtbot)
    card = window.cards[speaker]
    key = Qt.Key_F1 if speaker == 0 else Qt.Key_F2
    if control == "button":
        qtbot.mousePress(card.hold, Qt.LeftButton)
    else:
        qtbot.keyPress(card.original, key)
    parts = [np.full(16000 * 2, .12, dtype=np.float32),
             np.zeros(16000 * 2, dtype=np.float32),
             np.full(16000 * 5, .23, dtype=np.float32)]
    for part in parts:
        audio.samples.emit(speaker, part)
        qtbot.wait(20)
        assert not worker.submissions, "Neither pauses nor the old eight-second split may submit held speech"
        assert not audio.plays
        assert not card.original.toPlainText() and not card.translated.toPlainText()
        assert window._held[speaker]
    if control == "button":
        qtbot.mouseRelease(card.hold, Qt.LeftButton)
    else:
        qtbot.keyRelease(card.original, key)
    assert len(worker.submissions) == 1
    generation, submitted_speaker, samples = worker.submissions[0]
    assert (generation, submitted_speaker) == (window.generation, speaker)
    np.testing.assert_array_equal(samples, np.concatenate(parts))
    window._release(speaker)
    assert len(worker.submissions) == 1, "A repeated release must not duplicate the turn"


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("single_chunk", [False, True])
def test_overlong_ptt_is_discarded_without_early_or_partial_translation(gui, qtbot, speaker, single_chunk):
    window, audio, worker = gui
    start(window, worker, qtbot)
    limit = 16000 * 30
    window._press(speaker)
    lengths = [limit + 1] if single_chunk else [limit, 1]
    for length in lengths:
        audio.samples.emit(speaker, np.full(length, .1, dtype=np.float32))
    assert window._held[speaker]
    assert not worker.submissions
    assert window._sizes[speaker] == 0 and not window._chunks[speaker], "Over-limit audio must be discarded, not silently truncated"
    assert "30" in window.status.text() and "discard" in window.status.text().lower()
    audio.samples.emit(speaker, np.ones(16000, dtype=np.float32))
    assert not window._chunks[speaker], "Keep the rejected hold bounded until release"
    window._release(speaker)
    assert not window._held[speaker] and not worker.submissions
    assert "discard" in window.cards[speaker].state.text().lower()
    window._press(speaker)
    fresh = np.full(8000, .2, dtype=np.float32)
    audio.samples.emit(speaker, fresh)
    window._release(speaker)
    assert len(worker.submissions) == 1
    np.testing.assert_array_equal(worker.submissions[0][2], fresh)


def test_exact_capture_limit_waits_for_release_and_is_not_truncated(gui, qtbot):
    window, audio, worker = gui
    start(window, worker, qtbot)
    samples = np.full(16000 * 30, .1, dtype=np.float32)
    window._press(0)
    audio.samples.emit(0, samples)
    assert not worker.submissions
    window._release(0)
    assert len(worker.submissions) == 1
    np.testing.assert_array_equal(worker.submissions[0][2], samples)


def test_explicit_audio_submission_rejects_overlimit_without_truncation(gui, qtbot):
    window, audio, worker = gui
    start(window, worker, qtbot)
    assert window.submit_audio(0, np.zeros(16000 * 30 + 1, dtype=np.float32)) is False
    assert not worker.submissions


def test_push_to_talk_has_no_live_translation_switch(gui):
    window, audio, worker = gui
    assert not any(box.text() == "Live PTT" for box in window.findChildren(QCheckBox))
    assert not hasattr(window, "live_ptt"), "Live PTT must not be accidentally re-enabled"


def test_active_ptt_cannot_be_switched_to_pause_triggered_capture(gui, qtbot):
    window, audio, worker = gui
    start(window, worker, qtbot)
    window._press(0)
    audio.samples.emit(0, np.full(8000, .1, dtype=np.float32))
    window.mode.setCurrentIndex(window.mode.findData("auto"))
    assert window.mode.currentData() == "ptt", "Capture mode is fixed until the conversation stops"
    audio.samples.emit(0, np.full(8000, .2, dtype=np.float32))
    audio.samples.emit(0, np.zeros(16000, dtype=np.float32))
    assert not worker.submissions
    window._release(0)
    assert len(worker.submissions) == 1
    assert len(worker.submissions[0][2]) == 8000 + 8000 + 16000
