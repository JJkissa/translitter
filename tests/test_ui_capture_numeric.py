"""Capture numeric admission must not alter accepted PTT/automatic audio."""
import warnings

import numpy as np
import pytest

from test_ui import gui, start


def retained(window, speaker):
    segmenter = window.segmenters[speaker]
    return (
        tuple(part.tobytes() for part in segmenter.active),
        segmenter.pending.tobytes(),
        tuple(part.tobytes() for part in segmenter.preroll),
        segmenter.length, segmenter.voiced, segmenter.quiet,
        tuple(part.tobytes() for part in window._chunks[speaker]),
        window._sizes[speaker], window._capture_overflow[speaker],
        window.cards[speaker].state.text(), tuple(window._busy),
    )


INVALID = [
    np.full(321, .3 + .2j, dtype=np.complex64),
    np.full(321, .3 + 0j, dtype=np.complex128),
    np.full(321, np.finfo(np.float64).max),
    np.full(321, np.nan), np.full(321, np.inf),
    np.ones((2, 321)), np.array(.3),
    np.array(["not PCM"]), [[.1], [.2, .3]],
]


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("mode", ["auto", "ptt"])
@pytest.mark.parametrize("invalid", INVALID)
def test_invalid_capture_preserves_accepted_input_and_recovers(gui, qtbot, speaker, mode, invalid):
    window, audio, worker = gui
    window.mode.setCurrentIndex(window.mode.findData(mode))
    start(window, worker, qtbot)
    if mode == "ptt":
        window._press(speaker)
    # Active full frames plus pending subframe samples in automatic mode.
    prefix = np.full(4161, .125, dtype=np.float32)
    window._samples(speaker, prefix)
    before = [retained(window, n) for n in (0, 1)]
    with warnings.catch_warnings(), np.errstate(over="raise", invalid="raise"):
        warnings.simplefilter("error")
        # Direct slot invocation exposes exceptions that Qt would only log.
        window._samples(speaker, invalid)
    assert [retained(window, n) for n in (0, 1)] == before
    assert not worker.submissions
    suffix = np.full(4159, .25, dtype=np.float32)
    audio.samples.emit(speaker, suffix)
    if mode == "auto":
        audio.samples.emit(speaker, np.zeros(9600, dtype=np.float32))
    else:
        window._release(speaker)
    assert len(worker.submissions) == 1
    generation, actual_speaker, samples = worker.submissions[0]
    assert (generation, actual_speaker) == (window.generation, speaker)
    np.testing.assert_array_equal(samples[:8320], np.concatenate((prefix, suffix)))
    assert np.count_nonzero(samples) == 8320


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("mode", ["auto", "ptt"])
@pytest.mark.parametrize("layout", ["strided", "reversed", "endian", "readonly"])
def test_real_capture_layouts_are_owned_and_exact(gui, qtbot, speaker, mode, layout):
    window, audio, worker = gui
    window.mode.setCurrentIndex(window.mode.findData(mode))
    start(window, worker, qtbot)
    if mode == "ptt":
        window._press(speaker)
    base = np.linspace(.1, .5, 16640, dtype=np.float64)
    source = base[::2]
    if layout == "reversed":
        source = source[::-1]
    elif layout == "endian":
        source = source.astype(">f8")
    elif layout == "readonly":
        source.flags.writeable = False
    expected = source.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        audio.samples.emit(speaker, source)
    base[:] = np.nan
    if source.flags.writeable:
        source[:] = np.nan
    if mode == "auto":
        audio.samples.emit(speaker, np.zeros(9600, dtype=np.float32))
    else:
        window._release(speaker)
    assert len(worker.submissions) == 1
    np.testing.assert_array_equal(worker.submissions[0][2][:8320], expected)
