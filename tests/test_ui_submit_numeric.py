"""Explicit demo turns reject invalid numeric input before worker dispatch."""
import warnings

import numpy as np
import pytest

from test_ui import gui, start
from test_ui_capture_numeric import retained
from translitter.core import MAX_TURN_SAMPLES


INVALID = [
    np.full(8000, .3 + .2j, dtype=np.complex64),
    np.full(8000, .3 + 0j, dtype=np.complex128),
    np.full(8000, np.finfo(np.float64).max),
    np.full(8000, np.nan), np.full(8000, np.inf),
    np.ones((2, 8000)), np.array(.3),
    np.array(["not PCM"] * 8000), [[.1], [.2, .3]],
    np.full(8000, ".3"),
]


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("policy", ["warn", "raise"])
@pytest.mark.parametrize("invalid", INVALID)
def test_explicit_invalid_turn_does_not_dispatch_or_consume_busy(gui, qtbot, speaker, policy, invalid):
    window, audio, worker = gui
    start(window, worker, qtbot)
    assert window.submit_audio(1 - speaker, np.full(8000, .1, dtype=np.float32))
    window._press(speaker)
    window._samples(speaker, np.full(4161, .125, dtype=np.float32))
    before = [retained(window, n) for n in (0, 1)]
    before_busy = tuple(window._busy)
    count = len(worker.submissions)
    with warnings.catch_warnings(), np.errstate(over=policy, invalid=policy):
        warnings.simplefilter("error")
        assert window.submit_audio(speaker, invalid) is False
    assert len(worker.submissions) == count
    assert tuple(window._busy) == before_busy
    # Rejection stays visible in shared status without hiding held capture.
    after = [retained(window, n) for n in (0, 1)]
    assert after == before
    assert "invalid" in window.status.text().lower()
    valid = np.linspace(.1, .3, 8000, dtype=np.float32)
    assert window.submit_audio(speaker, valid) is True
    assert len(worker.submissions) == count + 1
    assert worker.submissions[-1][:2] == (window.generation, speaker)
    np.testing.assert_array_equal(worker.submissions[-1][2], valid)
    assert window._busy[speaker] == before_busy[speaker] + 1


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("layout", ["strided", "reversed", "endian", "readonly", "list"])
def test_explicit_real_layouts_reach_worker_exactly(gui, qtbot, speaker, layout):
    window, audio, worker = gui
    start(window, worker, qtbot)
    base = np.linspace(.1, .5, 16000, dtype=np.float64)
    source = base[::2]
    if layout == "reversed":
        source = source[::-1]
    elif layout == "endian":
        source = source.astype(">f8")
    elif layout == "readonly":
        source.flags.writeable = False
    elif layout == "list":
        source = source.tolist()
    expected = np.asarray(source, dtype=np.float32).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert window.submit_audio(speaker, source) is True
    base[:] = np.nan
    np.testing.assert_array_equal(worker.submissions[-1][2], expected)
    assert window._busy[speaker] == 1


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("length,accepted", [(3999, False), (4000, True), (MAX_TURN_SAMPLES, True), (MAX_TURN_SAMPLES + 1, False)])
def test_explicit_length_boundaries(gui, qtbot, speaker, length, accepted):
    window, audio, worker = gui
    start(window, worker, qtbot)
    assert window.submit_audio(speaker, np.full(length, .2, dtype=np.float32)) is accepted
    assert len(worker.submissions) == int(accepted)
    assert window._busy[speaker] == int(accepted)
    if accepted:
        assert len(worker.submissions[0][2]) == length
