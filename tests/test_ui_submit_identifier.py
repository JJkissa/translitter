"""Public demo submission uses the worker's strict participant-ID contract."""
import warnings

import numpy as np
import pytest

from test_ui import gui, start
from test_ui_capture_numeric import retained


INVALID_IDS = [
    True, False, 0.0, 1.0, np.int64(0), np.int32(1),
    np.bool_(True), np.float64(0), 0j, 1 + 0j,
    None, "0", -1, 2, [], {}, np.array([0, 1]), np.array(0),
]


@pytest.mark.parametrize("held_speaker", [0, 1])
@pytest.mark.parametrize("identifier", INVALID_IDS)
def test_invalid_identifier_is_rejected_before_dispatch_or_state_change(
        gui, qtbot, held_speaker, identifier):
    window, audio, worker = gui
    start(window, worker, qtbot)
    turn = np.full(8000, .25, dtype=np.float32)
    assert window.submit_audio(1 - held_speaker, turn)
    window._press(held_speaker)
    prefix = np.full(4161, .125, dtype=np.float32)
    window._samples(held_speaker, prefix)
    before = [retained(window, n) for n in (0, 1)]
    status = window.status.text()
    count = len(worker.submissions)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert window.submit_audio(identifier, turn) is False
        # Identifier admission also precedes malformed sample conversion.
        assert window.submit_audio(identifier, [[.1], [.2, .3]]) is False
    assert len(worker.submissions) == count
    assert [retained(window, n) for n in (0, 1)] == before
    assert window.status.text() == status
    assert window.submit_audio(held_speaker, turn) is True
    assert worker.submissions[-1][:2] == (window.generation, held_speaker)
    np.testing.assert_array_equal(worker.submissions[-1][2], turn)
    window._samples(held_speaker, np.full(3839, .375, dtype=np.float32))
    window._release(held_speaker)
    np.testing.assert_array_equal(
        worker.submissions[-1][2],
        np.concatenate((prefix, np.full(3839, .375, dtype=np.float32))),
    )
    assert window._busy[held_speaker] == 2
    assert window._busy[1 - held_speaker] == 1


@pytest.mark.parametrize("identifier", [0, 1, *INVALID_IDS])
def test_stopped_submission_is_inert_for_all_identifiers(gui, identifier):
    window, audio, worker = gui
    before = [retained(window, n) for n in (0, 1)]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert window.submit_audio(identifier, np.full(8000, .25, dtype=np.float32)) is False
    assert not worker.submissions
    assert [retained(window, n) for n in (0, 1)] == before


@pytest.mark.parametrize("speaker", [0, 1])
def test_missing_worker_rejects_valid_identifier_without_state_change(gui, qtbot, speaker):
    window, audio, worker = gui
    start(window, worker, qtbot)
    window.worker = None
    before = [retained(window, n) for n in (0, 1)]
    try:
        assert window.submit_audio(speaker, np.full(8000, .25, dtype=np.float32)) is False
        assert [retained(window, n) for n in (0, 1)] == before
        assert not worker.submissions
    finally:
        window.worker = worker
