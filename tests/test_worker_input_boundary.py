"""Worker submission numeric validation and caller-buffer ownership."""
import warnings

import numpy as np
import pytest

from test_worker import ROUTES, harness


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("invalid", [
    np.array([.25], dtype=object), np.array([".25"]),
    np.array([b".25"]), np.array(["2026-10-18"], dtype="datetime64[D]"),
    np.array([1], dtype="timedelta64[s]"),
    np.array([(.25,)], dtype=[("sample", "f4")]),
    np.array([np.nan]), np.array([np.inf]), np.array([-.0, -np.inf]),
])
def test_invalid_numeric_domains_do_not_consume_queue_capacity(harness, speaker, invalid):
    h = harness
    h.worker.set_session(1, ROUTES)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not h.worker.submit(1, speaker, invalid)
    assert all(not queue for queue in h.worker._pending)
    assert h.worker.submit(1, speaker, [.125, -.25])
    h.worker.start()
    h.qtbot.waitUntil(lambda: len(h.results) == 1)
    np.testing.assert_array_equal(h.state.turns[0][2], [.125, -.25])
    assert not h.errors


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("layout", ["strided", "reversed", "endian", "readonly", "integer", "bool"])
def test_submission_owns_exact_real_samples_before_producer_reuse(harness, speaker, layout):
    h = harness
    h.worker.set_session(1, ROUTES)
    base = np.linspace(-.75, .75, 18, dtype=np.float64)
    source = base[::2] if layout == "strided" else base[::-1]
    if layout == "endian":
        source = base.astype(">f8")
    elif layout == "readonly":
        source = base.view()
        source.flags.writeable = False
    elif layout == "integer":
        source = np.arange(-8, 8, dtype=np.int16)
    elif layout == "bool":
        source = np.array([True, False, True])
    expected = source.astype(np.float32)
    assert h.worker.submit(1, speaker, source)
    queued = h.worker._pending[speaker][0][2]
    assert queued.dtype == np.float32
    assert not np.shares_memory(queued, source)
    if layout == "readonly":
        base[:] = np.nan
    else:
        source[:] = 0
    # A second accepted turn must also survive producer reuse.
    next_audio = np.array([.375, -.625], dtype=np.float32)
    assert h.worker.submit(1, speaker, next_audio)
    next_audio[:] = np.nan
    h.worker.start()
    h.qtbot.waitUntil(lambda: len(h.results) == 2)
    np.testing.assert_array_equal(h.state.turns[0][2], expected)
    np.testing.assert_array_equal(h.state.turns[1][2], [.375, -.625])
    assert not h.errors



@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("dtype", [np.complex64, np.complex128])
@pytest.mark.parametrize("imaginary", [0, .25])
def test_complex_input_rejected_without_warning_or_queue_displacement(
        harness, speaker, dtype, imaginary):
    h = harness
    h.worker.set_session(1, ROUTES)
    first = np.array([.125, -.25], dtype=np.float32)
    second = np.array([.5, -.75], dtype=np.float32)
    assert h.worker.submit(1, speaker, first)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not h.worker.submit(1, speaker, np.array([.5 + imaginary * 1j], dtype=dtype))
    assert h.worker.submit(1, speaker, second)
    h.worker.start()
    h.qtbot.waitUntil(lambda: len(h.results) == 2)
    assert [turn.speaker for turn in h.results] == [speaker, speaker]
    for actual, expected in zip(h.state.turns, (first, second)):
        np.testing.assert_array_equal(actual[2], expected)
    assert not h.errors


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("numpy_errors", ["warn", "raise"])
def test_float32_overflow_is_boolean_rejection_with_queue_recovery(
        harness, speaker, sign, numpy_errors):
    h = harness
    h.worker.set_session(1, ROUTES)
    first = np.array([.125, -.25], dtype=np.float32)
    second = np.array([.5, -.75], dtype=np.float32)
    assert h.worker.submit(1, speaker, first)
    with warnings.catch_warnings(), np.errstate(over=numpy_errors, invalid=numpy_errors):
        warnings.simplefilter("error")
        assert not h.worker.submit(1, speaker, np.array([sign * np.finfo(np.float64).max]))
    assert h.worker.submit(1, speaker, second)
    h.worker.start()
    h.qtbot.waitUntil(lambda: len(h.results) == 2)
    for actual, expected in zip(h.state.turns, (first, second)):
        np.testing.assert_array_equal(actual[2], expected)
    assert not h.errors
