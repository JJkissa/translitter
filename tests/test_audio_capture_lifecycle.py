"""Microphone callbacks use fake endpoints; no physical audio is opened."""
import numpy as np
import pytest

from test_audio import explicit_audio, _set_capture_chunk
from translitter.core import make_routes


def test_interleaved_microphones_keep_independent_partial_samples(explicit_audio):
    hub, pairs, _ = explicit_audio
    hub.start(make_routes(*pairs))
    expected = {0: np.array([-32768, 12345], dtype="<i2"),
                1: np.array([32767, -12345], dtype="<i2")}
    received = {0: [], 1: []}
    hub.samples.connect(lambda who, data: received[who].append(data.copy()))
    raw = {who: pcm.tobytes() for who, pcm in expected.items()}
    for index in range(4):
        for who in (0, 1):
            _set_capture_chunk(hub.inputs[who], raw[who][index:index + 1])
            hub.inputs[who].readyRead.emit()
            assert hub.capture_bytes[who] == (index + 1) // 2 * 2
            assert hub.tails[who] == (raw[who][index:index + 1] if index % 2 == 0 else b"")
    for who in (0, 1):
        np.testing.assert_array_equal(np.concatenate(received[who]),
                                      expected[who].astype(np.float32) / 32768)


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("restart", [False, True])
def test_samples_slot_can_retire_capture_without_late_mutation(explicit_audio, speaker, restart):
    hub, pairs, _ = explicit_audio
    routes = make_routes(*pairs)
    hub.start(routes)
    received = []

    def retire(who, data):
        received.append((who, data.copy()))
        hub.samples.disconnect(retire)
        hub.stop()
        if restart:
            hub.start(routes)
            _set_capture_chunk(hub.inputs[speaker], b"\x02\x00\x7f")
            hub._read(speaker)

    hub.samples.connect(retire)
    _set_capture_chunk(hub.inputs[speaker], b"\x01\x00\xff")
    hub._read(speaker)
    assert len(received) == 1 and received[0][0] == speaker
    np.testing.assert_array_equal(received[0][1], np.array([1 / 32768], np.float32))
    assert hub.running is restart
    assert hub.tails[speaker] == (b"\x7f" if restart else b"")
    assert hub.capture_bytes[speaker] == 2
    assert hub.capture_bytes[1 - speaker] == 0


@pytest.mark.parametrize("speaker", [0, 1])
@pytest.mark.parametrize("restart", [False, True])
def test_level_slot_cannot_forward_retired_capture(explicit_audio, speaker, restart):
    hub, pairs, _ = explicit_audio
    routes = make_routes(*pairs)
    hub.start(routes)
    old_io = hub.inputs[speaker]
    received = []
    hub.samples.connect(lambda who, data: received.append((who, data.copy())))
    _set_capture_chunk(old_io, b"\x01\x00\xff")

    def retire(who, value):
        hub.level.disconnect(retire)
        hub.stop()
        if restart:
            hub.start(routes)
            _set_capture_chunk(hub.inputs[speaker], b"\x02\x00\x7f")
            hub._read(speaker)

    hub.level.connect(retire)
    hub._read(speaker)
    if restart:
        assert len(received) == 1
        assert received[0][0] == speaker
        np.testing.assert_array_equal(received[0][1], np.array([2 / 32768], np.float32))
        assert hub.capture_bytes[speaker] == 2
        assert hub.tails[speaker] == b"\x7f"
    else:
        assert received == []
        assert hub.tails == {0: b"", 1: b""}
    assert hub.capture_bytes[1 - speaker] == 0


@pytest.mark.parametrize("speaker", [0, 1])
def test_retired_ready_read_cannot_consume_replacement_input(explicit_audio, speaker):
    hub, pairs, _ = explicit_audio
    routes = make_routes(*pairs)
    hub.start(routes)
    old_io = hub.inputs[speaker]
    hub.stop()
    hub.start(routes)
    current_io = hub.inputs[speaker]
    received, levels = [], []
    hub.samples.connect(lambda who, data: received.append((who, data.copy())))
    hub.level.connect(lambda who, value: levels.append((who, value)))
    _set_capture_chunk(current_io, b"\x02\x00\x7f")
    old_io.readyRead.emit()
    assert current_io.pos() == 0
    assert hub.capture_bytes == [0, 0]
    assert hub.tails == {0: b"", 1: b""}
    assert not received and not levels
    current_io.readyRead.emit()
    assert len(received) == len(levels) == 1
    np.testing.assert_array_equal(received[0][1], np.array([2 / 32768], np.float32))
    assert hub.capture_bytes[speaker] == 2
    assert hub.tails[speaker] == b"\x7f"
