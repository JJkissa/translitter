"""Bounded response failures stay sanitized and cannot poison the next request."""
import io
import json
from http.client import HTTPResponse
from types import SimpleNamespace

import pytest

from translitter.quality import LocalTranslator, MODEL_ID, QualityUnavailable
from test_quality import Reply, events
from test_engine import backends
from translitter.core import Route
from translitter.engine import ModelEngine


def event(payload):
    return b"data: " + json.dumps(payload).encode() + b"\n\n"


INVALID_STREAMS = [
    pytest.param(b'data: {"private":\n\n', id="malformed-json"),
    pytest.param(b'data: {"private":"\xff"}\n\n', id="invalid-utf8"),
    pytest.param(event([]), id="wrong-top-level"),
    pytest.param(event({"model": MODEL_ID, "choices": {}}), id="wrong-choices"),
    pytest.param(event({"model": MODEL_ID, "choices": [{"delta": []}]}), id="wrong-delta"),
    pytest.param(event({"model": MODEL_ID, "choices": [{"delta": {"content": 42}}]}), id="nontext-content"),
    pytest.param(events(["partial"], finish="length"), id="token-truncation"),
    pytest.param(events(["partial"], finish=None), id="no-finish"),
    pytest.param(events(["private\x00text"]), id="control-character"),
    pytest.param(events(["x" * 8001]), id="output-limit"),
    pytest.param(b"data: " + b"x" * 65536 + b"\n", id="line-limit"),
    pytest.param((b":" + b"x" * 65534 + b"\n") * 17, id="total-response-limit"),
]


class ResponseOpener:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    def open(self, request, timeout):
        self.requests.append(request)
        return next(self.responses)


def truncated_chunked_response():
    # Use the real HTTP parser: an advertised chunk ends before its body does.
    wire = (b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
            b"100\r\nprivate-response-fragment")
    response = HTTPResponse(SimpleNamespace(makefile=lambda *args: io.BufferedReader(io.BytesIO(wire))))
    response.begin()
    return response


@pytest.mark.parametrize("operation", ["probe", "translate"])
def test_truncated_http_body_is_sanitized_closed_and_next_request_recovers(operation):
    broken = truncated_chunked_response()
    recovery = Reply(events(["Hei"]))
    opener = ResponseOpener([broken, recovery])
    client = LocalTranslator(opener=opener)

    with pytest.raises(QualityUnavailable) as caught:
        if operation == "probe":
            client.probe()
        else:
            client.translate("private source text", "en", "fi")

    assert "private" not in str(caught.value)
    assert "fragment" not in str(caught.value)
    assert broken.isclosed()
    assert client.last_metrics == {}
    assert client.translate("hello", "en", "fi") == "Hei"
    assert recovery.closed
    assert len(opener.requests) == 2


@pytest.mark.parametrize("payload", INVALID_STREAMS)
def test_rejected_response_never_reaches_voice_and_next_turn_recovers(backends, payload):
    root, calls, _ = backends
    engine = ModelEngine(root)
    engine.load()
    broken, recovery = Reply(payload), Reply(events(["Hei"]))
    engine._quality_client = LocalTranslator(opener=ResponseOpener([broken, recovery]))
    engine.translation_backend = "translategemma"
    calls.clear()
    route = Route(0, 1, "en", "fi", "mic", "out")
    try:
        with pytest.raises(QualityUnavailable) as caught:
            engine.interpret_text("private source text", route, 1)
        assert "private" not in str(caught.value)
        assert broken.closed
        assert not any(call[0] == "tts" for call in calls)
        turn = engine.interpret_text("hello", route, 1)
        assert turn.translation == "Hei"
        assert len([call for call in calls if call[0] == "tts"]) == 1
        assert recovery.closed
    finally:
        engine.close()


@pytest.mark.parametrize("during_finish", [False, True])
def test_cancellation_inside_response_read_discards_output_and_recovers(during_finish):
    cancelled = [False]

    class CancellingReply(Reply):
        def readline(self, size=-1):
            line = super().readline(size)
            marker = b'"finish_reason": "stop"' if during_finish else b'"content"'
            if marker in line:
                cancelled[0] = True
            return line

    broken = CancellingReply(events(["partial private translation"]))
    recovery = Reply(events(["Hei"]))
    client = LocalTranslator(opener=ResponseOpener([broken, recovery]))
    assert client.translate("hello", "en", "fi", lambda: cancelled[0]) is None
    assert broken.closed and client.last_metrics == {}
    cancelled[0] = False
    assert client.translate("hello", "en", "fi", lambda: cancelled[0]) == "Hei"
    assert recovery.closed
