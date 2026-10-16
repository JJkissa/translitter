"""Loopback-only quality translator protocol; no external network/model loads."""
import io
import json

import pytest


class Reply(io.BytesIO):
    def __enter__(self): return self
    def __exit__(self, *args): self.close()


class Opener:
    def __init__(self, payloads):
        self.payloads = iter(payloads)
        self.requests = []
    def open(self, request, timeout):
        self.requests.append((request, timeout))
        payload = next(self.payloads)
        if isinstance(payload, Exception):
            raise payload
        return Reply(payload)


def events(parts, finish="stop", model="translitter-quality"):
    chunks = ["data: " + json.dumps({"model": model, "choices": [{"delta": {"content": part}, "finish_reason": None}]}) + "\n\n" for part in parts]
    chunks.append("data: " + json.dumps({"model": model, "choices": [{"delta": {}, "finish_reason": finish}]}) + "\n\n")
    return ("".join(chunks) + "data: [DONE]\n\n").encode()


def test_default_client_cannot_adopt_an_unowned_tcp_endpoint(monkeypatch):
    from translitter import quality
    requests = []
    monkeypatch.setattr(quality, "build_opener", lambda *a: requests.append(a), raising=False)
    with pytest.raises(quality.QualityUnavailable, match="owned"):
        quality.LocalTranslator()
    assert requests == []


def test_quality_client_uses_fixed_loopback_and_single_translation_prompt():
    from translitter.quality import LocalTranslator
    opener = Opener([events(["Tänään ", "haluan ostaa leipää."])])
    client = LocalTranslator(opener=opener)
    result = client.translate("Сегодня я хочу купить хлеб.", "ru", "fi")
    assert result == "Tänään haluan ostaa leipää."
    request, timeout = opener.requests[0]
    assert request.full_url == "http://127.0.0.1:1235/v1/chat/completions"
    assert 0 < timeout <= 20
    body = json.loads(request.data)
    assert body["model"] == "translitter-quality" and body["stream"] is True
    assert body["temperature"] == 0 and body["max_tokens"] == 512
    assert len(body["messages"]) == 1
    message = body["messages"][0]
    assert message["role"] == "user"
    assert "Russian (ru) to Finnish (fi)" in message["content"]
    assert message["content"].endswith("\n\n\nСегодня я хочу купить хлеб.")


@pytest.mark.parametrize("source,target,text", [("xx", "fi", "x"), ("fi", "fi", "x"), ("ru", "fi", ""), ("ru", "fi", "x" * 2001), ("ru", "fi", "a\x00b")])
def test_quality_invalid_inputs_never_reach_endpoint(source, target, text):
    from translitter.quality import LocalTranslator
    opener = Opener([])
    with pytest.raises(ValueError):
        LocalTranslator(opener=opener).translate(text, source, target)
    assert not opener.requests


@pytest.mark.parametrize("reply", [events(["bad"], "length"), events([]), b"data: not-json\n\n", events(["bad\ufffd"]), b"data: [DONE]\n\n"])
def test_bad_or_truncated_outputs_are_not_spoken(reply):
    from translitter.quality import LocalTranslator, QualityUnavailable
    with pytest.raises(QualityUnavailable):
        LocalTranslator(opener=Opener([reply])).translate("hello", "en", "fi")


def test_cancellation_before_request_does_not_generate():
    from translitter.quality import LocalTranslator
    opener = Opener([])
    assert LocalTranslator(opener=opener).translate("hello", "en", "fi", lambda: True) is None
    assert not opener.requests


def test_cancellation_during_stream_discards_partial_translation():
    from translitter.quality import LocalTranslator
    opener = Opener([events(["Hei", " siellä"])])
    checks = []
    def cancelled():
        checks.append(True)
        return len(checks) >= 4
    assert LocalTranslator(opener=opener).translate("hello", "en", "fi", cancelled) is None


@pytest.mark.parametrize("model", [None, "unrelated-user-model"])
def test_response_requires_exact_model_not_only_inventory(model):
    from translitter.quality import LocalTranslator, QualityUnavailable
    with pytest.raises(QualityUnavailable):
        LocalTranslator(opener=Opener([events(["Hei"], model=model)])).translate("Hello", "en", "fi")


def test_probe_requires_exact_loaded_model():
    from translitter.quality import LocalTranslator, QualityUnavailable
    good = Opener([json.dumps({"data": [{"id": "translitter-quality"}]}).encode()])
    assert LocalTranslator(opener=good).probe() is True
    assert good.requests[0][0].full_url == "http://127.0.0.1:1235/v1/models"
    bad = Opener([json.dumps({"data": [{"id": "unrelated-user-model"}]}).encode()])
    with pytest.raises(QualityUnavailable):
        LocalTranslator(opener=bad).probe()


def test_connection_failure_does_not_leak_source_text_in_error():
    from translitter.quality import LocalTranslator, QualityUnavailable
    opener = Opener([OSError("untrusted error payload")])
    with pytest.raises(QualityUnavailable) as caught:
        LocalTranslator(opener=opener).translate("private conversation", "en", "fi")
    assert "private conversation" not in str(caught.value)
    assert "untrusted error payload" not in str(caught.value)
