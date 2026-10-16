"""Optional TranslateGemma protocol client; production uses owned private IPC."""
import json
from time import monotonic
from urllib.error import URLError
from urllib.request import Request, HTTPRedirectHandler

from .languages import LANGUAGE_NAMES, translation_steps

BASE_URL = "http://127.0.0.1:1235/v1"
MODEL_ID = "translitter-quality"


class QualityUnavailable(RuntimeError):
    """The explicitly selected local backend could not produce a complete result."""


class _NoRedirects(HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, newurl):
        return None


def _valid_text(value, maximum):
    return (isinstance(value, str) and bool(value.strip()) and len(value) <= maximum
            and "\ufffd" not in value
            and not any(ord(c) < 32 and c not in "\t\n\r" for c in value))


def translation_prompt(text, source, target):
    translation_steps(source, target)
    if not _valid_text(text, 2000) or any(token in text for token in
            ("<start_of_turn>", "<end_of_turn>", "<bos>", "<eos>")):
        raise ValueError("Translation requires valid bounded source text")
    first, second = LANGUAGE_NAMES[source], LANGUAGE_NAMES[target]
    return (f"You are a professional {first} ({source}) to {second} ({target}) translator. "
            f"Your goal is to accurately convey the meaning and nuances of the original {first} text "
            f"while adhering to {second} grammar, vocabulary, and cultural sensitivities.\n"
            f"Produce only the {second} translation, without any additional explanations or commentary. "
            f"Please translate the following {first} text into {second}:\n\n\n{text}")


class LocalTranslator:
    """Single model worker owns this client; partial HTTP output is never spoken."""
    def __init__(self, opener=None, timeout=20):
        if not isinstance(timeout, (int, float)) or not 0 < timeout <= 20:
            raise ValueError("Local translator timeout must be at most 20 seconds")
        if opener is None:
            raise QualityUnavailable("Start an owned QualityRuntime before creating a quality client")
        self._opener = opener
        self.timeout = timeout
        self.last_metrics = {}

    def probe(self):
        try:
            request = Request(BASE_URL + "/models", headers={"Accept": "application/json"})
            with self._opener.open(request, timeout=self.timeout) as response:
                raw = response.read(65537)
            if len(raw) > 65536:
                raise QualityUnavailable("Invalid local model inventory")
            payload = json.loads(raw)
            models = payload.get("data") if isinstance(payload, dict) else None
            if not isinstance(models, list) or not any(isinstance(row, dict) and row.get("id") == MODEL_ID for row in models):
                raise QualityUnavailable("Local quality model is not loaded")
            return True
        except QualityUnavailable:
            raise
        except (OSError, URLError, ValueError, TypeError):
            raise QualityUnavailable("Owned local quality translator is unavailable") from None

    def translate(self, text, source, target, cancelled=lambda: False):
        prompt = translation_prompt(text, source, target)
        if cancelled():
            return None
        body = {"model": MODEL_ID, "messages": [{"role": "user", "content": prompt}],
                "stream": True, "temperature": 0, "max_tokens": 512}
        request = Request(BASE_URL + "/chat/completions", data=json.dumps(body).encode(),
                          headers={"Content-Type": "application/json", "Accept": "text/event-stream"}, method="POST")
        started = monotonic()
        first_token = None
        parts, size, total_bytes, stopped = [], 0, 0, False
        try:
            with self._opener.open(request, timeout=self.timeout) as response:
                while True:
                    if cancelled():
                        return None
                    if monotonic() - started > self.timeout:
                        raise QualityUnavailable("Local quality translation exceeded its time limit")
                    line = response.readline(65537)
                    if not line:
                        break
                    total_bytes += len(line)
                    if len(line) > 65536 or total_bytes > 1048576:
                        raise QualityUnavailable("Local quality translation exceeded its response limit")
                    if not line.startswith(b"data:"):
                        continue
                    raw = line[5:].strip()
                    if raw == b"[DONE]":
                        break
                    payload = json.loads(raw)
                    choices = payload.get("choices") if isinstance(payload, dict) else None
                    if choices == [] and "usage" in payload:
                        continue
                    if not isinstance(payload, dict) or payload.get("model") != MODEL_ID:
                        raise QualityUnavailable("Unexpected local translation model")
                    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
                        raise QualityUnavailable("Malformed local quality response")
                    choice = choices[0]
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        raise QualityUnavailable("Malformed local quality response")
                    piece = delta.get("content")
                    if piece is not None:
                        if not isinstance(piece, str):
                            raise QualityUnavailable("Malformed local quality text")
                        if piece:
                            if first_token is None:
                                first_token = monotonic() - started
                            parts.append(piece)
                            size += len(piece)
                            if size > 8000:
                                raise QualityUnavailable("Local quality output is too long")
                    finish = choice.get("finish_reason")
                    if finish is not None:
                        if finish != "stop":
                            raise QualityUnavailable("Local quality translation was truncated")
                        stopped = True
                        break
            if cancelled():
                return None
            if monotonic() - started > self.timeout:
                raise QualityUnavailable("Local quality translation exceeded its time limit")
            result = "".join(parts).strip()
            if not stopped or not _valid_text(result, 8000):
                raise QualityUnavailable("Local quality translator produced no valid complete translation")
            self.last_metrics = {"seconds": monotonic() - started, "first_token_seconds": first_token}
            return result
        except QualityUnavailable:
            raise
        except (OSError, URLError, ValueError, TypeError):
            raise QualityUnavailable("Local quality translation failed; check the local runtime") from None
