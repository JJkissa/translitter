"""Hard response deadline, including slow byte trickles through real Unix IPC."""
import io
import json
import os
from pathlib import Path
import socket
import tempfile
from threading import Event, Thread
from types import SimpleNamespace
from urllib.request import build_opener, ProxyHandler

import pytest

from translitter.quality import LocalTranslator, QualityUnavailable, _NoRedirects
from translitter.quality_transport import OwnedUnixHTTPHandler


def test_trickling_response_cannot_extend_total_deadline():
    done = Event()
    sent = []
    payload = b"data: incomplete-event-without-newline"
    with tempfile.TemporaryDirectory(prefix="qd-", dir=os.environ["TMPDIR"]) as directory:
        path = str(Path(directory) / "api.sock")
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
            server.bind(path)
            server.listen(1)
            server.settimeout(1)
            def serve():
                try:
                    peer, _ = server.accept()
                    with peer:
                        peer.settimeout(1)
                        request = b""
                        while b"\r\n\r\n" not in request:
                            request += peer.recv(4096)
                        header, body = request.split(b"\r\n\r\n", 1)
                        length = next(int(line.split(b":", 1)[1]) for line in header.split(b"\r\n")
                                      if line.lower().startswith(b"content-length:"))
                        while len(body) < length:
                            body += peer.recv(4096)
                        peer.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nConnection: close\r\n\r\n")
                        for byte in payload:
                            if done.wait(.02):
                                break
                            peer.sendall(bytes([byte]))
                            sent.append(byte)
                except (OSError, TimeoutError):
                    pass  # Expected when the deadline closes the receiving socket.
            thread = Thread(target=serve, daemon=True)
            thread.start()
            process = SimpleNamespace(pid=os.getpid(), poll=lambda: None)
            opener = build_opener(ProxyHandler({}), _NoRedirects(), OwnedUnixHTTPHandler(path, process))
            try:
                with pytest.raises(QualityUnavailable):
                    LocalTranslator(opener=opener, timeout=.08).translate("synthetic source", "en", "fi")
                assert len(sent) < len(payload), "readline consumed an entire slow trickle beyond the total deadline"
            finally:
                done.set()
                thread.join(2)
            assert not thread.is_alive()


def test_buffered_finish_after_deadline_is_not_returned(monkeypatch):
    from translitter import quality
    clock = [0.0]
    line = b"data: " + json.dumps({"model": "translitter-quality", "choices": [
        {"delta": {"content": "Hei"}, "finish_reason": "stop"}]}).encode() + b"\n\n"
    class Reply(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *args): self.close()
        def readline(self, size):
            result = super().readline(size)
            clock[0] = 21.0
            return result
    opener = SimpleNamespace(open=lambda *args, **kwargs: Reply(line))
    monkeypatch.setattr(quality, "monotonic", lambda: clock[0])
    with pytest.raises(QualityUnavailable, match="time limit"):
        LocalTranslator(opener=opener).translate("synthetic source", "en", "fi")
