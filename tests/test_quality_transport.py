"""Exercise the production urllib boundary without TCP or model processes."""
import io
import os
import socket
import struct
from types import SimpleNamespace
from urllib.request import ProxyHandler, build_opener

import pytest

from translitter.quality import LocalTranslator, QualityUnavailable, _NoRedirects
from test_quality_runtime import small_model, stub_runtime


@pytest.fixture
def endpoint(monkeypatch):
    from translitter import quality_transport as transport
    state = SimpleNamespace(pid=4321, uid=os.getuid(), returncode=None, sockets=[],
                            die_on_connect=False, paths=[], polls=0)

    class Socket:
        def __init__(self):
            self.sent = []
            self.closed = False
            self.peer = (state.pid, state.uid, os.getgid())
            state.sockets.append(self)

        def settimeout(self, timeout):
            assert 0 < timeout <= 20

        def connect(self, path):
            state.paths.append(path)
            if state.die_on_connect:
                state.returncode = 1

        def getsockopt(self, level, option, size):
            assert (level, option, size) == (socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i"))
            return struct.pack("3i", *self.peer)

        def sendall(self, data):
            self.sent.append(bytes(data))

        def makefile(self, *args):
            body = b'data: {"model":"translitter-quality","choices":[{"delta":{"content":"Hei"},"finish_reason":"stop"}]}\n\n'
            if b"GET /v1/models " in b"".join(self.sent):
                body = b'{"data":[{"id":"translitter-quality"}]}'
            return io.BytesIO(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)

        def close(self):
            self.closed = True

    def make_socket(family, kind):
        assert (family, kind) == (socket.AF_UNIX, socket.SOCK_STREAM), "TCP must never be used"
        return Socket()

    def poll():
        state.polls += 1
        return state.returncode

    state.process = SimpleNamespace(pid=4321, poll=poll)
    monkeypatch.setattr(transport, "socket", SimpleNamespace(socket=make_socket,
        AF_UNIX=socket.AF_UNIX, SOCK_STREAM=socket.SOCK_STREAM,
        SOL_SOCKET=socket.SOL_SOCKET, SO_PEERCRED=socket.SO_PEERCRED))
    monkeypatch.setattr(transport, "_DeadlineSocket", lambda family, kind, timeout: make_socket(family, kind))
    state.opener = build_opener(ProxyHandler({}), _NoRedirects(),
                               transport.OwnedUnixHTTPHandler("/private/server.sock", state.process))
    state.client = LocalTranslator(opener=state.opener)
    return state


def test_managed_readiness_and_translation_use_private_ipc_never_tcp(stub_runtime, endpoint, monkeypatch):
    from translitter import quality_runtime as qr
    monkeypatch.setattr(qr, "LocalTranslator", LocalTranslator)
    endpoint.pid = stub_runtime.process.pid
    client = stub_runtime.runtime.start()
    assert client.probe()
    assert client.translate("PRIVATE SOURCE", "en", "fi") == "Hei"
    assert len(endpoint.sockets) == 3
    assert all(path.endswith("/api.sock") for path in endpoint.paths)
    stub_runtime.returncode = 1
    with pytest.raises(QualityUnavailable):
        client.translate("AFTER DEATH", "en", "fi")
    assert len(endpoint.sockets) == 3


def test_startup_path_race_never_sends_inventory_or_text_to_wrong_peer(stub_runtime, endpoint, monkeypatch):
    from translitter import quality_runtime as qr
    monkeypatch.setattr(qr, "LocalTranslator", LocalTranslator)
    endpoint.pid = stub_runtime.process.pid + 1
    with pytest.raises(qr.RuntimeUnavailable, match="readiness timed out"):
        stub_runtime.runtime.start()
    assert endpoint.sockets
    assert all(connection.sent == [] and connection.closed for connection in endpoint.sockets)
    assert stub_runtime.runtime._process is None
    assert stub_runtime.runtime._socket_directory is None


@pytest.mark.parametrize("when", ["before", "during"])
def test_dead_owned_child_receives_zero_http_bytes(endpoint, when):
    endpoint.returncode = 1 if when == "before" else None
    endpoint.die_on_connect = when == "during"
    with pytest.raises(QualityUnavailable):
        endpoint.client.translate("PRIVATE SOURCE", "en", "fi")
    assert all(connection.sent == [] and connection.closed for connection in endpoint.sockets)
    assert len(endpoint.sockets) == int(when == "during")


def test_healthy_owned_peer_permits_translation_then_replacement_gets_no_bytes(endpoint):
    assert endpoint.client.translate("PRIVATE SOURCE", "en", "fi") == "Hei"
    assert b"PRIVATE SOURCE" in b"".join(endpoint.sockets[0].sent)
    assert endpoint.paths == ["/private/server.sock"]
    endpoint.pid += 1  # Same private pathname, different accepting process.
    with pytest.raises(QualityUnavailable):
        endpoint.client.translate("SECOND PRIVATE SOURCE", "en", "fi")
    assert endpoint.sockets[-1].sent == []
    assert endpoint.sockets[-1].closed


@pytest.mark.parametrize("peer", ["pid", "uid"])
def test_wrong_peer_receives_zero_http_bytes(endpoint, peer):
    setattr(endpoint, peer, getattr(endpoint, peer) + 1)
    with pytest.raises(QualityUnavailable):
        endpoint.client.translate("PRIVATE SOURCE", "en", "fi")
    assert len(endpoint.sockets) == 1
    assert endpoint.sockets[0].sent == []
    assert endpoint.sockets[0].closed
