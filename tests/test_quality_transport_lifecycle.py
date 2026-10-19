"""Owned IPC connection failure isolation, retry and real Linux peer checks."""
import os
from pathlib import Path
import socket
import tempfile
import threading
from types import SimpleNamespace

import pytest

from translitter import quality_transport as transport
from translitter.quality import LocalTranslator, QualityUnavailable, _NoRedirects
from test_quality_transport import endpoint
from urllib.request import ProxyHandler, build_opener


@pytest.mark.parametrize("operation", ["probe", "translate"])
@pytest.mark.parametrize("boundary", ["connect", "credentials", "second_poll"])
@pytest.mark.parametrize("failure", [FileNotFoundError, PermissionError, OSError])
def test_connection_failure_closes_without_publication_and_recovers(
        endpoint, monkeypatch, operation, boundary, failure):
    factory = transport._DeadlineSocket
    private = "/private/DO-NOT-DISCLOSE-source.sock"

    def make_socket(*args, **kwargs):
        connection = factory(*args, **kwargs)
        if boundary != "second_poll":
            method = "connect" if boundary == "connect" else "getsockopt"

            def fail(*args):
                raise failure(private)

            monkeypatch.setattr(connection, method, fail)
        return connection

    monkeypatch.setattr(transport, "_DeadlineSocket", make_socket)
    poll = endpoint.process.poll
    if boundary == "second_poll":
        def fail_poll():
            if endpoint.polls:
                raise failure(private)
            return poll()
        monkeypatch.setattr(endpoint.process, "poll", fail_poll)

    def invoke():
        if operation == "probe":
            return endpoint.client.probe()
        return endpoint.client.translate("PRIVATE SOURCE", "en", "fi")

    with pytest.raises(QualityUnavailable) as error:
        invoke()
    assert private not in str(error.value)
    assert "PRIVATE SOURCE" not in str(error.value)
    assert len(endpoint.sockets) == 1
    assert endpoint.sockets[0].closed
    assert endpoint.sockets[0].sent == []

    monkeypatch.setattr(transport, "_DeadlineSocket", factory)
    monkeypatch.setattr(endpoint.process, "poll", poll)
    assert invoke() == (True if operation == "probe" else "Hei")
    assert len(endpoint.sockets) == 2
    assert endpoint.sockets[-1].closed
    assert endpoint.sockets[-1].sent


@pytest.mark.parametrize("operation", ["probe", "translate"])
def test_child_dying_during_credential_lookup_cannot_receive_http(endpoint, monkeypatch, operation):
    factory = transport._DeadlineSocket

    def make_socket(*args, **kwargs):
        connection = factory(*args, **kwargs)
        lookup = connection.getsockopt

        def die(*args):
            credentials = lookup(*args)
            endpoint.returncode = 7
            return credentials

        monkeypatch.setattr(connection, "getsockopt", die)
        return connection

    monkeypatch.setattr(transport, "_DeadlineSocket", make_socket)
    with pytest.raises(QualityUnavailable):
        if operation == "probe":
            endpoint.client.probe()
        else:
            endpoint.client.translate("PRIVATE SOURCE", "en", "fi")
    assert len(endpoint.sockets) == 1
    assert endpoint.sockets[0].closed
    assert endpoint.sockets[0].sent == []
    monkeypatch.setattr(transport, "_DeadlineSocket", factory)
    endpoint.returncode = None
    assert endpoint.client.translate("RECOVERED", "en", "fi") == "Hei"


@pytest.mark.parametrize("identity", ["match", "pid", "uid"])
def test_real_unix_peer_authentication_precedes_http_publication(identity):
    """A local thread is the peer: kernel credentials must name this process."""
    received, failures = [], []
    with tempfile.TemporaryDirectory(prefix="tqi-", dir=os.environ["TMPDIR"]) as directory:
        path = str(Path(directory) / "api.sock")
        server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server.settimeout(3)
        server.bind(path)
        server.listen(1)

        def serve():
            try:
                connection, _ = server.accept()
                with connection:
                    connection.settimeout(3)
                    request = bytearray()
                    while b"\r\n\r\n" not in request:
                        chunk = connection.recv(4096)
                        if not chunk:
                            break
                        request.extend(chunk)
                    received.append(bytes(request))
                    if request:
                        body = b'{"data":[{"id":"translitter-quality"}]}'
                        connection.sendall(b"HTTP/1.1 200 OK\r\nConnection: close\r\nContent-Length: "
                                           + str(len(body)).encode() + b"\r\n\r\n" + body)
            except BaseException as error:
                failures.append(error)

        process = SimpleNamespace(pid=os.getpid() + int(identity == "pid"), poll=lambda: None)
        handler = transport.OwnedUnixHTTPHandler(path, process)
        if identity == "uid":
            handler._owner_uid += 1
        client = LocalTranslator(opener=build_opener(ProxyHandler({}), _NoRedirects(), handler), timeout=2)
        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        try:
            if identity == "match":
                assert client.probe() is True
            else:
                with pytest.raises(QualityUnavailable):
                    client.probe()
        finally:
            thread.join(timeout=4)
            server.close()
        assert not thread.is_alive()
        assert failures == []
        assert len(received) == 1
        if identity == "match":
            assert received[0].startswith(b"GET /v1/models HTTP/1.1\r\n")
        else:
            assert received == [b""]
