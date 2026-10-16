"""HTTP over private Unix IPC, authenticated before any HTTP bytes leave."""
from http.client import HTTPConnection
import os
import socket
import struct
from time import monotonic
from urllib.request import HTTPHandler


class _DeadlineSocket(socket.socket):
    """Bound every low-level read, even inside a buffered HTTP readline."""
    def __init__(self, *args, timeout, **kwargs):
        super().__init__(*args, **kwargs)
        self._deadline = monotonic() + timeout
        self.settimeout(timeout)

    def _remaining(self):
        remaining = self._deadline - monotonic()
        if remaining <= 0:
            raise TimeoutError("Local quality response deadline exceeded")
        self.settimeout(remaining)

    def connect(self, address):
        self._remaining()
        return super().connect(address)

    def recv_into(self, buffer, nbytes=0, flags=0):
        self._remaining()
        return super().recv_into(buffer, nbytes, flags)

    def recv(self, size, flags=0):
        self._remaining()
        return super().recv(size, flags)

    def sendall(self, data, flags=0):
        self._remaining()
        return super().sendall(data, flags)


class _OwnedUnixConnection(HTTPConnection):
    def __init__(self, host, *, socket_path, process, owner_uid, **kwargs):
        super().__init__(host, **kwargs)
        self._socket_path = socket_path
        self._process = process
        self._owner_uid = owner_uid

    def connect(self):
        if self._process.poll() is not None:
            raise OSError("Owned quality runtime has exited")
        connection = _DeadlineSocket(socket.AF_UNIX, socket.SOCK_STREAM, timeout=self.timeout)
        try:
            connection.settimeout(self.timeout)
            connection.connect(self._socket_path)
            pid, uid, _ = struct.unpack("3i", connection.getsockopt(
                socket.SOL_SOCKET, socket.SO_PEERCRED, struct.calcsize("3i")))
            if self._process.poll() is not None:
                raise OSError("Owned quality runtime has exited")
            if (pid, uid) != (self._process.pid, self._owner_uid):
                raise OSError("Quality endpoint identity mismatch")
        except BaseException:
            connection.close()
            raise
        self.sock = connection


class OwnedUnixHTTPHandler(HTTPHandler):
    """urllib transport that never resolves or connects to the logical TCP host."""
    def __init__(self, socket_path, process):
        super().__init__()
        self._socket_path = str(socket_path)
        self._process = process
        self._owner_uid = os.getuid()

    def http_open(self, request):
        def connection(host, **kwargs):
            return _OwnedUnixConnection(host, socket_path=self._socket_path,
                process=self._process, owner_uid=self._owner_uid, **kwargs)
        return self.do_open(connection, request)
