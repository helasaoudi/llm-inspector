"""
Unix domain socket transport for embedded inspector RPC.

Protocol: newline-delimited JSON (one request → one response per connection).
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

from llm_inspector.rpc import SCHEMA_VERSION, CollectorRequest, CollectorResponse
from llm_inspector.transport.base import EmbeddedHandler, Transport
from llm_inspector.transport.discovery import SOCKET_ENV_VAR, default_socket_path

if TYPE_CHECKING:
    pass

_log = logging.getLogger(__name__)

_DEFAULT_TIMEOUT = 2.0


class UnixSocketTransport(Transport):
    """Unix socket transport — local-only, no network exposure."""

    def __init__(self, pid: int | None = None, path: Path | None = None) -> None:
        self._pid = pid or os.getpid()
        self._path = path or default_socket_path(self._pid)
        self._handlers: dict[str, EmbeddedHandler] = {}
        self._server_sock: socket.socket | None = None
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()

    @property
    def path(self) -> Path:
        return self._path

    def serve(self, handlers: dict[str, EmbeddedHandler]) -> None:
        if self._thread and self._thread.is_alive():
            return

        self._handlers = handlers
        self._path.parent.mkdir(parents=True, exist_ok=True)
        if self._path.exists():
            self._path.unlink()

        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.bind(str(self._path))
        os.chmod(self._path, 0o600)
        sock.listen(8)
        sock.settimeout(0.5)
        self._server_sock = sock

        # Publish socket path for CLI discovery via /proc environ
        os.environ[SOCKET_ENV_VAR] = str(self._path)

        self._stop.clear()
        self._thread = threading.Thread(
            target=self._serve_loop,
            name=f"llminspect-transport-{self._pid}",
            daemon=True,
        )
        self._thread.start()
        _log.debug("Embedded transport listening on %s", self._path)

    def stop(self) -> None:
        self._stop.set()
        if self._server_sock:
            with contextlib_suppress():
                self._server_sock.close()
            self._server_sock = None
        if self._thread:
            self._thread.join(timeout=2.0)
            self._thread = None
        if self._path.exists():
            with contextlib_suppress():
                self._path.unlink()
        os.environ.pop(SOCKET_ENV_VAR, None)

    def fetch(self, request: CollectorRequest) -> CollectorResponse | None:
        if not self._path.exists():
            return None
        try:
            client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            client.settimeout(_DEFAULT_TIMEOUT)
            client.connect(str(self._path))
            payload = json.dumps(request.model_dump()) + "\n"
            client.sendall(payload.encode())
            data = _recv_line(client)
            client.close()
            if not data:
                return None
            parsed = json.loads(data)
            return CollectorResponse.model_validate(parsed)
        except OSError as exc:
            _log.debug("Transport fetch failed: %s", exc)
            return None

    def _serve_loop(self) -> None:
        assert self._server_sock is not None
        while not self._stop.is_set():
            try:
                conn, _ = self._server_sock.accept()
            except TimeoutError:
                continue
            except OSError:
                break
            threading.Thread(
                target=self._handle_connection,
                args=(conn,),
                daemon=True,
            ).start()

    def _handle_connection(self, conn: socket.socket) -> None:
        try:
            conn.settimeout(_DEFAULT_TIMEOUT)
            line = _recv_line(conn)
            if not line:
                return
            req = CollectorRequest.model_validate(json.loads(line))
            response = self._dispatch(req)
            conn.sendall((json.dumps(response.model_dump()) + "\n").encode())
        except Exception as exc:  # noqa: BLE001 — never crash host
            err = CollectorResponse(
                collector="unknown",
                error=f"Embedded transport error: {exc}",
            )
            with contextlib_suppress():
                conn.sendall((json.dumps(err.model_dump()) + "\n").encode())
        finally:
            with contextlib_suppress():
                conn.close()

    def _dispatch(self, request: CollectorRequest) -> CollectorResponse:
        if request.schema_version != SCHEMA_VERSION:
            return CollectorResponse(
                collector=request.collector,
                error=f"Unsupported schema version {request.schema_version}",
            )

        handler = self._handlers.get(request.collector)
        if handler is None:
            return CollectorResponse(
                collector=request.collector,
                error=f"Unknown collector '{request.collector}'",
            )

        start = time.perf_counter()
        try:
            data = handler(request)
            elapsed = (time.perf_counter() - start) * 1000
            return CollectorResponse(
                collector=request.collector,
                data=data,
                elapsed_ms=elapsed,
            )
        except Exception as exc:  # noqa: BLE001
            return CollectorResponse(
                collector=request.collector,
                error=str(exc),
            )


def _recv_line(sock: socket.socket, max_bytes: int = 1_048_576) -> str:
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        total += len(chunk)
        if b"\n" in chunk or total >= max_bytes:
            break
    if not chunks:
        return ""
    return b"".join(chunks).split(b"\n", 1)[0].decode()


class contextlib_suppress:
    """Minimal suppress — avoid importing contextlib in hot path."""

    def __enter__(self) -> None:
        pass

    def __exit__(self, *_: object) -> bool:
        return True
