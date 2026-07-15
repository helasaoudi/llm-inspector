"""Tests for Unix socket transport."""

from __future__ import annotations

import os
import tempfile
import threading
from pathlib import Path

from llm_inspector.rpc import CollectorRequest
from llm_inspector.transport.unix import UnixSocketTransport


def test_unix_socket_round_trip():
    with tempfile.TemporaryDirectory() as tmp:
        sock_path = Path(tmp) / f"{os.getpid()}.sock"

        def memory_handler(request: CollectorRequest) -> dict:
            return {"collector": request.collector, "value": 42}

        server = UnixSocketTransport(path=sock_path)
        server.serve({"memory": memory_handler})

        try:
            client = UnixSocketTransport(path=sock_path)
            resp = client.fetch(CollectorRequest(collector="memory"))
            assert resp is not None
            assert resp.succeeded
            assert resp.data is not None
            assert resp.data["value"] == 42
        finally:
            server.stop()

        assert not sock_path.exists()


def test_unknown_collector_returns_error():
    with tempfile.TemporaryDirectory() as tmp:
        sock_path = Path(tmp) / "test.sock"
        server = UnixSocketTransport(path=sock_path)
        server.serve({})

        try:
            client = UnixSocketTransport(path=sock_path)
            resp = client.fetch(CollectorRequest(collector="layers"))
            assert resp is not None
            assert not resp.succeeded
            assert resp.error is not None
            assert "Unknown collector" in resp.error
        finally:
            server.stop()
