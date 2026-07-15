"""Tests for EmbeddedSource and discovery."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

from llm_inspector.inspector.context import InspectionContext
from llm_inspector.models.enums import BackendKind, RuntimeKind
from llm_inspector.models.results import HardwareResult, ProcessResult
from llm_inspector.plugins.unknown import UnknownPlugin
from llm_inspector.rpc import CollectorRequest
from llm_inspector.serialization import model_to_dict
from llm_inspector.sources.embedded import EmbeddedSource
from llm_inspector.transport.discovery import discover_socket_path, default_socket_path
from llm_inspector.transport.unix import UnixSocketTransport
from llm_inspector.models.results import MemoryResult
from llm_inspector.models.measurement import Measurement
from tests.test_hardware_collector import MockEmptyBackend


def _make_ctx(pid: int) -> InspectionContext:
    return InspectionContext(
        pid=pid,
        process=ProcessResult(pid=pid, runtime_kind=RuntimeKind.UNKNOWN),
        hardware=HardwareResult(backend=BackendKind.CPU),
        plugin=UnknownPlugin(),
        backend=MockEmptyBackend(),
    )


class TestDiscovery:
    def test_discover_from_environ(self):
        with tempfile.TemporaryDirectory() as tmp:
            sock = Path(tmp) / "test.sock"
            sock.touch()
            path = discover_socket_path(9999, {"LLM_INSPECTOR_SOCK": str(sock)})
            assert path == sock

    def test_discover_missing_returns_none(self):
        path = discover_socket_path(99999999)
        assert path is None


class TestEmbeddedSource:
    def test_can_answer_when_socket_exists(self):
        with tempfile.TemporaryDirectory() as tmp:
            pid = os.getpid()
            sock = Path(tmp) / f"{pid}.sock"

            def handler(req: CollectorRequest) -> dict:
                result = MemoryResult(
                    gpu_allocated=Measurement[int].available(
                        999, source="test"
                    ),
                )
                return model_to_dict(result)

            transport = UnixSocketTransport(pid=pid, path=sock)
            transport.serve({"memory": handler})

            try:
                source = EmbeddedSource()
                ctx = _make_ctx(pid)

                with patch(
                    "llm_inspector.sources.embedded.discover_socket_path",
                    return_value=sock,
                ):
                    assert source.can_answer("memory", ctx)
                    data = source.collect("memory", ctx)
                    assert data is not None
                    assert data["gpu_allocated"]["status"] == "available"
            finally:
                transport.stop()

    def test_cannot_answer_without_socket(self):
        source = EmbeddedSource()
        ctx = _make_ctx(99999999)
        with patch(
            "llm_inspector.sources.embedded.discover_socket_path",
            return_value=None,
        ):
            assert not source.can_answer("memory", ctx)
