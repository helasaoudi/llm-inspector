"""EmbeddedSource — reads deep metrics from in-process Unix socket RPC."""

from __future__ import annotations

import logging
from typing import Any

from llm_inspector.inspector.context import InspectionContext
from llm_inspector.rpc import CollectorRequest
from llm_inspector.sources.base import Source
from llm_inspector.transport.discovery import discover_socket_path
from llm_inspector.transport.unix import UnixSocketTransport
from llm_inspector.utils import procfs

_log = logging.getLogger(__name__)

# Collectors served by embedded runtime
_EMBEDDED_COLLECTORS = frozenset({"memory", "model", "memory-breakdown"})


class EmbeddedSource(Source):
    """
    Highest-priority source — fetches from embedded inspector via RPC.

    Discovers socket via LLM_INSPECTOR_SOCK env var or default paths.
    """

    name = "embedded"

    def can_answer(self, collector: str, ctx: InspectionContext) -> bool:
        if collector not in _EMBEDDED_COLLECTORS:
            return False
        snap = procfs.snapshot(ctx.pid)
        path = discover_socket_path(ctx.pid, snap.environ)
        return path is not None

    def collect(self, collector: str, ctx: InspectionContext) -> dict[str, Any] | None:
        snap = procfs.snapshot(ctx.pid)
        path = discover_socket_path(ctx.pid, snap.environ)
        if path is None:
            return None

        transport = UnixSocketTransport(pid=ctx.pid, path=path)
        request = CollectorRequest(collector=collector)
        response = transport.fetch(request)

        if response is None or not response.succeeded or response.data is None:
            _log.debug(
                "EmbeddedSource failed for %s pid=%s: %s",
                collector,
                ctx.pid,
                response.error if response else "no response",
            )
            return None

        return response.data
