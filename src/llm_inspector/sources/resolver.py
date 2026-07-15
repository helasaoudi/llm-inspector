"""
SourceResolver — priority chain for collector data.

Priority (highest first):
  1. EmbeddedSource   — Unix socket RPC (in-process inspector)
  2. RuntimeAPISource — vLLM / Ollama / FastAPI HTTP APIs (future refactor)
  3. ProcFSSource     — /proc maps, environ
  4. NVMLSource       — GPU totals via NVML

v0.3: EmbeddedSource wired for memory collector PyTorch fields.
Runtime API / ProcFS remain in existing collector logic until Phase B refactor.
"""

from __future__ import annotations

from typing import Any

from llm_inspector.inspector.context import InspectionContext
from llm_inspector.sources.base import Source
from llm_inspector.sources.embedded import EmbeddedSource


class SourceResolver:
    """Try sources in priority order until one answers."""

    def __init__(self, sources: list[Source] | None = None) -> None:
        self._sources: list[Source] = sources or [
            EmbeddedSource(),
        ]

    def fetch(self, collector: str, ctx: InspectionContext) -> dict[str, Any] | None:
        for source in self._sources:
            if source.can_answer(collector, ctx):
                data = source.collect(collector, ctx)
                if data is not None:
                    return data
        return None

    def is_embedded_available(self, collector: str, ctx: InspectionContext) -> bool:
        for source in self._sources:
            if isinstance(source, EmbeddedSource) and source.can_answer(collector, ctx):
                return True
        return False


# Module-level default resolver
_default_resolver = SourceResolver()


def get_resolver() -> SourceResolver:
    return _default_resolver
