"""
RuntimePlugin ABC — the contract all runtime plugins must satisfy.

Design: capability methods, not a monolithic enrich()
------------------------------------------------------
Each plugin implements only the capabilities its runtime exposes.
Collectors call the specific method they need:
  - ModelCollector      → plugin.get_model_info(ctx)
  - MemoryBreakdown     → plugin.get_memory_breakdown(ctx)
  - RuntimeCollector    → plugin.get_runtime_details(ctx)
  - MemoryCollector     → plugin.get_memory_stats(ctx)  [Phase 3]

Default implementations return None / empty — collectors fall back to
Measurement.unavailable() when the plugin doesn't implement a capability.
This is the Interface Segregation Principle: plugins implement only what
their runtime actually exposes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from llm_inspector.models.enums import RuntimeKind
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import (
    MemoryBreakdownResult,
    ModelResult,
    RuntimeResult,
)

if TYPE_CHECKING:
    from llm_inspector.inspector.context import InspectionContext
    from llm_inspector.models.results import ProcessResult


class RuntimePlugin(ABC):
    """
    Abstract base class for all LLM runtime plugins.

    Plugins ENRICH inspection data — they do not own the inspection pipeline.
    They are called by collectors which use their output to populate result fields.
    """

    kind: RuntimeKind = RuntimeKind.UNKNOWN
    display_name: str = "Unknown"
    description: str = ""

    @abstractmethod
    def supports(self, process: "ProcessResult") -> bool:
        """
        Confirm whether this plugin can handle the given process.

        Secondary check after heuristic RuntimeKind detection.
        Return False to fall back to UnknownPlugin.
        """

    # ── Capability methods — override only what your runtime exposes ──────────

    def get_model_info(self, ctx: "InspectionContext") -> ModelResult | None:
        """
        Return model identity information, or None if unavailable.

        Override in plugins that can extract model name, precision, context
        length, etc. from cmdline args or a REST API.
        """
        return None

    def get_memory_breakdown(
        self, ctx: "InspectionContext"
    ) -> MemoryBreakdownResult | None:
        """
        Return a breakdown of GPU memory by category, or None.

        Override in plugins whose runtime exposes per-category memory data.
        Each MemoryComponent carries its own Measurement[int] with provenance.
        Never estimate — return Measurement.unavailable() for unknown categories.
        """
        return None

    def get_runtime_details(
        self, ctx: "InspectionContext"
    ) -> dict[str, Measurement[str]]:
        """
        Return runtime-specific key/value metadata.

        Examples:
          vLLM:   {"PagedAttention": M.available("Enabled", "vLLM config")}
          Ollama: {"Scheduler": M.unavailable("Not exposed")}
        Override to add runtime-specific details to the Runtime section.
        """
        return {}

    def get_version(self, ctx: "InspectionContext") -> Measurement[str]:
        """Return the runtime version string, or Measurement.unavailable()."""
        return Measurement[str].unavailable("Version not available for this runtime.")
