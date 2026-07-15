"""
MemoryBreakdownCollector — Phase B.

Asks the runtime plugin to decompose GPU VRAM into named categories.
Each MemoryComponent carries its own Measurement[int] with provenance.

The collector is the gatekeeper: it ensures a valid MemoryBreakdownResult
is always returned (never None), even when the plugin has no data.

v0.2 roadmap:
  - Weights:     available from some runtimes via REST API / Prometheus
  - KV Cache:    available from vLLM /metrics (newer builds)
  - Activations: Phase 4
  - Workspace:   Phase 4
  - Other:       Phase 4
"""

from __future__ import annotations

from llm_inspector.collectors.base import Collector, CollectorResult
from llm_inspector.models.enums import CollectorPhase
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import (
    COMPONENT_ORDER,
    ComponentName,
    MemoryBreakdownResult,
    MemoryComponent,
)


class MemoryBreakdownCollector(Collector[MemoryBreakdownResult]):
    """
    Collect per-category GPU memory breakdown via the runtime plugin.

    Phase B — requires InspectionContext with a selected plugin.
    """

    name = "memory-breakdown"
    phase = CollectorPhase.B

    def collect(self, ctx: object) -> CollectorResult[MemoryBreakdownResult]:  # type: ignore[override]
        return super().collect(ctx)

    def _collect(self, ctx: object) -> MemoryBreakdownResult:  # type: ignore[override]
        from llm_inspector.inspector.context import InspectionContext  # noqa: PLC0415

        assert isinstance(ctx, InspectionContext)

        result = ctx.plugin.get_memory_breakdown(ctx)
        if result is not None:
            return result

        # Plugin returned None — return fully-unavailable breakdown
        reason = (
            f"Runtime plugin '{ctx.plugin.display_name}' does not implement "
            "memory breakdown inspection."
        )
        return MemoryBreakdownResult(
            components=[
                MemoryComponent(
                    name=name,
                    measurement=Measurement[int].unavailable(reason),
                    order=COMPONENT_ORDER.get(name, 99),
                )
                for name in (
                    ComponentName.WEIGHTS,
                    ComponentName.KV_CACHE,
                    ComponentName.ACTIVATIONS,
                    ComponentName.WORKSPACE,
                    ComponentName.OTHER,
                )
            ],
            total=Measurement[int].unavailable(reason),
        )
