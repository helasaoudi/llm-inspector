"""
RuntimeCollector — Phase B (plugin-enriched in Phase 2).

Combines heuristic runtime detection from ProcessCollector with plugin-
supplied details (PagedAttention, scheduler, tensor-parallel metadata, etc.)

Phase 1: version from NVML backend, empty details dict.
Phase 2: plugin.get_version() + plugin.get_runtime_details() — runtime-
         specific key/value pairs with full Measurement provenance.
"""

from __future__ import annotations

from llm_inspector.collectors.base import Collector, CollectorResult
from llm_inspector.models.enums import CollectorPhase
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import RuntimeResult


class RuntimeCollector(Collector[RuntimeResult]):
    """
    Collect runtime identity and plugin-enriched metadata.

    Phase B — receives a full InspectionContext.
    """

    name = "runtime"
    phase = CollectorPhase.B

    def collect(self, ctx: object) -> CollectorResult[RuntimeResult]:  # type: ignore[override]
        return super().collect(ctx)

    def _collect(self, ctx: object) -> RuntimeResult:  # type: ignore[override]
        from llm_inspector.inspector.context import InspectionContext  # noqa: PLC0415

        assert isinstance(ctx, InspectionContext)

        runtime_kind = ctx.process.runtime_kind

        # Version — plugin first, fall back to backend compute runtime
        version = ctx.plugin.get_version(ctx)
        if not version.is_available:
            cuda_v = ctx.backend.runtime_version()
            if cuda_v:
                version = Measurement[str].available(
                    cuda_v, source="NVML nvmlSystemGetCudaDriverVersion()"
                )

        # Plugin-specific details (PagedAttention, Scheduler, etc.)
        details = ctx.plugin.get_runtime_details(ctx)

        return RuntimeResult(
            runtime_kind=runtime_kind,
            version=version,
            details=details,
        )
