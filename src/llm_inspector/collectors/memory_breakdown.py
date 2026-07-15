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
        from llm_inspector.utils import procfs  # noqa: PLC0415

        assert isinstance(ctx, InspectionContext)

        # 1. Try plugin (API-based — vLLM Prometheus, Ollama /api/ps, etc.)
        result = ctx.plugin.get_memory_breakdown(ctx)
        if result is not None:
            return result

        # 2. Fallback: /proc/<pid>/maps — extract weights size from
        #    memory-mapped model weight files. Works for any process on Linux.
        snap = procfs.snapshot(ctx.process.pid)
        weights_bytes = snap.total_weights_bytes
        weights_path = snap.hf_model_path

        weights_meas: Measurement[int]
        if not snap.maps_readable and not snap.fd_readable:
            weights_meas = Measurement[int].unavailable(
                f"Cannot read /proc/{ctx.process.pid}/maps (permission denied). "
                "Try running llminspect with sudo."
            )
        elif weights_bytes:
            source = (
                f"/proc/{ctx.process.pid}/maps → {weights_path}"
                if weights_path
                else f"/proc/{ctx.process.pid}/maps → model weight files"
            )
            weights_meas = Measurement[int].available(weights_bytes, source=source)
        else:
            weights_meas = Measurement[int].unavailable(
                "No model weight files found in /proc maps. "
                "Weights may be on GPU-only memory with no CPU mapping."
            )

        unavail_kv = Measurement[int].unavailable(
            "KV cache is allocated dynamically on GPU. "
            "Requires runtime API (e.g. vLLM Prometheus /metrics)."
        )
        unavail_act = Measurement[int].unavailable(
            "Activations are transient — not measurable without in-process hooks."
        )
        unavail_ws = Measurement[int].unavailable(
            "Workspace memory requires in-process PyTorch/CUDA hooks."
        )
        unavail_other = Measurement[int].unavailable(
            "Other GPU memory not attributable from external inspection."
        )

        # Total: weights only if we have it; otherwise unavailable
        total = (
            Measurement[int].available(
                weights_bytes,
                source=f"/proc/{ctx.process.pid}/maps → sum of weight files",
            )
            if weights_bytes
            else Measurement[int].unavailable(
                "Total breakdown unavailable without runtime API."
            )
        )

        return MemoryBreakdownResult(
            components=[
                MemoryComponent(
                    name=ComponentName.WEIGHTS,
                    measurement=weights_meas,
                    order=COMPONENT_ORDER.get(ComponentName.WEIGHTS, 0),
                ),
                MemoryComponent(
                    name=ComponentName.KV_CACHE,
                    measurement=unavail_kv,
                    order=COMPONENT_ORDER.get(ComponentName.KV_CACHE, 1),
                ),
                MemoryComponent(
                    name=ComponentName.ACTIVATIONS,
                    measurement=unavail_act,
                    order=COMPONENT_ORDER.get(ComponentName.ACTIVATIONS, 2),
                ),
                MemoryComponent(
                    name=ComponentName.WORKSPACE,
                    measurement=unavail_ws,
                    order=COMPONENT_ORDER.get(ComponentName.WORKSPACE, 3),
                ),
                MemoryComponent(
                    name=ComponentName.OTHER,
                    measurement=unavail_other,
                    order=COMPONENT_ORDER.get(ComponentName.OTHER, 4),
                ),
            ],
            total=total,
        )
