"""
MemoryBreakdownCollector — Phase B.

Asks the runtime plugin to decompose GPU VRAM into named categories.
Each MemoryComponent carries its own Measurement[int] with provenance.

The collector is the gatekeeper: it ensures a valid MemoryBreakdownResult
is always returned (never None), even when the plugin has no data.

Resolution order:
  1. EmbeddedSource (adapter weights / KV from EngineCore)
  2. Runtime plugin API (vLLM Prometheus, Ollama /api/ps, …)
  3. /proc/<pid>/maps weight files
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
from llm_inspector.serialization import model_from_dict
from llm_inspector.sources.resolver import get_resolver


def merge_memory_breakdown(
    primary: MemoryBreakdownResult,
    secondary: MemoryBreakdownResult | None,
) -> MemoryBreakdownResult:
    """Merge breakdowns field-by-field; primary wins when available."""
    if secondary is None:
        return primary

    secondary_by_name = {c.name: c for c in secondary.components}
    merged: list[MemoryComponent] = []
    seen: set[str] = set()

    for comp in primary.components:
        seen.add(comp.name)
        other = secondary_by_name.get(comp.name)
        if comp.measurement.is_available or other is None:
            merged.append(comp)
        else:
            merged.append(
                MemoryComponent(
                    name=comp.name,
                    measurement=other.measurement,
                    description=comp.description or other.description,
                    order=comp.order,
                )
            )

    for comp in secondary.components:
        if comp.name not in seen:
            merged.append(comp)

    total = (
        primary.total
        if primary.total.is_available
        else secondary.total
    )
    if not total.is_available:
        measured = [
            c.measurement.value
            for c in merged
            if c.measurement.is_available and c.measurement.value is not None
        ]
        if measured:
            total = Measurement[int].available(
                sum(measured),
                source="Sum of measured breakdown components",
            )

    return MemoryBreakdownResult(components=merged, total=total)


class MemoryBreakdownCollector(Collector[MemoryBreakdownResult]):
    """
    Collect per-category GPU memory breakdown via embedded + runtime plugin.

    Phase B — requires InspectionContext with a selected plugin.
    """

    name = "memory-breakdown"
    phase = CollectorPhase.B

    def collect(self, ctx: object) -> CollectorResult[MemoryBreakdownResult]:  # type: ignore[override]
        return super().collect(ctx)

    def _collect(self, ctx: object) -> MemoryBreakdownResult:  # type: ignore[override]
        from llm_inspector.inspector.context import InspectionContext  # noqa: PLC0415

        assert isinstance(ctx, InspectionContext)

        # 1. Embedded inspector (EngineCore weights / KV tensors)
        embedded: MemoryBreakdownResult | None = None
        embedded_data = get_resolver().fetch("memory-breakdown", ctx)
        if embedded_data is not None:
            embedded = model_from_dict(MemoryBreakdownResult, embedded_data)

        # 2. Plugin API (Prometheus /metrics, Ollama, …)
        plugin_result = ctx.plugin.get_memory_breakdown(ctx)

        if embedded is not None:
            has_any = any(c.measurement.is_available for c in embedded.components)
            if has_any:
                return merge_memory_breakdown(embedded, plugin_result)

        if plugin_result is not None:
            has_any = any(c.measurement.is_available for c in plugin_result.components)
            if has_any:
                return plugin_result
            # Keep plugin reasons (e.g. missing Prometheus byte metrics) unless
            # /proc can fill Weights.
            proc_fallback = self._from_proc(ctx)
            return merge_memory_breakdown(plugin_result, proc_fallback)

        return self._from_proc(ctx)

    @staticmethod
    def _from_proc(ctx: object) -> MemoryBreakdownResult:
        from llm_inspector.inspector.context import InspectionContext  # noqa: PLC0415
        from llm_inspector.utils import procfs  # noqa: PLC0415

        assert isinstance(ctx, InspectionContext)

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
            "Requires runtime API (e.g. vLLM Prometheus /metrics) or embedded attach."
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

        total = (
            Measurement[int].available(
                weights_bytes,
                source=f"/proc/{ctx.process.pid}/maps → sum of weight files",
            )
            if weights_bytes
            else Measurement[int].unavailable(
                "Total breakdown unavailable without runtime API or embedded attach."
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
