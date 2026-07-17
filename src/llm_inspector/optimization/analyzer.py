"""OptimizationAnalyzer — post-inspect projections from measured report."""

from __future__ import annotations

from llm_inspector.models.enums import RuntimeKind
from llm_inspector.models.report import InspectionReport
from llm_inspector.models.results import ComponentName
from llm_inspector.optimization.base import OptimizationContext, OptimizationStrategy
from llm_inspector.optimization.models import OptimizationAnalysis
from llm_inspector.optimization.quantization.strategy import QuantizationStrategy
from llm_inspector.optimization.recommendations import build_recommendation


def _component_bytes(report: InspectionReport, name: str) -> int | None:
    bd = report.memory_breakdown
    if bd is None:
        return None
    comp = bd.get(name)
    if comp is None or not comp.measurement.is_available:
        return None
    return comp.measurement.value


def build_context(report: InspectionReport) -> OptimizationContext:
    runtime = (
        report.runtime.runtime_kind
        if report.runtime is not None
        else report.process.runtime_kind
    )
    if not isinstance(runtime, RuntimeKind):
        runtime = RuntimeKind.UNKNOWN

    param_count = None
    precision = None
    if report.model is not None:
        if report.model.parameter_count.is_available:
            param_count = report.model.parameter_count.value
        if report.model.precision.is_available:
            precision = report.model.precision.value

    weights = _component_bytes(report, ComponentName.WEIGHTS)
    kv = _component_bytes(report, ComponentName.KV_CACHE)
    workspace = _component_bytes(report, ComponentName.WORKSPACE)
    other = _component_bytes(report, ComponentName.OTHER)
    activations = _component_bytes(report, ComponentName.ACTIVATIONS)

    current_total = None
    if report.memory_breakdown is not None and report.memory_breakdown.total.is_available:
        current_total = report.memory_breakdown.total.value

    vram_total = report.hardware.vram_total_bytes

    return OptimizationContext(
        report=report,
        runtime=runtime,
        param_count=param_count,
        precision=precision,
        weights_bytes=weights,
        kv_bytes=kv,
        workspace_bytes=workspace,
        other_bytes=other,
        activations_bytes=activations,
        current_total=current_total,
        vram_total=vram_total,
    )


class OptimizationAnalyzer:
    """
    Compute Optimization Analysis after InspectionReport is built.

    Does not run collectors. Does not mutate models.
    """

    def __init__(self, strategies: list[OptimizationStrategy] | None = None) -> None:
        self._strategies = strategies or [QuantizationStrategy()]

    def analyze(self, report: InspectionReport) -> OptimizationAnalysis:
        ctx = build_context(report)

        if ctx.weights_bytes is None and ctx.param_count is None:
            return OptimizationAnalysis(
                groups=[],
                recommendation=None,
                skipped_reason=(
                    "Optimization Analysis needs measured Weights or parameter_count. "
                    "Use embedded attach() for full projections."
                ),
            )

        groups = [s.analyze(ctx) for s in self._strategies]
        recommendation = build_recommendation(ctx, groups)
        return OptimizationAnalysis(
            groups=groups,
            recommendation=recommendation,
        )
