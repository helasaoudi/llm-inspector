"""Optimization Analysis — quantization projections on measured reports."""

from __future__ import annotations

from llm_inspector.models.enums import BackendKind, RuntimeKind
from llm_inspector.models.measurement import Measurement, MeasurementStatus
from llm_inspector.models.report import InspectionReport
from llm_inspector.models.results import (
    COMPONENT_ORDER,
    ComponentName,
    HardwareResult,
    MemoryBreakdownResult,
    MemoryComponent,
    ModelResult,
    ProcessResult,
    RuntimeResult,
)
from llm_inspector.optimization.analyzer import OptimizationAnalyzer
from llm_inspector.optimization.models import OptimizationKind


def _comp(name: str, value: int) -> MemoryComponent:
    return MemoryComponent(
        name=name,
        measurement=Measurement[int].available(value, source="test"),
        order=COMPONENT_ORDER.get(name, 99),
    )


def _report(
    *,
    weights: int = 2_300_000_000,
    kv: int = 2_900_000_000,
    workspace: int = 353_000_000,
    other: int = 223_000_000,
    params: int = 1_236_000_000,
    precision: str = "torch.bfloat16",
    runtime: RuntimeKind = RuntimeKind.VLLM,
    vram_total: int | None = 12_000_000_000,
) -> InspectionReport:
    total = weights + kv + workspace + other
    return InspectionReport(
        pid=1,
        process=ProcessResult(pid=1, cmdline=["VLLM::EngineCore"], runtime_kind=runtime),
        hardware=HardwareResult(
            backend=BackendKind.CUDA,
            vram_total_bytes=vram_total,
        ),
        model=ModelResult(
            name=Measurement[str].available("test-model", source="test"),
            parameter_count=Measurement[int].available(params, source="test"),
            precision=Measurement[str].available(precision, source="test"),
        ),
        memory_breakdown=MemoryBreakdownResult(
            components=[
                _comp(ComponentName.WEIGHTS, weights),
                _comp(ComponentName.KV_CACHE, kv),
                _comp(ComponentName.WORKSPACE, workspace),
                _comp(ComponentName.OTHER, other),
                _comp(ComponentName.ACTIVATIONS, 0),
            ],
            total=Measurement[int].available(total, source="test"),
        ),
        runtime=RuntimeResult(runtime_kind=runtime),
    )


class TestMeasurementSimulated:
    def test_simulated_not_available(self) -> None:
        m = Measurement[int].simulated(100, source="projected: test")
        assert m.is_simulated
        assert not m.is_available
        assert m.has_value
        assert m.status == MeasurementStatus.SIMULATED


class TestQuantizationAnalyzer:
    def test_projects_awq_and_saves(self) -> None:
        report = _report()
        opt = OptimizationAnalyzer().analyze(report)
        assert opt.skipped_reason is None
        quant = next(g for g in opt.groups if g.kind == OptimizationKind.QUANTIZATION)
        by_name = {s.method_name: s for s in quant.scenarios}

        awq = by_name["AWQ 4-bit"]
        assert awq.new_total.is_simulated
        assert awq.new_total.value is not None
        assert awq.saved_bytes.value is not None
        assert awq.saved_bytes.value > 0
        # New total should be less than current (~5.8 GB)
        assert awq.new_total.value < report.memory_breakdown.total.value  # type: ignore[union-attr]

        # KV/workspace unchanged in the projection sum:
        # new_total ≈ projected_weights + kv + workspace + other
        w = awq.details["weights"].value
        assert w is not None
        assert w < 2_300_000_000

    def test_gguf_unsupported_on_vllm(self) -> None:
        opt = OptimizationAnalyzer().analyze(_report(runtime=RuntimeKind.VLLM))
        quant = opt.groups[0]
        gguf = next(s for s in quant.scenarios if s.method_name.startswith("GGUF"))
        assert not gguf.new_total.has_value

    def test_recommendation_mentions_kv_bottleneck(self) -> None:
        # KV larger than weights
        opt = OptimizationAnalyzer().analyze(
            _report(weights=1_000_000_000, kv=4_000_000_000)
        )
        assert opt.recommendation is not None
        assert opt.recommendation.suggested_method is not None
        assert any("KV Cache" in w for w in opt.recommendation.warnings)

    def test_skipped_without_weights_or_params(self) -> None:
        report = InspectionReport(
            pid=1,
            process=ProcessResult(pid=1, cmdline=[], runtime_kind=RuntimeKind.VLLM),
            hardware=HardwareResult(backend=BackendKind.CUDA),
        )
        opt = OptimizationAnalyzer().analyze(report)
        assert opt.skipped_reason is not None
