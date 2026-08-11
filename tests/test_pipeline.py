"""Tests for CollectorPipeline — Phase A (sequential) / Phase B (parallel)."""

from __future__ import annotations

from unittest.mock import MagicMock

from llm_inspector.backends.cpu import CPUBackend
from llm_inspector.collectors.base import CollectorResult
from llm_inspector.inspector.context import InspectionContext
from llm_inspector.inspector.pipeline import (
    COLLECT_MEMORY,
    COLLECT_MODEL,
    CollectorPipeline,
)
from llm_inspector.models.enums import BackendKind, CollectorStatus, RuntimeKind
from llm_inspector.models.results import HardwareResult, MemoryResult, ModelResult, ProcessResult
from llm_inspector.plugins.unknown import UnknownPlugin


def _make_ctx() -> InspectionContext:
    process = ProcessResult(pid=1, cmdline=[], runtime_kind=RuntimeKind.UNKNOWN)
    hardware = HardwareResult(backend=BackendKind.CPU)
    return InspectionContext(
        pid=1, process=process, hardware=hardware, plugin=UnknownPlugin(), backend=CPUBackend()
    )


def _result(name: str, data: object) -> CollectorResult:
    return CollectorResult(collector_name=name, status=CollectorStatus.SUCCESS, data=data)


class TestRunPhaseA:
    def test_calls_process_then_hardware_collectors(self) -> None:
        pipeline = CollectorPipeline()
        pipeline._process_collector = MagicMock()
        pipeline._hardware_collector = MagicMock()
        pipeline._process_collector.collect.return_value = _result(
            "process", ProcessResult(pid=7, cmdline=[], runtime_kind=RuntimeKind.UNKNOWN)
        )
        pipeline._hardware_collector.collect.return_value = _result(
            "hardware", HardwareResult(backend=BackendKind.CPU)
        )
        backend = CPUBackend()

        phase_a = pipeline.run_phase_a(pid=7, backend=backend)

        pipeline._process_collector.collect.assert_called_once_with(7)
        pipeline._hardware_collector.collect.assert_called_once_with(7, backend)
        assert phase_a.succeeded
        assert phase_a.process.data.pid == 7

    def test_failed_process_collector_marks_phase_a_failed(self) -> None:
        pipeline = CollectorPipeline()
        pipeline._process_collector = MagicMock()
        pipeline._hardware_collector = MagicMock()
        pipeline._process_collector.collect.return_value = CollectorResult(
            collector_name="process", status=CollectorStatus.FAILED, error="boom"
        )
        pipeline._hardware_collector.collect.return_value = _result(
            "hardware", HardwareResult(backend=BackendKind.CPU)
        )

        phase_a = pipeline.run_phase_a(pid=1, backend=CPUBackend())

        assert not phase_a.succeeded


class TestRunPhaseB:
    def test_all_filter_runs_every_collector(self) -> None:
        pipeline = CollectorPipeline()
        pipeline._memory_collector = MagicMock()
        pipeline._model_collector = MagicMock()
        pipeline._memory_breakdown_collector = MagicMock()
        pipeline._runtime_collector = MagicMock()

        pipeline._memory_collector.collect.return_value = _result("memory", MemoryResult())
        pipeline._model_collector.collect.return_value = _result("model", ModelResult())
        pipeline._memory_breakdown_collector.collect.return_value = _result(
            "memory-breakdown", None
        )
        pipeline._runtime_collector.collect.return_value = _result("runtime", None)

        phase_b = pipeline.run_phase_b(_make_ctx(), collect_filter="all")

        assert phase_b.memory is not None and phase_b.memory.data is not None
        assert phase_b.model is not None
        assert phase_b.memory_breakdown is not None
        assert phase_b.runtime is not None
        assert phase_b.errors == {}

    def test_single_filter_runs_only_that_collector(self) -> None:
        pipeline = CollectorPipeline()
        pipeline._memory_collector = MagicMock()
        pipeline._model_collector = MagicMock()
        pipeline._memory_breakdown_collector = MagicMock()
        pipeline._runtime_collector = MagicMock()
        pipeline._memory_collector.collect.return_value = _result("memory", MemoryResult())

        phase_b = pipeline.run_phase_b(_make_ctx(), collect_filter=COLLECT_MEMORY)

        assert phase_b.memory is not None
        assert phase_b.model is None
        pipeline._model_collector.collect.assert_not_called()
        pipeline._memory_breakdown_collector.collect.assert_not_called()
        pipeline._runtime_collector.collect.assert_not_called()

    def test_unknown_filter_returns_empty_result(self) -> None:
        pipeline = CollectorPipeline()
        pipeline._memory_collector = MagicMock()

        phase_b = pipeline.run_phase_b(_make_ctx(), collect_filter="not-a-real-collector")

        assert phase_b.memory is None
        assert phase_b.model is None
        pipeline._memory_collector.collect.assert_not_called()

    def test_collector_exception_is_recorded_not_raised(self) -> None:
        pipeline = CollectorPipeline()
        pipeline._memory_collector = MagicMock()
        pipeline._memory_collector.collect.side_effect = ValueError("kaboom")

        phase_b = pipeline.run_phase_b(_make_ctx(), collect_filter=COLLECT_MEMORY)

        assert phase_b.memory is None
        assert "memory" in phase_b.errors
        assert "kaboom" in phase_b.errors["memory"]

    def test_failed_collector_result_is_recorded_in_errors(self) -> None:
        pipeline = CollectorPipeline()
        pipeline._model_collector = MagicMock()
        pipeline._model_collector.collect.return_value = CollectorResult(
            collector_name="model", status=CollectorStatus.FAILED, error="plugin unavailable"
        )

        phase_b = pipeline.run_phase_b(_make_ctx(), collect_filter=COLLECT_MODEL)

        assert phase_b.errors["model"] == "plugin unavailable"
        assert phase_b.model is not None
        assert phase_b.model.failed
