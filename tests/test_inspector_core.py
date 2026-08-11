"""Tests for Inspector — the central orchestrator."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_inspector.backends.cpu import CPUBackend
from llm_inspector.collectors.base import CollectorResult
from llm_inspector.inspector.core import Inspector
from llm_inspector.inspector.pipeline import COLLECT_ALL, PhaseAResult, PhaseBResult
from llm_inspector.models.enums import BackendKind, CollectorStatus, RuntimeKind
from llm_inspector.models.results import HardwareResult, ProcessResult
from llm_inspector.plugins.unknown import UnknownPlugin


def _process_result(pid: int = 1) -> ProcessResult:
    return ProcessResult(pid=pid, cmdline=["python"], runtime_kind=RuntimeKind.UNKNOWN)


def _hardware_result() -> HardwareResult:
    return HardwareResult(backend=BackendKind.CPU)


def _succeeded_phase_a(pid: int = 1) -> PhaseAResult:
    process_result = CollectorResult(
        collector_name="process", status=CollectorStatus.SUCCESS, data=_process_result(pid)
    )
    hardware_result = CollectorResult(
        collector_name="hardware", status=CollectorStatus.SUCCESS, data=_hardware_result()
    )
    return PhaseAResult(process_result, hardware_result)


def _failed_phase_a() -> PhaseAResult:
    process_result = CollectorResult(
        collector_name="process", status=CollectorStatus.FAILED, error="No such process"
    )
    hardware_result = CollectorResult(
        collector_name="hardware", status=CollectorStatus.SUCCESS, data=_hardware_result()
    )
    return PhaseAResult(process_result, hardware_result)


def _make_inspector(pipeline: MagicMock) -> Inspector:
    backend_registry = MagicMock()
    backend_registry.best_available.return_value = CPUBackend()
    plugin_registry = MagicMock()
    plugin_registry.select.return_value = UnknownPlugin()
    return Inspector(
        backend_registry=backend_registry, plugin_registry=plugin_registry, pipeline=pipeline
    )


class TestInspect:
    def test_success_assembles_report(self) -> None:
        pipeline = MagicMock()
        pipeline.run_phase_a.return_value = _succeeded_phase_a(pid=42)
        pipeline.run_phase_b.return_value = PhaseBResult()
        inspector = _make_inspector(pipeline)

        report = inspector.inspect(pid=42, collect_filter="hardware")

        assert report.pid == 42
        assert report.process.pid == 42
        assert report.hardware.backend == BackendKind.CPU
        pipeline.run_phase_b.assert_called_once()

    def test_phase_a_failure_raises_runtime_error(self) -> None:
        pipeline = MagicMock()
        pipeline.run_phase_a.return_value = _failed_phase_a()
        inspector = _make_inspector(pipeline)

        with pytest.raises(RuntimeError, match="No such process"):
            inspector.inspect(pid=1)

        pipeline.run_phase_b.assert_not_called()

    def test_collect_filter_all_appends_optimization(self) -> None:
        pipeline = MagicMock()
        pipeline.run_phase_a.return_value = _succeeded_phase_a()
        pipeline.run_phase_b.return_value = PhaseBResult()
        inspector = _make_inspector(pipeline)

        fake_analysis = MagicMock()
        with patch("llm_inspector.optimization.analyzer.OptimizationAnalyzer") as mock_cls:
            mock_cls.return_value.analyze.return_value = fake_analysis
            report = inspector.inspect(pid=1, collect_filter=COLLECT_ALL)

        assert report.optimization is fake_analysis

    def test_non_all_filter_skips_optimization(self) -> None:
        pipeline = MagicMock()
        pipeline.run_phase_a.return_value = _succeeded_phase_a()
        pipeline.run_phase_b.return_value = PhaseBResult()
        inspector = _make_inspector(pipeline)

        report = inspector.inspect(pid=1, collect_filter="hardware")

        assert report.optimization is None

    def test_optimization_failure_is_swallowed(self) -> None:
        pipeline = MagicMock()
        pipeline.run_phase_a.return_value = _succeeded_phase_a()
        pipeline.run_phase_b.return_value = PhaseBResult()
        inspector = _make_inspector(pipeline)

        with patch("llm_inspector.optimization.analyzer.OptimizationAnalyzer") as mock_cls:
            mock_cls.return_value.analyze.side_effect = RuntimeError("boom")
            report = inspector.inspect(pid=1, collect_filter=COLLECT_ALL)

        assert report.optimization is None

    def test_collector_errors_and_elapsed_recorded(self) -> None:
        pipeline = MagicMock()
        pipeline.run_phase_a.return_value = _succeeded_phase_a()
        phase_b = PhaseBResult()
        phase_b.errors["model"] = "ModelCollector failed"
        phase_b.elapsed["model"] = 12.5
        pipeline.run_phase_b.return_value = phase_b
        inspector = _make_inspector(pipeline)

        report = inspector.inspect(pid=1, collect_filter="hardware")

        assert report.collector_errors == {"model": "ModelCollector failed"}
        assert report.elapsed_ms["model"] == 12.5
        assert "process" in report.elapsed_ms
        assert "hardware" in report.elapsed_ms


class TestScan:
    def test_merges_gpu_and_heuristic_pids_and_sorts(self) -> None:
        pipeline = MagicMock()
        pipeline.run_phase_a.side_effect = lambda pid, backend: _succeeded_phase_a(pid=pid)
        pipeline.run_phase_b.return_value = PhaseBResult()

        backend_registry = MagicMock()
        backend = MagicMock()
        backend.pid_to_devices.return_value = {5: []}
        backend_registry.best_available.return_value = backend

        plugin_registry = MagicMock()
        plugin_registry.select.return_value = UnknownPlugin()

        inspector = Inspector(
            backend_registry=backend_registry, plugin_registry=plugin_registry, pipeline=pipeline
        )

        with patch.object(inspector, "_heuristic_scan", return_value={2: []}):
            reports = inspector.scan()

        assert [r.pid for r in reports] == [2, 5]

    def test_skips_process_that_exited_between_scan_and_inspect(self) -> None:
        pipeline = MagicMock()

        def run_phase_a(pid, backend):
            if pid == 1:
                return _failed_phase_a()
            return _succeeded_phase_a(pid=pid)

        pipeline.run_phase_a.side_effect = run_phase_a
        pipeline.run_phase_b.return_value = PhaseBResult()

        backend_registry = MagicMock()
        backend = MagicMock()
        backend.pid_to_devices.return_value = {1: [], 2: []}
        backend_registry.best_available.return_value = backend

        plugin_registry = MagicMock()
        plugin_registry.select.return_value = UnknownPlugin()

        inspector = Inspector(
            backend_registry=backend_registry, plugin_registry=plugin_registry, pipeline=pipeline
        )

        with patch.object(inspector, "_heuristic_scan", return_value={}):
            reports = inspector.scan()

        assert [r.pid for r in reports] == [2]


class TestHeuristicScan:
    def test_detects_known_runtime_from_cmdline(self) -> None:
        inspector = Inspector()
        with (
            patch("llm_inspector.utils.proc.list_all_pids", return_value=[1, 2]),
            patch(
                "llm_inspector.utils.proc.read_cmdline",
                side_effect=lambda pid: ["ollama", "serve"] if pid == 1 else ["bash"],
            ),
        ):
            found = inspector._heuristic_scan()

        assert found == {1: []}

    def test_ignores_processes_that_raise(self) -> None:
        inspector = Inspector()
        with (
            patch("llm_inspector.utils.proc.list_all_pids", return_value=[1]),
            patch("llm_inspector.utils.proc.read_cmdline", side_effect=OSError("gone")),
        ):
            found = inspector._heuristic_scan()

        assert found == {}


class TestMisc:
    def test_available_backend_delegates_to_registry(self) -> None:
        backend_registry = MagicMock()
        cpu_backend = CPUBackend()
        backend_registry.best_available.return_value = cpu_backend
        inspector = Inspector(backend_registry=backend_registry)

        assert inspector.available_backend() is cpu_backend

    def test_registered_plugins_delegates_to_registry(self) -> None:
        plugin_registry = MagicMock()
        plugin_registry.all_plugins.return_value = [UnknownPlugin()]
        inspector = Inspector(plugin_registry=plugin_registry)

        plugins = inspector.registered_plugins()

        assert len(plugins) == 1
        assert isinstance(plugins[0], UnknownPlugin)
