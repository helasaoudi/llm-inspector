"""Tests for MemoryCollector using a mock InspectionContext."""

import os
from unittest.mock import MagicMock

from llm_inspector.backends.base import DeviceProcess
from llm_inspector.collectors.memory import MemoryCollector
from llm_inspector.models.enums import BackendKind
from llm_inspector.models.results import HardwareResult, ProcessResult
from llm_inspector.models.enums import RuntimeKind


def _make_context(pid: int, with_gpu: bool = True):
    """Build a minimal InspectionContext mock for testing."""
    from llm_inspector.inspector.context import InspectionContext
    from llm_inspector.plugins.unknown import UnknownPlugin
    from tests.test_hardware_collector import MockCUDABackend, MockEmptyBackend

    backend = MockCUDABackend(pid=pid) if with_gpu else MockEmptyBackend()

    process = ProcessResult(pid=pid, cmdline=["python", "test.py"])
    hardware = HardwareResult(
        backend=BackendKind.CUDA if with_gpu else BackendKind.CPU,
        device_index=0 if with_gpu else None,
    )

    return InspectionContext(
        pid=pid,
        process=process,
        hardware=hardware,
        plugin=UnknownPlugin(),
        backend=backend,
    )


class TestMemoryCollector:
    def test_gpu_used_available_when_gpu_attached(self):
        pid = os.getpid()
        ctx = _make_context(pid, with_gpu=True)
        collector = MemoryCollector()
        result = collector.collect(ctx)

        assert result.succeeded
        data = result.data
        assert data is not None
        assert data.gpu_used.is_available
        assert data.gpu_used.value == 21 * 1024**3

    def test_gpu_used_unavailable_when_no_gpu(self):
        pid = os.getpid()
        ctx = _make_context(pid, with_gpu=False)
        collector = MemoryCollector()
        result = collector.collect(ctx)

        assert result.succeeded
        data = result.data
        assert data is not None
        assert not data.gpu_used.is_available

    def test_torch_stats_unavailable_in_phase_1(self):
        pid = os.getpid()
        ctx = _make_context(pid)
        collector = MemoryCollector()
        result = collector.collect(ctx)

        data = result.data
        assert data is not None
        # Phase 1: PyTorch stats always unavailable
        assert not data.gpu_allocated.is_available
        assert not data.gpu_reserved.is_available
        assert not data.peak.is_available

    def test_process_ram_available_for_current_process(self):
        pid = os.getpid()
        ctx = _make_context(pid)
        collector = MemoryCollector()
        result = collector.collect(ctx)

        data = result.data
        assert data is not None
        assert data.process_ram.is_available
        assert data.process_ram.value is not None
        assert data.process_ram.value > 0

    def test_collector_name(self):
        assert MemoryCollector.name == "memory"
