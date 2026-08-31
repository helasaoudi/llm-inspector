"""Tests for HardwareCollector using a mock backend."""

import os

from llm_inspector.backends.base import DeviceInfo, DeviceProcess, HardwareBackend
from llm_inspector.collectors.hardware import HardwareCollector
from llm_inspector.models.enums import BackendKind


class MockCUDABackend(HardwareBackend):
    """Minimal mock backend that simulates one NVIDIA GPU with one attached process."""

    kind = BackendKind.CUDA

    def __init__(self, pid: int) -> None:
        self._pid = pid

    def is_available(self) -> bool:
        return True

    def list_devices(self) -> list[DeviceInfo]:
        return [
            DeviceInfo(
                index=0,
                name="NVIDIA RTX 4090",
                vram_total_bytes=24 * 1024**3,
                vram_used_bytes=21 * 1024**3,
                gpu_utilization_pct=93,
                driver_version="575.64",
                cuda_version="12.4",
            )
        ]

    def device_processes(self, device_index: int) -> list[DeviceProcess]:
        return [DeviceProcess(pid=self._pid, device_index=0, vram_used_bytes=21 * 1024**3)]

    def driver_version(self) -> str | None:
        return "575.64"

    def runtime_version(self) -> str | None:
        return "12.4"


class MockEmptyBackend(HardwareBackend):
    """Backend that returns no devices — simulates CPU-only machine."""

    kind = BackendKind.CPU

    def is_available(self) -> bool:
        return True

    def list_devices(self) -> list[DeviceInfo]:
        return []

    def device_processes(self, device_index: int) -> list[DeviceProcess]:
        return []

    def driver_version(self) -> str | None:
        return None

    def runtime_version(self) -> str | None:
        return None


class TestHardwareCollector:
    def test_gpu_process_populates_result(self):
        pid = os.getpid()
        backend = MockCUDABackend(pid=pid)
        collector = HardwareCollector()
        result = collector.collect(pid=pid, backend=backend)

        assert result.succeeded
        data = result.data
        assert data is not None
        assert data.backend == BackendKind.CUDA
        assert data.gpu_name == "NVIDIA RTX 4090"
        assert data.device_index == 0
        assert data.vram_used_bytes == 21 * 1024**3
        assert data.gpu_utilization_pct == 93
        assert data.driver_version == "575.64"
        assert data.cuda_version == "12.4"
        assert len(data.devices) == 1
        assert data.devices[0].index == 0
        assert data.devices[0].process_vram_bytes == 21 * 1024**3
        assert data.gpu_indices == [0]
        assert data.process_vram_total_bytes == 21 * 1024**3

    def test_no_devices_returns_cpu_result(self):
        pid = os.getpid()
        backend = MockEmptyBackend()
        collector = HardwareCollector()
        result = collector.collect(pid=pid, backend=backend)

        assert result.succeeded
        data = result.data
        assert data is not None
        assert data.backend == BackendKind.CPU
        assert data.gpu_name is None
        assert data.device_index is None

    def test_never_raises(self):
        backend = MockEmptyBackend()
        collector = HardwareCollector()
        result = collector.collect(pid=9_999_999, backend=backend)
        assert result is not None

    def test_collector_name(self):
        assert HardwareCollector.name == "hardware"
