"""Multi-GPU hardware collection — measured attachments only (no fake metrics).

Discovery is injected via a mock HardwareBackend so CI can exercise 1/2/4 GPU
layouts on a machine that may have fewer physical devices. All VRAM numbers
come from that backend's returned DeviceInfo / DeviceProcess values — never
invented by the collector.
"""

from __future__ import annotations

from llm_inspector.backends.base import DeviceInfo, DeviceProcess, HardwareBackend
from llm_inspector.collectors.hardware import HardwareCollector
from llm_inspector.models.enums import BackendKind


class MultiGPUBackend(HardwareBackend):
    """Configurable mock: N devices, PID attached to a subset."""

    kind = BackendKind.CUDA

    def __init__(
        self,
        pid: int,
        n_devices: int,
        attached: list[int] | None = None,
        process_vram: int = 10 * 1024**3,
    ) -> None:
        self._pid = pid
        self._n = n_devices
        self._attached = attached if attached is not None else list(range(n_devices))
        self._process_vram = process_vram

    def is_available(self) -> bool:
        return True

    def list_devices(self) -> list[DeviceInfo]:
        return [
            DeviceInfo(
                index=i,
                name=f"FakeGPU-{i}",
                vram_total_bytes=80 * 1024**3,
                vram_used_bytes=40 * 1024**3,
                gpu_utilization_pct=50 + i,
                driver_version="575.64",
                cuda_version="12.4",
            )
            for i in range(self._n)
        ]

    def device_processes(self, device_index: int) -> list[DeviceProcess]:
        if device_index not in self._attached:
            return []
        return [
            DeviceProcess(
                pid=self._pid,
                device_index=device_index,
                vram_used_bytes=self._process_vram,
            )
        ]

    def driver_version(self) -> str | None:
        return "575.64"

    def runtime_version(self) -> str | None:
        return "12.4"


class TestMultiGPUHardware:
    def test_one_gpu(self) -> None:
        backend = MultiGPUBackend(pid=42, n_devices=1)
        result = HardwareCollector().collect(pid=42, backend=backend)
        assert result.succeeded
        data = result.data
        assert data is not None
        assert data.gpu_indices == [0]
        assert len(data.devices) == 1
        assert data.process_vram_total_bytes == 10 * 1024**3

    def test_two_gpus(self) -> None:
        backend = MultiGPUBackend(pid=42, n_devices=2)
        result = HardwareCollector().collect(pid=42, backend=backend)
        data = result.data
        assert data is not None
        assert data.gpu_indices == [0, 1]
        assert len(data.devices) == 2
        assert data.process_vram_total_bytes == 20 * 1024**3
        assert data.device_index == 0  # primary = first

    def test_four_gpus(self) -> None:
        backend = MultiGPUBackend(pid=7, n_devices=4)
        result = HardwareCollector().collect(pid=7, backend=backend)
        data = result.data
        assert data is not None
        assert data.gpu_indices == [0, 1, 2, 3]
        assert len(data.devices) == 4
        assert data.process_vram_total_bytes == 40 * 1024**3

    def test_uneven_attachment_no_duplicates_or_loss(self) -> None:
        """PID on GPUs 0 and 2 only — never invent GPU 1 or 3."""
        backend = MultiGPUBackend(pid=9, n_devices=4, attached=[0, 2])
        result = HardwareCollector().collect(pid=9, backend=backend)
        data = result.data
        assert data is not None
        assert data.gpu_indices == [0, 2]
        assert [d.index for d in data.devices] == [0, 2]
        assert data.process_vram_total_bytes == 20 * 1024**3

    def test_each_attachment_keeps_measured_process_vram(self) -> None:
        backend = MultiGPUBackend(pid=1, n_devices=2, process_vram=3 * 1024**3)
        data = HardwareCollector().collect(pid=1, backend=backend).data
        assert data is not None
        for d in data.devices:
            assert d.process_vram_bytes == 3 * 1024**3
            assert d.vram_free_bytes == 40 * 1024**3  # 80 - 40 device-level
