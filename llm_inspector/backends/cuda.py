"""
CUDABackend — NVIDIA GPU support via pynvml.

This is the primary backend for production Linux servers.
All pynvml calls are isolated here; the rest of the codebase
never imports pynvml directly.
"""

from __future__ import annotations

import contextlib

from llm_inspector.backends.base import DeviceInfo, DeviceProcess, HardwareBackend
from llm_inspector.models.enums import BackendKind


class CUDABackend(HardwareBackend):
    """NVIDIA GPU backend using pynvml."""

    kind = BackendKind.CUDA

    def is_available(self) -> bool:
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            pynvml.nvmlShutdown()
            return True
        except Exception:  # noqa: BLE001
            return False

    def list_devices(self) -> list[DeviceInfo]:
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            try:
                count = pynvml.nvmlDeviceGetCount()
                return [self._read_device(pynvml, i) for i in range(count)]
            finally:
                pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            return []

    def device_processes(self, device_index: int) -> list[DeviceProcess]:
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                return self._read_processes(pynvml, handle, device_index)
            finally:
                pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            return []

    def driver_version(self) -> str | None:
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            try:
                return pynvml.nvmlSystemGetDriverVersion()
            finally:
                pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            return None

    def runtime_version(self) -> str | None:
        """Return CUDA runtime version as a dotted string (e.g. '12.4')."""
        try:
            import pynvml  # noqa: PLC0415

            pynvml.nvmlInit()
            try:
                raw = pynvml.nvmlSystemGetCudaDriverVersion()
                major, minor = divmod(raw, 1000)
                return f"{major}.{minor // 10}"
            finally:
                pynvml.nvmlShutdown()
        except Exception:  # noqa: BLE001
            return None

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _read_device(pynvml: object, index: int) -> DeviceInfo:  # type: ignore[override]
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)  # type: ignore[attr-defined]
        name: str = pynvml.nvmlDeviceGetName(handle)  # type: ignore[attr-defined]
        mem = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[attr-defined]
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[attr-defined]

        driver: str | None = None
        cuda_v: str | None = None
        with contextlib.suppress(Exception):
            driver = pynvml.nvmlSystemGetDriverVersion()  # type: ignore[attr-defined]
        with contextlib.suppress(Exception):
            raw = pynvml.nvmlSystemGetCudaDriverVersion()  # type: ignore[attr-defined]
            major, minor = divmod(raw, 1000)
            cuda_v = f"{major}.{minor // 10}"

        return DeviceInfo(
            index=index,
            name=name,
            vram_total_bytes=mem.total,
            vram_used_bytes=mem.used,
            gpu_utilization_pct=util.gpu,
            driver_version=driver,
            cuda_version=cuda_v,
        )

    @staticmethod
    def _read_processes(
        pynvml: object,  # type: ignore[override]
        handle: object,
        device_index: int,
    ) -> list[DeviceProcess]:
        procs: list[DeviceProcess] = []
        with contextlib.suppress(Exception):
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):  # type: ignore[attr-defined]
                procs.append(
                    DeviceProcess(
                        pid=p.pid,
                        device_index=device_index,
                        vram_used_bytes=p.usedGpuMemory or 0,
                    )
                )
        with contextlib.suppress(Exception):
            for p in pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle):  # type: ignore[attr-defined]
                if not any(existing.pid == p.pid for existing in procs):
                    procs.append(
                        DeviceProcess(
                            pid=p.pid,
                            device_index=device_index,
                            vram_used_bytes=p.usedGpuMemory or 0,
                        )
                    )
        return procs
