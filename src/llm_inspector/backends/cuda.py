"""
CUDABackend — NVIDIA GPU support via pynvml.

All pynvml calls are isolated here; the rest of the codebase
never imports pynvml directly.

Grace Blackwell / unified-memory GPUs (GB10, GB200, etc.) report
"Not Supported" for nvmlDeviceGetMemoryInfo(). Every device-level
query is wrapped individually so one unsupported call never kills
the whole device listing.
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
                devices = []
                for i in range(count):
                    with contextlib.suppress(Exception):
                        devices.append(self._read_device(pynvml, i))
                return devices
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

        # Name — always available
        name: str = pynvml.nvmlDeviceGetName(handle)  # type: ignore[attr-defined]

        # Memory — not supported on unified-memory GPUs (GB10, GB200)
        vram_total: int | None = None
        vram_used: int | None = None
        with contextlib.suppress(Exception):
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)  # type: ignore[attr-defined]
            vram_total = mem.total
            vram_used = mem.used

        # Utilization — may also be unsupported
        gpu_util: int | None = None
        with contextlib.suppress(Exception):
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)  # type: ignore[attr-defined]
            gpu_util = util.gpu

        # Driver / CUDA versions — system-wide, should always work
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
            vram_total_bytes=vram_total,  # None on unified-memory GPUs
            vram_used_bytes=vram_used,    # None on unified-memory GPUs
            gpu_utilization_pct=gpu_util, # None if unsupported
            driver_version=driver,
            cuda_version=cuda_v,
        )

    @staticmethod
    def _read_processes(
        pynvml: object,  # type: ignore[override]
        handle: object,
        device_index: int,
    ) -> list[DeviceProcess]:
        """
        Return only *compute* processes on this device.

        Graphics processes (Xorg, gnome-shell, plasmashell…) are
        deliberately excluded — LLM inference is always a compute
        workload (Type C in nvidia-smi). Including display-server
        PIDs would pollute ``llminspect ps`` with irrelevant rows.
        """
        procs: list[DeviceProcess] = []
        with contextlib.suppress(Exception):
            for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):  # type: ignore[attr-defined]
                # usedGpuMemory may be None on unified-memory GPUs (GB10)
                mem = getattr(p, "usedGpuMemory", None) or 0
                procs.append(DeviceProcess(pid=p.pid, device_index=device_index, vram_used_bytes=mem))
        return procs
