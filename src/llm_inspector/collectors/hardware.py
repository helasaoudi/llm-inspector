"""
HardwareCollector — Phase A.

Describes the hardware where inference is running.
Uses the HardwareBackend abstraction — never calls pynvml directly.

GPU fields are populated when a CUDA/ROCm backend is active.
CPU/system fields are always populated via psutil — so the Hardware
section always shows real data regardless of backend.
"""

from __future__ import annotations

import platform

from llm_inspector.backends.base import HardwareBackend
from llm_inspector.collectors.base import Collector, CollectorResult
from llm_inspector.models.enums import CollectorPhase
from llm_inspector.models.results import HardwareResult


def _collect_cpu_info() -> dict:
    """Gather CPU and system RAM info via psutil. Returns empty dict on failure."""
    try:
        import psutil  # noqa: PLC0415

        vm = psutil.virtual_memory()
        return {
            "cpu_name": _cpu_friendly_name(),
            "cpu_physical_cores": psutil.cpu_count(logical=False),
            "cpu_logical_cores": psutil.cpu_count(logical=True),
            # interval=0.1 blocks briefly for a real live reading.
            # interval=None returns 0.0 on first call (stale kernel delta).
            "cpu_utilization_pct": psutil.cpu_percent(interval=0.1),
            "system_ram_total_bytes": vm.total,
            "system_ram_used_bytes": vm.used,
        }
    except Exception:  # noqa: BLE001
        return {}


def _cpu_friendly_name() -> str | None:
    """
    Return a human-readable CPU name.

    On macOS with Apple Silicon, ``platform.processor()`` returns "arm"
    which is useless.  Use ``sysctl hw.model`` to get e.g. "Apple M3 Pro".
    Falls back to ``platform.processor()`` on Linux/Windows.
    """
    if platform.system() == "Darwin":
        try:
            import subprocess  # noqa: PLC0415

            out = (
                subprocess.check_output(
                    ["sysctl", "-n", "machdep.cpu.brand_string"],
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                .decode()
                .strip()
            )
            if out:
                return out
            # Apple Silicon: brand_string is empty, use hw.model instead
            out = (
                subprocess.check_output(
                    ["sysctl", "-n", "hw.model"],
                    stderr=subprocess.DEVNULL,
                    timeout=2,
                )
                .decode()
                .strip()
            )
            return out or None
        except Exception:  # noqa: BLE001
            pass
    return platform.processor() or platform.machine() or None


class HardwareCollector(Collector[HardwareResult]):
    """
    Read hardware metrics for a given PID via the active backend.

    Phase A — runs before InspectionContext is built.
    Inputs: pid (int), backend (HardwareBackend).
    """

    name = "hardware"
    phase = CollectorPhase.A

    def collect(  # type: ignore[override]
        self,
        pid: int,
        backend: HardwareBackend,
    ) -> CollectorResult[HardwareResult]:
        return super().collect(pid, backend)

    def _collect(  # type: ignore[override]
        self,
        pid: int,
        backend: HardwareBackend,
    ) -> HardwareResult:
        from llm_inspector.models.results import GpuAttachment  # noqa: PLC0415

        cpu_info = _collect_cpu_info()
        devices = backend.list_devices()

        pid_map = backend.pid_to_devices()
        pid_procs = pid_map.get(pid, [])

        if not pid_procs or not devices:
            return HardwareResult(
                backend=backend.kind,
                driver_version=backend.driver_version(),
                cuda_version=backend.runtime_version(),
                **cpu_info,
            )

        # Every GPU this PID is attached to (NVML-measured).
        device_by_index = {d.index: d for d in devices}
        attachments: list[GpuAttachment] = []
        for proc in sorted(pid_procs, key=lambda p: p.device_index):
            device = device_by_index.get(proc.device_index)
            if device is None:
                continue
            attachments.append(
                GpuAttachment(
                    index=device.index,
                    name=device.name,
                    vram_total_bytes=device.vram_total_bytes,
                    vram_used_bytes=device.vram_used_bytes,
                    gpu_utilization_pct=device.gpu_utilization_pct,
                    process_vram_bytes=proc.vram_used_bytes,
                )
            )

        if not attachments:
            return HardwareResult(backend=backend.kind, **cpu_info)

        primary = attachments[0]
        return HardwareResult(
            backend=backend.kind,
            device_index=primary.index,
            gpu_name=primary.name,
            vram_total_bytes=primary.vram_total_bytes,
            # Primary VRAM used = this PID on the primary GPU (measured)
            vram_used_bytes=primary.process_vram_bytes
            if primary.process_vram_bytes is not None
            else primary.vram_used_bytes,
            gpu_utilization_pct=primary.gpu_utilization_pct,
            driver_version=backend.driver_version()
            or device_by_index[primary.index].driver_version,
            cuda_version=backend.runtime_version() or device_by_index[primary.index].cuda_version,
            devices=attachments,
            **cpu_info,
        )
