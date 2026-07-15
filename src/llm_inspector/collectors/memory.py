"""
MemoryCollector — Phase B.

Reports raw, measurable memory facts about the inference process.
Does NOT explain where memory goes — that is MemoryBreakdownCollector.

In Phase 1, this collector uses only NVML (via the backend) and psutil.
PyTorch allocated/reserved memory is marked Unavailable until the plugin
infrastructure is in place (Phase 2).
"""

from __future__ import annotations

from llm_inspector.collectors.base import Collector, CollectorResult
from llm_inspector.models.enums import CollectorPhase
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import MemoryResult


class MemoryCollector(Collector[MemoryResult]):
    """
    Collect raw memory metrics for a process.

    Phase B — receives a full InspectionContext.
    In Phase 1: reads process RAM from psutil, GPU memory from backend.
    """

    name = "memory"
    phase = CollectorPhase.B

    def collect(self, ctx: object) -> CollectorResult[MemoryResult]:  # type: ignore[override]
        return super().collect(ctx)

    def _collect(self, ctx: object) -> MemoryResult:  # type: ignore[override]
        from llm_inspector.inspector.context import InspectionContext  # noqa: PLC0415

        assert isinstance(ctx, InspectionContext)

        pid = ctx.pid
        backend = ctx.backend

        # ── Process RAM via psutil ────────────────────────────────────────────
        process_ram = self._read_process_ram(pid)

        # ── GPU memory via backend ────────────────────────────────────────────
        gpu_used = Measurement[int].unavailable(
            "No GPU device attached to this process."
        )
        if ctx.hardware.device_index is not None:
            pid_map = backend.pid_to_devices()
            pid_procs = pid_map.get(pid, [])
            if pid_procs:
                total_vram = sum(p.vram_used_bytes for p in pid_procs)
                gpu_used = Measurement[int].available(
                    total_vram,
                    source=f"NVML nvmlDeviceGetComputeRunningProcesses() — GPU {pid_procs[0].device_index}",
                )

        # PyTorch stats require plugin — unavailable in Phase 1
        unavail_torch = Measurement[int].unavailable(
            "PyTorch memory stats require runtime plugin (Phase 2)."
        )

        return MemoryResult(
            process_ram=process_ram,
            gpu_used=gpu_used,
            gpu_allocated=unavail_torch,
            gpu_reserved=unavail_torch,
            peak=unavail_torch,
        )

    @staticmethod
    def _read_process_ram(pid: int) -> Measurement[int]:
        try:
            import psutil  # noqa: PLC0415

            proc = psutil.Process(pid)
            rss = proc.memory_info().rss
            return Measurement[int].available(rss, source="psutil.Process.memory_info().rss")
        except Exception as exc:  # noqa: BLE001
            return Measurement[int].unavailable(f"psutil error: {exc}")
