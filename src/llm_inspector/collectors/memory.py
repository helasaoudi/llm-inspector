"""
MemoryCollector — Phase B.

Reports raw, measurable memory facts about the inference process.

Data resolution:
  - process_ram, gpu_used: always external (psutil, NVML)
  - gpu_allocated, gpu_reserved, peak: EmbeddedSource if attached,
    otherwise Unavailable with honest reason
  - devices: per-GPU torch metrics only when attach reports them
"""

from __future__ import annotations

from llm_inspector.collectors.base import Collector, CollectorResult
from llm_inspector.models.enums import CollectorPhase
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import MemoryResult
from llm_inspector.serialization import model_from_dict
from llm_inspector.sources.resolver import get_resolver


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

        # ── GPU memory via backend (sum across all attached GPUs) ─────────────
        gpu_used = Measurement[int].unavailable("No GPU device attached to this process.")
        if ctx.hardware.devices:
            pid_map = backend.pid_to_devices()
            pid_procs = pid_map.get(pid, [])
            if pid_procs:
                total_vram = sum(p.vram_used_bytes for p in pid_procs)
                gpu_ids = ",".join(
                    str(p.device_index) for p in sorted(pid_procs, key=lambda p: p.device_index)
                )
                gpu_used = Measurement[int].available(
                    total_vram,
                    source=(
                        "NVML nvmlDeviceGetComputeRunningProcesses() — "
                        f"sum of process VRAM on GPU(s) {gpu_ids}"
                    ),
                )
        elif ctx.hardware.device_index is not None:
            # Compat: primary-only path if devices list empty
            pid_map = backend.pid_to_devices()
            pid_procs = pid_map.get(pid, [])
            if pid_procs:
                total_vram = sum(p.vram_used_bytes for p in pid_procs)
                gpu_used = Measurement[int].available(
                    total_vram,
                    source=(
                        "NVML nvmlDeviceGetComputeRunningProcesses() — "
                        f"GPU {pid_procs[0].device_index}"
                    ),
                )

        # GPU Allocated / Reserved / Peak — try embedded source first
        gpu_allocated = Measurement[int].unavailable(
            "GPU allocated requires embedded inspector. "
            "Add: from llm_inspector import attach; attach(model=...)"
        )
        gpu_reserved = Measurement[int].unavailable(
            "GPU reserved requires embedded inspector. "
            "Add: from llm_inspector import attach; attach(model=...)"
        )
        peak = Measurement[int].unavailable(
            "Peak GPU memory requires embedded inspector with streaming counters."
        )
        device_memories: list = []

        embedded_data = get_resolver().fetch("memory", ctx)
        if embedded_data is not None:
            embedded = model_from_dict(MemoryResult, embedded_data)
            if embedded.gpu_allocated.is_available:
                gpu_allocated = embedded.gpu_allocated
            if embedded.gpu_reserved.is_available:
                gpu_reserved = embedded.gpu_reserved
            if embedded.peak.is_available:
                peak = embedded.peak
            device_memories = list(embedded.devices)

        return MemoryResult(
            process_ram=process_ram,
            gpu_used=gpu_used,
            gpu_allocated=gpu_allocated,
            gpu_reserved=gpu_reserved,
            peak=peak,
            devices=device_memories,
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
