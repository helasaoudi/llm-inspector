"""
Inspector — the central orchestrator of LLM Inspector.

This module coordinates:
  - BackendRegistry (hardware detection)
  - CollectorPipeline (Phase A + Phase B)
  - PluginRegistry (runtime plugin selection)
  - InspectionContext (shared frozen input for Phase B)
  - InspectionReport (final aggregated output)

The Inspector contains orchestration logic only.
It has no I/O of its own — all data comes from collectors and backends.
"""

from __future__ import annotations

from llm_inspector.backends.base import HardwareBackend
from llm_inspector.backends.registry import BackendRegistry
from llm_inspector.inspector.context import InspectionContext
from llm_inspector.inspector.pipeline import COLLECT_ALL, VALID_COLLECT_NAMES, CollectorPipeline
from llm_inspector.models.report import InspectionReport
from llm_inspector.models.results import HardwareResult, ProcessResult
from llm_inspector.plugins.registry import PluginRegistry, build_default_registry


class Inspector:
    """
    Orchestrates the full inspection of one LLM inference process.

    Usage::

        inspector = Inspector()
        report = inspector.inspect(pid=28431)
        reports = inspector.scan()          # all GPU-attached processes
    """

    def __init__(
        self,
        backend_registry: BackendRegistry | None = None,
        plugin_registry: PluginRegistry | None = None,
        pipeline: CollectorPipeline | None = None,
    ) -> None:
        self._backend_registry = backend_registry or BackendRegistry()
        self._plugin_registry = plugin_registry or build_default_registry()
        self._pipeline = pipeline or CollectorPipeline()

    # ── Public API ────────────────────────────────────────────────────────────

    def inspect(
        self,
        pid: int,
        collect_filter: str = COLLECT_ALL,
    ) -> InspectionReport:
        """
        Run a full inspection of the process identified by *pid*.

        Args:
            pid:            Target process ID.
            collect_filter: Limit Phase B to one collector name,
                            or ``"all"`` to run everything.

        Returns:
            InspectionReport with all available data populated.

        Raises:
            RuntimeError: If Phase A fails (process not found, access denied).
        """
        backend = self._backend_registry.best_available()

        # ── Phase A: build the context ────────────────────────────────────────
        phase_a = self._pipeline.run_phase_a(pid, backend)

        if not phase_a.succeeded:
            errors = []
            if phase_a.process.failed:
                errors.append(f"ProcessCollector: {phase_a.process.error}")
            if phase_a.hardware.failed:
                errors.append(f"HardwareCollector: {phase_a.hardware.error}")
            raise RuntimeError(
                f"Phase A failed for PID {pid}. " + " | ".join(errors)
            )

        process: ProcessResult = phase_a.process.data  # type: ignore[assignment]
        hardware: HardwareResult = phase_a.hardware.data  # type: ignore[assignment]

        # ── Plugin selection ──────────────────────────────────────────────────
        plugin = self._plugin_registry.select(process.runtime_kind)

        # ── Build frozen context ──────────────────────────────────────────────
        ctx = InspectionContext(
            pid=pid,
            process=process,
            hardware=hardware,
            plugin=plugin,
            backend=backend,
        )

        # ── Phase B: parallel collectors ──────────────────────────────────────
        phase_b = self._pipeline.run_phase_b(ctx, collect_filter)

        # ── Assemble report ───────────────────────────────────────────────────
        errors: dict[str, str] = {}
        elapsed: dict[str, float] = {
            phase_a.process.collector_name: phase_a.process.elapsed_ms,
            phase_a.hardware.collector_name: phase_a.hardware.elapsed_ms,
        }

        errors.update(phase_b.errors)
        elapsed.update(phase_b.elapsed)

        return InspectionReport(
            pid=pid,
            process=process,
            hardware=hardware,
            memory=phase_b.memory.data if phase_b.memory and phase_b.memory.succeeded else None,
            model=phase_b.model.data if phase_b.model and phase_b.model.succeeded else None,
            memory_breakdown=(
                phase_b.memory_breakdown.data
                if phase_b.memory_breakdown and phase_b.memory_breakdown.succeeded
                else None
            ),
            runtime=phase_b.runtime.data if phase_b.runtime and phase_b.runtime.succeeded else None,
            collector_errors=errors,
            elapsed_ms=elapsed,
        )

    def scan(self) -> list[InspectionReport]:
        """
        Inspect all LLM processes visible on this machine.

        Strategy:
        1. GPU-attached processes via the hardware backend (NVML on CUDA).
           All GPU-attached processes are shown regardless of runtime —
           GPU attachment is itself the signal of an ML workload.
        2. Additionally merge in any heuristic-detected processes (Ollama,
           vLLM, HF) that may not have GPU attachment (CPU inference).
        3. If no GPU backend available (macOS), heuristic scan only.

        Used by ``llminspect ps``.  Silently skips processes that fail
        Phase A (exited, access denied).
        """
        backend = self._backend_registry.best_available()
        pid_map = backend.pid_to_devices()

        # Always add heuristic-detected processes not already in the GPU map.
        # This catches: Ollama on macOS, CPU inference, processes that appear
        # in cmdline scan but not in NVML (e.g. launcher processes).
        for pid, devices in self._heuristic_scan().items():
            if pid not in pid_map:
                pid_map[pid] = devices

        reports: list[InspectionReport] = []
        for pid in pid_map:
            try:
                report = self.inspect(pid, collect_filter=COLLECT_ALL)
                reports.append(report)
            except RuntimeError:
                # Process may have exited between scan and inspect
                continue

        return sorted(reports, key=lambda r: r.pid)

    def _heuristic_scan(self) -> dict[int, list]:
        """
        Scan all running processes for known LLM runtime signatures.

        Returns a pid_map-compatible dict (pid → empty list, since there
        are no device processes on a CPU-only machine).
        Used as a fallback when no GPU-attached processes are found.
        """
        from llm_inspector.collectors.process import detect_runtime  # noqa: PLC0415
        from llm_inspector.models.enums import RuntimeKind  # noqa: PLC0415
        from llm_inspector.utils import proc as proc_utils  # noqa: PLC0415

        found: dict[int, list] = {}
        for pid in proc_utils.list_all_pids():
            try:
                cmdline = proc_utils.read_cmdline(pid)
                if not cmdline:
                    continue
                if detect_runtime(cmdline) != RuntimeKind.UNKNOWN:
                    found[pid] = []
            except Exception:  # noqa: BLE001
                continue
        return found

    def available_backend(self) -> HardwareBackend:
        """Return the backend that will be used for the next inspection."""
        return self._backend_registry.best_available()

    def registered_plugins(self) -> list[object]:
        """Return all registered plugins (for llminspect runtimes)."""
        return self._plugin_registry.all_plugins()
