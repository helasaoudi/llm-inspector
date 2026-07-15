"""
CollectorPipeline — runs Phase A sequentially, Phase B in parallel.

The pipeline is the only place that knows about collector ordering.
Collectors themselves are unaware of each other.
"""

from __future__ import annotations

import concurrent.futures
from typing import TYPE_CHECKING

from llm_inspector.collectors.base import CollectorResult
from llm_inspector.collectors.hardware import HardwareCollector
from llm_inspector.collectors.memory import MemoryCollector
from llm_inspector.collectors.memory_breakdown import MemoryBreakdownCollector
from llm_inspector.collectors.model import ModelCollector
from llm_inspector.collectors.process import ProcessCollector
from llm_inspector.collectors.runtime import RuntimeCollector
from llm_inspector.models.enums import CollectorStatus
from llm_inspector.models.results import (
    HardwareResult,
    MemoryBreakdownResult,
    MemoryResult,
    ModelResult,
    ProcessResult,
    RuntimeResult,
)

if TYPE_CHECKING:
    from llm_inspector.backends.base import HardwareBackend
    from llm_inspector.inspector.context import InspectionContext

# Collector name constants — match the --collect CLI flag values
COLLECT_ALL = "all"
COLLECT_MEMORY = "memory"
COLLECT_MEMORY_BREAKDOWN = "memory-breakdown"
COLLECT_RUNTIME = "runtime"
COLLECT_MODEL = "model"

# All valid collector names (used for --collect validation)
VALID_COLLECT_NAMES = frozenset([
    COLLECT_ALL,
    COLLECT_MEMORY,
    COLLECT_MEMORY_BREAKDOWN,
    COLLECT_RUNTIME,
    COLLECT_MODEL,
])


class PhaseAResult:
    """Holds the output of Phase A — used to build InspectionContext."""

    def __init__(
        self,
        process_result: CollectorResult[ProcessResult],
        hardware_result: CollectorResult[HardwareResult],
    ) -> None:
        self.process = process_result
        self.hardware = hardware_result

    @property
    def succeeded(self) -> bool:
        return self.process.succeeded and self.hardware.succeeded


class PhaseBResult:
    """Holds all Phase B collector results."""

    def __init__(self) -> None:
        self.memory: CollectorResult[MemoryResult] | None = None
        self.model: CollectorResult[ModelResult] | None = None
        self.memory_breakdown: CollectorResult[MemoryBreakdownResult] | None = None
        self.runtime: CollectorResult[RuntimeResult] | None = None
        self.errors: dict[str, str] = {}
        self.elapsed: dict[str, float] = {}

    def record(self, result: CollectorResult[object]) -> None:
        if result.failed and result.error:
            self.errors[result.collector_name] = result.error
        self.elapsed[result.collector_name] = result.elapsed_ms


class CollectorPipeline:
    """
    Runs collectors in two phases and returns typed results.

    Phase A: ProcessCollector → HardwareCollector (sequential).
    Phase B: MemoryCollector, ModelCollector, MemoryBreakdownCollector,
             RuntimeCollector (parallel, filtered by collect_filter).
    """

    def __init__(self) -> None:
        self._process_collector = ProcessCollector()
        self._hardware_collector = HardwareCollector()
        self._memory_collector = MemoryCollector()
        self._model_collector = ModelCollector()
        self._memory_breakdown_collector = MemoryBreakdownCollector()
        self._runtime_collector = RuntimeCollector()

    def run_phase_a(
        self,
        pid: int,
        backend: "HardwareBackend",
    ) -> PhaseAResult:
        """
        Run Phase A collectors sequentially.

        Results are needed to build InspectionContext before Phase B starts.
        Returns PhaseAResult.  Caller must check .succeeded before proceeding.
        """
        process_result = self._process_collector.collect(pid)
        hardware_result = self._hardware_collector.collect(pid, backend)
        return PhaseAResult(process_result, hardware_result)

    def run_phase_b(
        self,
        ctx: "InspectionContext",
        collect_filter: str = COLLECT_ALL,
    ) -> PhaseBResult:
        """
        Run Phase B collectors in parallel, filtered by *collect_filter*.

        Args:
            ctx:            Frozen InspectionContext from Phase A.
            collect_filter: Collector name to run (``"all"`` runs everything).
        """
        phase_b = PhaseBResult()

        collector_map = {
            COLLECT_MEMORY: self._memory_collector,
            COLLECT_MODEL: self._model_collector,
            COLLECT_MEMORY_BREAKDOWN: self._memory_breakdown_collector,
            COLLECT_RUNTIME: self._runtime_collector,
        }

        # Determine which collectors to run
        if collect_filter == COLLECT_ALL:
            tasks = dict(collector_map)
        elif collect_filter in collector_map:
            tasks = {collect_filter: collector_map[collect_filter]}
        else:
            return phase_b  # unknown filter — return empty

        # Run all selected collectors in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=max(len(tasks), 1)) as pool:
            futures = {
                pool.submit(collector.collect, ctx): name  # type: ignore[call-arg]
                for name, collector in tasks.items()
            }
            for future in concurrent.futures.as_completed(futures):
                name = futures[future]
                try:
                    result = future.result()
                    phase_b.record(result)
                    if name == COLLECT_MEMORY:
                        phase_b.memory = result  # type: ignore[assignment]
                    elif name == COLLECT_MODEL:
                        phase_b.model = result  # type: ignore[assignment]
                    elif name == COLLECT_MEMORY_BREAKDOWN:
                        phase_b.memory_breakdown = result  # type: ignore[assignment]
                    elif name == COLLECT_RUNTIME:
                        phase_b.runtime = result  # type: ignore[assignment]
                except Exception as exc:  # noqa: BLE001
                    phase_b.errors[name] = f"{type(exc).__name__}: {exc}"

        return phase_b
