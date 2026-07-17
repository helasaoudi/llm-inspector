"""Optimization strategy contract — pluggable projections."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_inspector.models.enums import RuntimeKind
from llm_inspector.models.report import InspectionReport
from llm_inspector.optimization.models import OptimizationGroup, OptimizationKind


@dataclass(frozen=True)
class OptimizationContext:
    """Measured inputs extracted once from InspectionReport."""

    report: InspectionReport
    runtime: RuntimeKind
    param_count: int | None
    precision: str | None
    weights_bytes: int | None
    kv_bytes: int | None
    workspace_bytes: int | None
    other_bytes: int | None
    activations_bytes: int | None
    current_total: int | None
    vram_total: int | None


class OptimizationStrategy(ABC):
    """
    One family of optimizations (Quantization, KV Cache, …).

    Strategies produce an OptimizationGroup of projected scenarios.
    They never mutate the model or the InspectionReport.
    """

    kind: OptimizationKind
    title: str

    @abstractmethod
    def analyze(self, ctx: OptimizationContext) -> OptimizationGroup:
        """Return projected scenarios for this strategy family."""
