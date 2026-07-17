"""Data models for Optimization Analysis (projected, not measured)."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from llm_inspector.models.measurement import Measurement


class QualityBand(StrEnum):
    """Typical quality impact declared by a strategy (metadata, not measured)."""

    EXCELLENT = "Excellent"
    VERY_GOOD = "Very Good"
    GOOD = "Good"
    FAIR = "Fair"
    UNKNOWN = "Unknown"


class OptimizationKind(StrEnum):
    """Category of optimization strategy (extensible)."""

    QUANTIZATION = "Quantization"
    KV_CACHE = "KV Cache"
    PARALLELISM = "Parallelism"
    CONTEXT = "Context"
    OTHER = "Other"


class OptimizationScenario(BaseModel):
    """One projected outcome for a single optimization strategy/method."""

    model_config = ConfigDict(frozen=True)

    kind: OptimizationKind
    method_name: str
    description: str = ""
    quality: QualityBand = QualityBand.UNKNOWN
    new_total: Measurement[int] = Field(
        default_factory=lambda: Measurement.unavailable("Not projected"),
    )
    saved_bytes: Measurement[int] = Field(
        default_factory=lambda: Measurement.unavailable("Not projected"),
    )
    details: dict[str, Measurement[int]] = Field(
        default_factory=dict,
        description="Optional projected component sizes (e.g. weights).",
    )


class OptimizationRecommendation(BaseModel):
    model_config = ConfigDict(frozen=True)

    summary: str
    warnings: list[str] = Field(default_factory=list)
    suggested_method: str | None = None
    bottleneck: str | None = None


class OptimizationGroup(BaseModel):
    """One strategy family under Optimization Analysis (e.g. Quantization)."""

    model_config = ConfigDict(frozen=True)

    kind: OptimizationKind
    title: str
    scenarios: list[OptimizationScenario] = Field(default_factory=list)
    note: str | None = None


class OptimizationAnalysis(BaseModel):
    """
    Continuation of InspectionReport — projected optimizations only.

    Never mutates measured sections.
    """

    model_config = ConfigDict(frozen=True)

    groups: list[OptimizationGroup] = Field(default_factory=list)
    recommendation: OptimizationRecommendation | None = None
    skipped_reason: str | None = None
