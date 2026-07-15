"""InspectionReport — the single output of the Inspector."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from llm_inspector.models.results import (
    HardwareResult,
    MemoryBreakdownResult,
    MemoryResult,
    ModelResult,
    ProcessResult,
    RuntimeResult,
)


class InspectionReport(BaseModel):
    """
    Aggregated result of a full inspection of one inference process.

    Phase A results (process, hardware) are always present — if either
    collector fails the entire inspection aborts.

    Phase B results (memory, model, memory_breakdown, runtime) are
    Optional.  Failure of one Phase B collector does not affect others;
    the error is recorded in collector_errors.
    """

    model_config = ConfigDict(frozen=True)

    captured_at: datetime = Field(default_factory=datetime.utcnow)
    pid: int

    # Phase A — always present
    process: ProcessResult
    hardware: HardwareResult

    # Phase B — Optional; None means the collector failed or was skipped
    memory: MemoryResult | None = None
    model: ModelResult | None = None
    memory_breakdown: MemoryBreakdownResult | None = None
    runtime: RuntimeResult | None = None

    # Audit trail
    collector_errors: dict[str, str] = Field(
        default_factory=dict,
        description="Maps collector_name → error message for failed collectors.",
    )
    elapsed_ms: dict[str, float] = Field(
        default_factory=dict,
        description="Maps collector_name → wall-clock time in milliseconds.",
    )
