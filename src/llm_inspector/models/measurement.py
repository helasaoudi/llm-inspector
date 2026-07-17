"""
Measurement[T] — the provenance-aware value type used by Phase B collectors
and optimization projections.

Philosophy
----------
Inspection values are either **measured** (AVAILABLE) or UNAVAILABLE.

Optimization Analysis may attach **SIMULATED** projections.  Those never
appear as measured facts in Process / Hardware / Model / Memory sections.
``ESTIMATED`` (guessing without a formula tied to measured inputs) does
not exist.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class MeasurementStatus(StrEnum):
    """
    Provenance status of a single value.

    - AVAILABLE:  measured from a live source
    - UNAVAILABLE: not exposed / not measurable
    - SIMULATED:  projected by Optimization Analysis from measured inputs
    """

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    SIMULATED = "simulated"


class Measurement(BaseModel, Generic[T]):
    """
    A single value with full provenance.

    Attributes:
        value:  The value, or ``None`` when status is UNAVAILABLE.
        status: Provenance kind.
        source: Origin (measured API) or projection formula (simulated).
        reason: Why unavailable (UNAVAILABLE only).
    """

    model_config = ConfigDict(frozen=True)

    value: T | None = None
    status: MeasurementStatus = MeasurementStatus.UNAVAILABLE
    source: str | None = None
    reason: str | None = None

    @classmethod
    def available(cls, value: T, source: str) -> "Measurement[T]":
        """Successfully measured value with its data origin."""
        return cls(value=value, status=MeasurementStatus.AVAILABLE, source=source)

    @classmethod
    def simulated(cls, value: T, source: str) -> "Measurement[T]":
        """Projected value for Optimization Analysis (not a measured fact)."""
        return cls(value=value, status=MeasurementStatus.SIMULATED, source=source)

    @classmethod
    def unavailable(cls, reason: str) -> "Measurement[T]":
        """Unavailable with a human-readable reason."""
        return cls(value=None, status=MeasurementStatus.UNAVAILABLE, reason=reason)

    @property
    def is_available(self) -> bool:
        """True if measured from a live source."""
        return self.status == MeasurementStatus.AVAILABLE

    @property
    def is_simulated(self) -> bool:
        """True if this is an optimization projection."""
        return self.status == MeasurementStatus.SIMULATED

    @property
    def has_value(self) -> bool:
        """True if a numeric/string value is present (measured or simulated)."""
        return (
            self.status
            in (MeasurementStatus.AVAILABLE, MeasurementStatus.SIMULATED)
            and self.value is not None
        )
