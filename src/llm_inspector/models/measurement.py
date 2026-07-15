"""
Measurement[T] — the provenance-aware value type used by all Phase B collectors.

Philosophy
----------
Every value displayed by LLM Inspector was either measured directly from the
runtime or declared unavailable.  There is no third option.

``DataSource.ESTIMATED`` does not exist in this codebase.  This is not an
oversight — it is an architectural guarantee.

Usage
-----
::

    # Collector found the value:
    m = Measurement.available(17_000_000_000, source="torch.named_parameters()")

    # Runtime does not expose this:
    m = Measurement.unavailable("vLLM does not expose activation memory.")

    # Reading the value safely:
    if m.is_available:
        print(f"{m.value / 1e9:.1f} GB")
    else:
        print("Unavailable")
"""

from __future__ import annotations

from enum import StrEnum
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

T = TypeVar("T")


class MeasurementStatus(StrEnum):
    """
    The provenance status of a single observed value.

    ``ESTIMATED`` is deliberately absent — LLM Inspector never estimates.
    """

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


class Measurement(BaseModel, Generic[T]):
    """
    A single observed value with full provenance.

    Attributes:
        value:  The measured value, or ``None`` when status is UNAVAILABLE.
        status: Whether the value was successfully measured.
        source: Human-readable description of the data origin.
                E.g. ``"NVML nvmlDeviceGetMemoryInfo()"`` or
                ``"vLLM /metrics endpoint"``.
                Always set when status is AVAILABLE.
        reason: Why the value is unavailable.
                Always set when status is UNAVAILABLE.
    """

    model_config = ConfigDict(frozen=True)

    value: T | None = None
    status: MeasurementStatus = MeasurementStatus.UNAVAILABLE
    source: str | None = None
    reason: str | None = None

    # ── Factory constructors ──────────────────────────────────────────────────

    @classmethod
    def available(cls, value: T, source: str) -> "Measurement[T]":
        """Create a successfully measured value with its data origin."""
        return cls(value=value, status=MeasurementStatus.AVAILABLE, source=source)

    @classmethod
    def unavailable(cls, reason: str) -> "Measurement[T]":
        """Create an unavailable measurement with a human-readable reason."""
        return cls(value=None, status=MeasurementStatus.UNAVAILABLE, reason=reason)

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def is_available(self) -> bool:
        """Return True if the value was successfully measured."""
        return self.status == MeasurementStatus.AVAILABLE
