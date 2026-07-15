"""
Collector ABC and CollectorResult — the shared contract for all collectors.

Every collector in LLM Inspector implements this interface.
"""

from __future__ import annotations

import time
import traceback
from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel, ConfigDict

from llm_inspector.models.enums import CollectorPhase, CollectorStatus

T = TypeVar("T")


class CollectorResult(BaseModel, Generic[T]):
    """
    The return type of every collector.collect() call.

    A collector NEVER raises.  If something goes wrong, it returns a
    CollectorResult with status=FAILED and the error message in ``error``.

    Attributes:
        collector_name: Identifies which collector produced this result.
        status:         SUCCESS, PARTIAL, or FAILED.
        data:           The typed result.  None only when status=FAILED.
        error:          Error message when status=FAILED.
        warnings:       Non-fatal issues when status=PARTIAL.
        elapsed_ms:     Wall-clock time for this collector in milliseconds.
    """

    model_config = ConfigDict(frozen=True)

    collector_name: str
    status: CollectorStatus
    data: T | None = None
    error: str | None = None
    warnings: list[str] = []
    elapsed_ms: float = 0.0

    @property
    def succeeded(self) -> bool:
        return self.status != CollectorStatus.FAILED

    @property
    def failed(self) -> bool:
        return self.status == CollectorStatus.FAILED


class Collector(ABC, Generic[T]):
    """
    Abstract base class for all LLM Inspector collectors.

    Subclasses must:
      1. Declare ``name`` (used in --collect flag and error reporting).
      2. Declare ``phase`` (Phase.A or Phase.B).
      3. Implement ``_collect()`` which does the actual work.

    The public ``collect()`` method wraps ``_collect()`` with timing,
    error handling, and CollectorResult construction.  Subclasses never
    override ``collect()`` directly.
    """

    name: str
    phase: CollectorPhase

    def collect(self, *args: object, **kwargs: object) -> CollectorResult[T]:
        """
        Public entry point.  Times _collect() and catches all exceptions.

        Returns a CollectorResult.  Never raises.
        """
        start = time.perf_counter()
        try:
            result = self._collect(*args, **kwargs)
            elapsed = (time.perf_counter() - start) * 1000
            return CollectorResult(
                collector_name=self.name,
                status=CollectorStatus.SUCCESS,
                data=result,
                elapsed_ms=elapsed,
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = (time.perf_counter() - start) * 1000
            return CollectorResult(
                collector_name=self.name,
                status=CollectorStatus.FAILED,
                data=None,
                error=f"{type(exc).__name__}: {exc}",
                elapsed_ms=elapsed,
            )

    @abstractmethod
    def _collect(self, *args: object, **kwargs: object) -> T:
        """
        Perform the actual data collection.

        Implementations should raise on unrecoverable errors — the base
        class collect() method converts them to CollectorResult(FAILED).
        For partial data, return the data and mark fields unavailable
        via Measurement.unavailable().
        """
