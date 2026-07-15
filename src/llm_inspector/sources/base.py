"""Source ABC — contract for all data providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from llm_inspector.inspector.context import InspectionContext


class Source(ABC):
    """
    A data provider for one or more collectors.

    Sources are tried in priority order by SourceResolver.
    """

    name: str = "unknown"

    @abstractmethod
    def can_answer(self, collector: str, ctx: InspectionContext) -> bool:
        """Return True if this source can provide data for *collector*."""

    @abstractmethod
    def collect(self, collector: str, ctx: InspectionContext) -> dict[str, Any] | None:
        """
        Collect data for *collector*.

        Returns a dict suitable for deserializing into the result type,
        or None if collection failed.
        """
