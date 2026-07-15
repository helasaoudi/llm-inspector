"""
RPC wire protocol for embedded ↔ external inspection.

The CLI requests a specific collector; the embedded runtime computes only
what was asked.  Responses reuse existing Pydantic result types.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = 1


class CollectorRequest(BaseModel):
    """Request sent from CLI to an embedded inspector."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    collector: str
    options: dict[str, Any] = Field(default_factory=dict)


class CollectorResponse(BaseModel):
    """Response from embedded inspector."""

    model_config = ConfigDict(frozen=True)

    schema_version: int = SCHEMA_VERSION
    collector: str
    data: dict[str, Any] | None = None
    elapsed_ms: float = 0.0
    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None and self.data is not None
