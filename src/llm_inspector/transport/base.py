"""Transport abstraction — how CLI talks to embedded inspectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from llm_inspector.rpc import CollectorRequest, CollectorResponse

# Registry of embedded collector handlers: name → callable(request) → response data dict
EmbeddedHandler = Callable[["CollectorRequest"], dict]


class Transport(ABC):
    """
    Bidirectional transport between embedded runtime and external CLI.

    Embedded side calls ``serve()`` with a handler registry.
    CLI side calls ``fetch()`` with a CollectorRequest.
    """

    @abstractmethod
    def serve(self, handlers: dict[str, EmbeddedHandler]) -> None:
        """Start serving collector RPC requests (embedded side)."""

    @abstractmethod
    def stop(self) -> None:
        """Stop the transport server."""

    @abstractmethod
    def fetch(self, request: CollectorRequest) -> CollectorResponse | None:
        """Fetch a collector result (CLI side). Returns None if unreachable."""
