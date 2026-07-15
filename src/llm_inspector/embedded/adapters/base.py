"""
Model adapters — framework-specific hooks for embedded inspection.

Each adapter knows how to extract model identity and memory from its
runtime.  New frameworks (AirLLM, LangChain, etc.) add an adapter here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from llm_inspector.models.measurement import Measurement


@dataclass
class AdapterContext:
    """Shared state passed to embedded collectors."""

    model: Any | None = None
    engine: Any | None = None
    runtime_kind: str = "pytorch"
    options: dict[str, Any] = field(default_factory=dict)


class ModelAdapter(ABC):
    """
    Framework adapter contract.

    Implementations: PyTorchAdapter (generic), HuggingFaceAdapter, VLLMAdapter.
    """

    name: str = "unknown"

    @abstractmethod
    def supports(self, obj: Any) -> bool:
        """Return True if this adapter can handle *obj*."""

    @abstractmethod
    def bind(self, obj: Any, engine: Any | None = None) -> AdapterContext:
        """Create an AdapterContext from a model or engine object."""

    def model_name(self, ctx: AdapterContext) -> Measurement[str]:
        return Measurement[str].unavailable("Model name not exposed by adapter.")

    def parameter_count(self, ctx: AdapterContext) -> Measurement[int]:
        return Measurement[int].unavailable("Parameter count not exposed by adapter.")

    def precision(self, ctx: AdapterContext) -> Measurement[str]:
        return Measurement[str].unavailable("Precision not exposed by adapter.")

    def weights_bytes(self, ctx: AdapterContext) -> Measurement[int]:
        return Measurement[int].unavailable("Weights size not exposed by adapter.")
