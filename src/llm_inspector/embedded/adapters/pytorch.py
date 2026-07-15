"""Generic PyTorch nn.Module adapter."""

from __future__ import annotations

from typing import Any

from llm_inspector.embedded.adapters.base import AdapterContext, ModelAdapter
from llm_inspector.models.measurement import Measurement


class PyTorchAdapter(ModelAdapter):
    """Works with any ``torch.nn.Module`` — HF, custom FastAPI servers, etc."""

    name = "pytorch"

    def supports(self, obj: Any) -> bool:
        try:
            import torch.nn as nn  # noqa: PLC0415

            return isinstance(obj, nn.Module)
        except ImportError:
            return False

    def bind(self, obj: Any, engine: Any | None = None) -> AdapterContext:
        return AdapterContext(model=obj, engine=engine, runtime_kind="pytorch")

    def parameter_count(self, ctx: AdapterContext) -> Measurement[int]:
        if ctx.model is None:
            return Measurement[int].unavailable("No model bound.")
        try:
            count = sum(p.numel() for p in ctx.model.parameters())
            return Measurement[int].available(
                count,
                source="model.parameters() → sum(numel)",
            )
        except Exception as exc:  # noqa: BLE001
            return Measurement[int].unavailable(str(exc))

    def precision(self, ctx: AdapterContext) -> Measurement[str]:
        if ctx.model is None:
            return Measurement[str].unavailable("No model bound.")
        try:
            dtypes: set[str] = set()
            for p in ctx.model.parameters():
                dtypes.add(str(p.dtype).replace("torch.", ""))
            if not dtypes:
                return Measurement[str].unavailable("No parameters found.")
            label = ", ".join(sorted(dtypes))
            return Measurement[str].available(
                label,
                source="model.parameters() → .dtype",
            )
        except Exception as exc:  # noqa: BLE001
            return Measurement[str].unavailable(str(exc))

    def weights_bytes(self, ctx: AdapterContext) -> Measurement[int]:
        if ctx.model is None:
            return Measurement[int].unavailable("No model bound.")
        try:
            total = sum(p.numel() * p.element_size() for p in ctx.model.parameters())
            return Measurement[int].available(
                total,
                source="model.parameters() → sum(numel * element_size)",
            )
        except Exception as exc:  # noqa: BLE001
            return Measurement[int].unavailable(str(exc))
