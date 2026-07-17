"""HuggingFace Transformers adapter."""

from __future__ import annotations

from typing import Any

from llm_inspector.embedded.adapters.base import AdapterContext, ModelAdapter
from llm_inspector.embedded.adapters.pytorch import PyTorchAdapter
from llm_inspector.models.measurement import Measurement


class HuggingFaceAdapter(ModelAdapter):
    """
    Adapter for ``transformers.PreTrainedModel``.

    Extends PyTorchAdapter with HF-specific model identity from config.
    """

    name = "huggingface"

    def __init__(self) -> None:
        self._torch = PyTorchAdapter()

    def supports(self, obj: Any) -> bool:
        try:
            from transformers import PreTrainedModel  # noqa: PLC0415

            return isinstance(obj, PreTrainedModel)
        except ImportError:
            return False

    def bind(self, obj: Any, engine: Any | None = None) -> AdapterContext:
        return AdapterContext(model=obj, engine=engine, runtime_kind="huggingface")

    def model_name(self, ctx: AdapterContext) -> Measurement[str]:
        if ctx.model is None:
            return Measurement[str].unavailable("No model bound.")
        try:
            cfg = getattr(ctx.model, "config", None)
            if cfg is None:
                return Measurement[str].unavailable("Model has no config.")
            # Prefer _name_or_path (HF cache id), fall back to model_type
            name = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
            if name and name not in (".", ""):
                return Measurement[str].available(
                    str(name),
                    source="model.config._name_or_path",
                )
            model_type = getattr(cfg, "model_type", None)
            if model_type:
                return Measurement[str].available(
                    str(model_type),
                    source="model.config.model_type",
                )
            return Measurement[str].unavailable("HF config has no model name.")
        except Exception as exc:  # noqa: BLE001
            return Measurement[str].unavailable(str(exc))

    def parameter_count(self, ctx: AdapterContext) -> Measurement[int]:
        return self._torch.parameter_count(ctx)

    def precision(self, ctx: AdapterContext) -> Measurement[str]:
        return self._torch.precision(ctx)

    def weights_bytes(self, ctx: AdapterContext) -> Measurement[int]:
        return self._torch.weights_bytes(ctx)

    def architecture(self, ctx: AdapterContext) -> Measurement[str]:
        if ctx.model is None:
            return Measurement[str].unavailable("No model bound.")
        try:
            arch = getattr(ctx.model.config, "architectures", None)
            if arch and isinstance(arch, list):
                return Measurement[str].available(
                    ", ".join(arch),
                    source="model.config.architectures",
                )
            model_type = getattr(ctx.model.config, "model_type", None)
            if model_type:
                return Measurement[str].available(
                    str(model_type),
                    source="model.config.model_type",
                )
            return Measurement[str].unavailable("HF config has no architecture.")
        except Exception as exc:  # noqa: BLE001
            return Measurement[str].unavailable(str(exc))

    def model_details(self, ctx: AdapterContext) -> dict[str, Measurement]:
        from llm_inspector.embedded.model_details import (  # noqa: PLC0415
            collect_model_detail_fields,
        )

        return collect_model_detail_fields(
            engine=ctx.engine,
            model=ctx.model,
        )

    def context_length(self, ctx: AdapterContext) -> Measurement[int]:
        if ctx.model is None:
            return Measurement[int].unavailable("No model bound.")
        try:
            cfg = getattr(ctx.model, "config", None)
            raw = getattr(cfg, "max_position_embeddings", None) or getattr(
                cfg, "max_sequence_length", None
            )
            if raw is not None:
                return Measurement[int].available(
                    int(raw),
                    source="model.config.max_position_embeddings",
                )
            return Measurement[int].unavailable(
                "max_position_embeddings not on model config."
            )
        except Exception as exc:  # noqa: BLE001
            return Measurement[int].unavailable(str(exc))
