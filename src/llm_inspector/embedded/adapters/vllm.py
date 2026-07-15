"""
vLLM engine adapter (stub for v0.3 — extended in future releases).

vLLM runs inference through LLMEngine / AsyncLLMEngine.  When the engine
object is passed to ``attach(engine=...)``, this adapter extracts what
vLLM exposes internally (model config, KV cache stats via engine APIs).

For now: detects vLLM engine objects and reports runtime kind; deep
metrics require vLLM-specific hooks added incrementally.
"""

from __future__ import annotations

from typing import Any

from llm_inspector.embedded.adapters.base import AdapterContext, ModelAdapter
from llm_inspector.models.measurement import Measurement


def _is_vllm_engine(obj: Any) -> bool:
    """Detect vLLM engine / EngineCore without hard import dependency."""
    cls_name = type(obj).__name__
    module = type(obj).__module__ or ""
    return (
        cls_name
        in (
            "LLMEngine",
            "AsyncLLMEngine",
            "LLM",
            "AsyncLLM",
            "EngineCore",          # vLLM ≥ 0.6 multiprocess worker
            "EngineCoreProc",
        )
        or "vllm" in module.lower()
    )


class VLLMAdapter(ModelAdapter):
    """Adapter for vLLM LLMEngine / AsyncLLMEngine."""

    name = "vllm"

    def supports(self, obj: Any) -> bool:
        return _is_vllm_engine(obj)

    def bind(self, obj: Any, engine: Any | None = None) -> AdapterContext:
        # obj may be engine directly, or model with engine passed separately
        bound_engine = obj if _is_vllm_engine(obj) else engine
        model = None if _is_vllm_engine(obj) else obj
        return AdapterContext(model=model, engine=bound_engine, runtime_kind="vllm")

    def model_name(self, ctx: AdapterContext) -> Measurement[str]:
        engine = ctx.engine
        if engine is None:
            return Measurement[str].unavailable("No vLLM engine bound.")
        try:
            cfg = self._model_config(engine)
            if cfg is not None:
                name = getattr(cfg, "model", None) or getattr(cfg, "served_model_name", None)
                if name:
                    return Measurement[str].available(
                        str(name),
                        source="vllm.engine.model_config.model",
                    )
            return Measurement[str].unavailable("vLLM engine model_config not accessible.")
        except Exception as exc:  # noqa: BLE001
            return Measurement[str].unavailable(str(exc))

    def parameter_count(self, ctx: AdapterContext) -> Measurement[int]:
        return Measurement[int].unavailable(
            "vLLM parameter count requires engine introspection (future release)."
        )

    def precision(self, ctx: AdapterContext) -> Measurement[str]:
        engine = ctx.engine
        if engine is None:
            return Measurement[str].unavailable("No vLLM engine bound.")
        try:
            cfg = self._model_config(engine)
            dtype = getattr(cfg, "dtype", None) if cfg else None
            if dtype:
                return Measurement[str].available(
                    str(dtype),
                    source="vllm.engine.model_config.dtype",
                )
            return Measurement[str].unavailable("vLLM dtype not exposed.")
        except Exception as exc:  # noqa: BLE001
            return Measurement[str].unavailable(str(exc))

    @staticmethod
    def _model_config(engine: Any) -> Any:
        """Read model config from LLMEngine or EngineCore (vLLM ≥ 0.6)."""
        cfg = getattr(engine, "model_config", None)
        if cfg is not None:
            return cfg
        vllm_config = getattr(engine, "vllm_config", None)
        if vllm_config is not None:
            return getattr(vllm_config, "model_config", None)
        return None
