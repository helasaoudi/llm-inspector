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
    """Adapter for vLLM LLMEngine / AsyncLLMEngine / EngineCore."""

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

    def architecture(self, ctx: AdapterContext) -> Measurement[str]:
        engine = ctx.engine
        if engine is None:
            return Measurement[str].unavailable("No vLLM engine bound.")
        try:
            cfg = self._model_config(engine)
            if cfg is None:
                return Measurement[str].unavailable("vLLM model_config not accessible.")

            arch_cfg = getattr(cfg, "architecture", None)
            arch_list = getattr(arch_cfg, "architectures", None) if arch_cfg else None
            if arch_list and isinstance(arch_list, list):
                return Measurement[str].available(
                    ", ".join(str(a) for a in arch_list),
                    source="vllm.engine.model_config.architecture.architectures",
                )

            hf_cfg = getattr(cfg, "hf_config", None)
            hf_arch = getattr(hf_cfg, "architectures", None) if hf_cfg else None
            if hf_arch and isinstance(hf_arch, list):
                return Measurement[str].available(
                    ", ".join(str(a) for a in hf_arch),
                    source="vllm.engine.model_config.hf_config.architectures",
                )

            model_type = getattr(cfg, "model_type", None) or (
                getattr(hf_cfg, "model_type", None) if hf_cfg else None
            )
            if model_type:
                return Measurement[str].available(
                    str(model_type),
                    source="vllm.engine.model_config.model_type",
                )
            return Measurement[str].unavailable("vLLM architecture not exposed on model_config.")
        except Exception as exc:  # noqa: BLE001
            return Measurement[str].unavailable(str(exc))

    def parameter_count(self, ctx: AdapterContext) -> Measurement[int]:
        engine = ctx.engine
        if engine is None:
            return Measurement[int].unavailable("No vLLM engine bound.")
        try:
            cfg = self._model_config(engine)
            hf_cfg = getattr(cfg, "hf_config", None) if cfg else None
            for attr in ("num_parameters", "n_params"):
                raw = getattr(hf_cfg, attr, None) if hf_cfg else None
                if raw is not None:
                    return Measurement[int].available(
                        int(raw),
                        source=f"vllm.engine.model_config.hf_config.{attr}",
                    )
            return Measurement[int].unavailable(
                "vLLM does not expose parameter count on model_config "
                "(requires model runner introspection)."
            )
        except Exception as exc:  # noqa: BLE001
            return Measurement[int].unavailable(str(exc))

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

    def context_length(self, ctx: AdapterContext) -> Measurement[int]:
        engine = ctx.engine
        if engine is None:
            return Measurement[int].unavailable("No vLLM engine bound.")
        try:
            cfg = self._model_config(engine)
            raw = getattr(cfg, "max_model_len", None) if cfg else None
            if raw is not None:
                return Measurement[int].available(
                    int(raw),
                    source="vllm.engine.model_config.max_model_len",
                )
            return Measurement[int].unavailable("vLLM max_model_len not exposed.")
        except Exception as exc:  # noqa: BLE001
            return Measurement[int].unavailable(str(exc))

    def tensor_parallel(self, ctx: AdapterContext) -> Measurement[int]:
        engine = ctx.engine
        if engine is None:
            return Measurement[int].unavailable("No vLLM engine bound.")
        try:
            parallel = self._parallel_config(engine)
            raw = getattr(parallel, "tensor_parallel_size", None) if parallel else None
            if raw is not None:
                return Measurement[int].available(
                    int(raw),
                    source="vllm.engine.vllm_config.parallel_config.tensor_parallel_size",
                )
            return Measurement[int].unavailable("vLLM tensor_parallel_size not exposed.")
        except Exception as exc:  # noqa: BLE001
            return Measurement[int].unavailable(str(exc))

    def pipeline_parallel(self, ctx: AdapterContext) -> Measurement[int]:
        engine = ctx.engine
        if engine is None:
            return Measurement[int].unavailable("No vLLM engine bound.")
        try:
            parallel = self._parallel_config(engine)
            raw = getattr(parallel, "pipeline_parallel_size", None) if parallel else None
            if raw is not None:
                return Measurement[int].available(
                    int(raw),
                    source="vllm.engine.vllm_config.parallel_config.pipeline_parallel_size",
                )
            return Measurement[int].unavailable("vLLM pipeline_parallel_size not exposed.")
        except Exception as exc:  # noqa: BLE001
            return Measurement[int].unavailable(str(exc))

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

    @staticmethod
    def _parallel_config(engine: Any) -> Any:
        vllm_config = getattr(engine, "vllm_config", None)
        if vllm_config is not None:
            return getattr(vllm_config, "parallel_config", None)
        return getattr(engine, "parallel_config", None)
