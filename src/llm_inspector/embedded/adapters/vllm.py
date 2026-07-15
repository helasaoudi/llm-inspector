"""
vLLM engine adapter.

When the engine object is passed to ``attach(engine=...)``, this adapter
extracts model identity and memory facts from EngineCore / LLMEngine
internals (model_config, parallel_config, model runner, KV cache config).
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
            model = self._resolve_model(engine)
            if model is not None:
                count = sum(p.numel() for p in model.parameters())
                if count > 0:
                    return Measurement[int].available(
                        count,
                        source="vllm.model_executor → model.parameters() → sum(numel)",
                    )

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
                "vLLM model runner not accessible for parameter counting."
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

    def weights_bytes(self, ctx: AdapterContext) -> Measurement[int]:
        engine = ctx.engine
        if engine is None:
            return Measurement[int].unavailable("No vLLM engine bound.")
        try:
            model = self._resolve_model(engine)
            if model is None:
                return Measurement[int].unavailable(
                    "vLLM model runner not accessible for weight sizing."
                )
            total = sum(p.numel() * p.element_size() for p in model.parameters())
            if total <= 0:
                return Measurement[int].unavailable("Model has no parameters.")
            return Measurement[int].available(
                total,
                source="vllm.model_executor → model.parameters() → sum(numel * element_size)",
            )
        except Exception as exc:  # noqa: BLE001
            return Measurement[int].unavailable(str(exc))

    def kv_cache_bytes(self, ctx: AdapterContext) -> Measurement[int]:
        engine = ctx.engine
        if engine is None:
            return Measurement[int].unavailable("No vLLM engine bound.")
        try:
            # Prefer sum of allocated KV tensor sizes (most accurate).
            kv_cfg = self._kv_cache_config(engine)
            if kv_cfg is not None:
                tensors = getattr(kv_cfg, "kv_cache_tensors", None) or []
                sizes = [int(getattr(t, "size", 0) or 0) for t in tensors]
                total = sum(sizes)
                if total > 0:
                    return Measurement[int].available(
                        total,
                        source="vllm.scheduler.kv_cache_config.kv_cache_tensors → sum(size)",
                    )

            # Fallback: bytes reserved for KV after weight profiling.
            reserved = getattr(engine, "available_gpu_memory_for_kv_cache", None)
            if reserved is not None and int(reserved) > 0:
                return Measurement[int].available(
                    int(reserved),
                    source="vllm.engine.available_gpu_memory_for_kv_cache",
                )

            cache = self._cache_config(engine)
            manual = getattr(cache, "kv_cache_memory_bytes", None) if cache else None
            if manual is not None and int(manual) > 0:
                return Measurement[int].available(
                    int(manual),
                    source="vllm.engine.vllm_config.cache_config.kv_cache_memory_bytes",
                )

            return Measurement[int].unavailable(
                "vLLM KV cache size not exposed on engine (kv_cache_config missing)."
            )
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

    @staticmethod
    def _cache_config(engine: Any) -> Any:
        vllm_config = getattr(engine, "vllm_config", None)
        if vllm_config is not None:
            return getattr(vllm_config, "cache_config", None)
        return getattr(engine, "cache_config", None)

    @staticmethod
    def _kv_cache_config(engine: Any) -> Any:
        scheduler = getattr(engine, "scheduler", None)
        if scheduler is not None:
            cfg = getattr(scheduler, "kv_cache_config", None)
            if cfg is not None:
                return cfg
        return getattr(engine, "kv_cache_config", None)

    @staticmethod
    def _resolve_model(engine: Any) -> Any:
        """Locate the loaded nn.Module on UniProc / EngineCore workers."""
        if engine is None:
            return None

        # Direct attribute walks common in UniProcExecutor / Worker wrappers.
        paths = (
            ("model_executor", "driver_worker", "model_runner", "model"),
            ("model_executor", "driver_worker", "worker", "model_runner", "model"),
            ("model_executor", "worker", "model_runner", "model"),
            ("model_executor", "driver_worker", "model"),
        )
        for path in paths:
            obj: Any = engine
            for attr in path:
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            else:
                if obj is not None:
                    return obj

        executor = getattr(engine, "model_executor", None)
        if executor is None:
            return None

        worker = getattr(executor, "driver_worker", None) or getattr(executor, "worker", None)
        if worker is not None:
            get_model = getattr(worker, "get_model", None)
            if callable(get_model):
                return get_model()
            runner = getattr(worker, "model_runner", None)
            if runner is not None:
                get_model = getattr(runner, "get_model", None)
                if callable(get_model):
                    return get_model()
                model = getattr(runner, "model", None)
                if model is not None:
                    return model

        # Last resort: in-process collective_rpc (UniProc only).
        rpc = getattr(executor, "collective_rpc", None)
        if callable(rpc):
            try:
                results = rpc("get_model")
                if results:
                    return results[0]
            except Exception:  # noqa: BLE001
                pass
        return None
