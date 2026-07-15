"""Adapter registry — select the right framework adapter."""

from __future__ import annotations

from typing import Any

from llm_inspector.embedded.adapters.base import AdapterContext, ModelAdapter
from llm_inspector.embedded.adapters.huggingface import HuggingFaceAdapter
from llm_inspector.embedded.adapters.pytorch import PyTorchAdapter
from llm_inspector.embedded.adapters.vllm import VLLMAdapter


def select_adapter(model: Any | None = None, engine: Any | None = None) -> ModelAdapter:
    """
    Pick the best adapter for *model* or *engine*.

    Priority: vLLM engine > HuggingFace PreTrainedModel > generic PyTorch.
    """
    candidates: list[ModelAdapter] = [
        VLLMAdapter(),
        HuggingFaceAdapter(),
        PyTorchAdapter(),
    ]
    for obj in (engine, model):
        if obj is None:
            continue
        for adapter in candidates:
            if adapter.supports(obj):
                return adapter
    return PyTorchAdapter()


def bind_context(
    model: Any | None = None,
    engine: Any | None = None,
    options: dict | None = None,
) -> tuple[ModelAdapter, AdapterContext]:
    """Select adapter and bind context."""
    adapter = select_adapter(model, engine)
    target = engine if engine is not None and adapter.name == "vllm" else (model or engine)
    ctx = adapter.bind(target, engine=engine)
    if options:
        ctx.options.update(options)
    return adapter, ctx
