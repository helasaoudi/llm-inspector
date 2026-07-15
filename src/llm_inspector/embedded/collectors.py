"""Embedded collectors — run inside the inference process."""

from __future__ import annotations

from typing import Any

from llm_inspector.embedded.adapters.base import AdapterContext
from llm_inspector.embedded.adapters.registry import bind_context
from llm_inspector.embedded.streaming import StreamingMetrics
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import MemoryResult, ModelResult
from llm_inspector.rpc import CollectorRequest
from llm_inspector.serialization import model_to_dict


def _torch_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.is_available()
    except ImportError:
        return False


def collect_memory_embedded(
    request: CollectorRequest,
    streaming: StreamingMetrics,
) -> dict[str, Any]:
    """
    Snapshot memory metrics from inside the process.

    Combines:
      - torch.cuda.memory_stats() (on demand)
      - StreamingMetrics peak counters (continuous)
    """
    del request  # memory collector has no options yet

    gpu_allocated = Measurement[int].unavailable("CUDA not available or torch not installed.")
    gpu_reserved = Measurement[int].unavailable("CUDA not available or torch not installed.")
    peak = Measurement[int].unavailable("CUDA not available or torch not installed.")

    if _torch_available():
        import torch  # noqa: PLC0415

        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        max_alloc = torch.cuda.max_memory_allocated()

        gpu_allocated = Measurement[int].available(
            allocated,
            source="torch.cuda.memory_allocated()",
        )
        gpu_reserved = Measurement[int].available(
            reserved,
            source="torch.cuda.memory_reserved()",
        )

        # Peak: prefer streaming counter (captures peaks between inspects),
        # fall back to torch max tracker
        stream_peak = streaming.peak_allocated_bytes
        if stream_peak > 0:
            peak_val = max(stream_peak, max_alloc)
            peak = Measurement[int].available(
                peak_val,
                source="StreamingMetrics.peak_allocated_bytes (max with torch.cuda.max_memory_allocated())",
            )
        elif max_alloc > 0:
            peak = Measurement[int].available(
                max_alloc,
                source="torch.cuda.max_memory_allocated()",
            )
        else:
            peak = Measurement[int].unavailable(
                "No peak recorded yet — run inference to populate streaming counters."
            )

        # Update streaming counters with current values
        streaming.record_allocated(allocated)
        streaming.record_reserved(reserved)

    # process_ram and gpu_used are filled by external collectors (psutil/NVML)
    result = MemoryResult(
        process_ram=Measurement[int].unavailable(
            "process_ram collected externally via psutil."
        ),
        gpu_used=Measurement[int].unavailable(
            "gpu_used collected externally via NVML."
        ),
        gpu_allocated=gpu_allocated,
        gpu_reserved=gpu_reserved,
        peak=peak,
    )
    return model_to_dict(result)


def collect_model_embedded(
    request: CollectorRequest,
    adapter_ctx: tuple[Any, AdapterContext] | None,
) -> dict[str, Any]:
    """Snapshot model identity from bound adapter."""
    if adapter_ctx is None:
        result = ModelResult(
            name=Measurement[str].unavailable("No model bound — call attach(model=...)."),
        )
        return model_to_dict(result)

    adapter, ctx = adapter_ctx
    name = adapter.model_name(ctx)
    arch_fn = getattr(adapter, "architecture", None)
    architecture = (
        arch_fn(ctx)
        if callable(arch_fn)
        else Measurement[str].unavailable("Architecture not exposed by adapter.")
    )

    def _optional_int(method_name: str, fallback: str) -> Measurement[int]:
        fn = getattr(adapter, method_name, None)
        if callable(fn):
            return fn(ctx)
        return Measurement[int].unavailable(fallback)

    result = ModelResult(
        name=name,
        architecture=architecture,
        parameter_count=adapter.parameter_count(ctx),
        precision=adapter.precision(ctx),
        context_length=_optional_int(
            "context_length",
            "Context length requires runtime-specific adapter extension.",
        ),
        tensor_parallel=_optional_int(
            "tensor_parallel",
            "Tensor parallel requires runtime-specific adapter extension.",
        ),
        pipeline_parallel=_optional_int(
            "pipeline_parallel",
            "Pipeline parallel requires runtime-specific adapter extension.",
        ),
    )
    return model_to_dict(result)
