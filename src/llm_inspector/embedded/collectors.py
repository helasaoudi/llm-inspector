"""Embedded collectors — run inside the inference process."""

from __future__ import annotations

from typing import Any

from llm_inspector.embedded.adapters.base import AdapterContext
from llm_inspector.embedded.streaming import StreamingMetrics
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import (
    COMPONENT_ORDER,
    ComponentName,
    MemoryBreakdownResult,
    MemoryComponent,
    MemoryResult,
    ModelResult,
)
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


def collect_memory_breakdown_embedded(
    request: CollectorRequest,
    adapter_ctx: tuple[Any, AdapterContext] | None,
    streaming: StreamingMetrics | None = None,
) -> dict[str, Any]:
    """
    Snapshot Weights / KV / Activations / Workspace / Other.

    Phase 4 accounting (measured, not guessed):
      - Weights / KV: adapter (model.parameters / kv_cache_tensors)
      - Workspace: reserved − allocated (torch caching allocator)
      - Activations: current (allocated − baseline) and peak watermark
      - Other: allocated − weights − KV − current activations
    """
    del request
    streaming = streaming or StreamingMetrics()

    def _unavail(reason: str) -> Measurement[int]:
        return Measurement[int].unavailable(reason)

    if adapter_ctx is None:
        reason = "No model/engine bound — call attach(model=...) or attach(engine=...)."
        # Still report Workspace from torch if CUDA is up (auto_attach case).
        workspace, activations, other = _phase4_residuals(
            weights=None, kv=None, streaming=streaming
        )
        result = MemoryBreakdownResult(
            components=[
                MemoryComponent(
                    name=ComponentName.WEIGHTS,
                    measurement=_unavail(reason),
                    order=COMPONENT_ORDER[ComponentName.WEIGHTS],
                ),
                MemoryComponent(
                    name=ComponentName.KV_CACHE,
                    measurement=_unavail(reason),
                    order=COMPONENT_ORDER[ComponentName.KV_CACHE],
                ),
                MemoryComponent(
                    name=ComponentName.ACTIVATIONS,
                    measurement=activations,
                    order=COMPONENT_ORDER[ComponentName.ACTIVATIONS],
                ),
                MemoryComponent(
                    name=ComponentName.WORKSPACE,
                    measurement=workspace,
                    order=COMPONENT_ORDER[ComponentName.WORKSPACE],
                ),
                MemoryComponent(
                    name=ComponentName.OTHER,
                    measurement=other if other.is_available else _unavail(reason),
                    order=COMPONENT_ORDER[ComponentName.OTHER],
                ),
            ],
            total=_sum_total(
                [activations, workspace, other]
            ),
        )
        return model_to_dict(result)

    adapter, ctx = adapter_ctx

    weights_fn = getattr(adapter, "weights_bytes", None)
    weights = (
        weights_fn(ctx)
        if callable(weights_fn)
        else _unavail("Weights not exposed by adapter.")
    )

    kv_fn = getattr(adapter, "kv_cache_bytes", None)
    kv = (
        kv_fn(ctx)
        if callable(kv_fn)
        else _unavail("KV cache not exposed by adapter.")
    )

    workspace, activations, other = _phase4_residuals(
        weights=weights if weights.is_available else None,
        kv=kv if kv.is_available else None,
        streaming=streaming,
    )

    components = [
        MemoryComponent(
            name=ComponentName.WEIGHTS,
            measurement=weights,
            description="Model weight tensors loaded into GPU VRAM.",
            order=COMPONENT_ORDER[ComponentName.WEIGHTS],
        ),
        MemoryComponent(
            name=ComponentName.KV_CACHE,
            measurement=kv,
            description="PagedAttention KV cache blocks allocated in GPU VRAM.",
            order=COMPONENT_ORDER[ComponentName.KV_CACHE],
        ),
        MemoryComponent(
            name=ComponentName.ACTIVATIONS,
            measurement=activations,
            description="Transient activation memory above attach baseline (peak watermark).",
            order=COMPONENT_ORDER[ComponentName.ACTIVATIONS],
        ),
        MemoryComponent(
            name=ComponentName.WORKSPACE,
            measurement=workspace,
            description="Torch caching-allocator pool (reserved − allocated).",
            order=COMPONENT_ORDER[ComponentName.WORKSPACE],
        ),
        MemoryComponent(
            name=ComponentName.OTHER,
            measurement=other,
            description="Allocated bytes not explained by weights, KV, or activations.",
            order=COMPONENT_ORDER[ComponentName.OTHER],
        ),
    ]

    return model_to_dict(
        MemoryBreakdownResult(
            components=components,
            total=_sum_total([c.measurement for c in components]),
        )
    )


def _phase4_residuals(
    *,
    weights: Measurement[int] | None,
    kv: Measurement[int] | None,
    streaming: StreamingMetrics,
) -> tuple[Measurement[int], Measurement[int], Measurement[int]]:
    """Compute Workspace, Activations, Other from torch + streaming baseline."""
    if not _torch_available():
        unavail = Measurement[int].unavailable(
            "CUDA not available or torch not installed."
        )
        return unavail, unavail, unavail

    import torch  # noqa: PLC0415

    allocated = int(torch.cuda.memory_allocated())
    reserved = int(torch.cuda.memory_reserved())
    streaming.record_allocated(allocated)
    streaming.record_reserved(reserved)

    workspace = Measurement[int].available(
        max(0, reserved - allocated),
        source="torch.cuda.memory_reserved() − memory_allocated()",
    )

    # Activations: current transient above attach baseline (0 when idle).
    # Peak since attach is noted in the source string for capacity planning.
    if streaming.baseline_allocated_bytes is not None:
        current_act = streaming.current_activation_bytes(allocated)
        peak_act = max(streaming.peak_activation_bytes, current_act)
        source = (
            "allocated − attach baseline "
            f"(current; peak since attach: {peak_act}"
        )
        if streaming.activation_hooks_enabled:
            source += "; forward hooks enabled"
        source += ")"
        activations = Measurement[int].available(current_act, source=source)
        act_for_residual = current_act
    else:
        activations = Measurement[int].unavailable(
            "Activation baseline not set — attach() after model load to enable."
        )
        act_for_residual = 0

    w_val = weights.value if weights is not None and weights.is_available else None
    kv_val = kv.value if kv is not None and kv.is_available else None

    if w_val is not None and kv_val is not None:
        other_val = max(0, allocated - int(w_val) - int(kv_val) - act_for_residual)
        other = Measurement[int].available(
            other_val,
            source=(
                "torch.cuda.memory_allocated() − weights − KV − current activations"
            ),
        )
    elif w_val is not None:
        other_val = max(0, allocated - int(w_val) - act_for_residual)
        other = Measurement[int].available(
            other_val,
            source="torch.cuda.memory_allocated() − weights − current activations",
        )
    else:
        other = Measurement[int].unavailable(
            "Other requires measured Weights (and preferably KV) from adapter."
        )

    return workspace, activations, other


def _sum_total(measurements: list[Measurement[int]]) -> Measurement[int]:
    measured = [
        m.value for m in measurements if m.is_available and m.value is not None
    ]
    if not measured:
        return Measurement[int].unavailable(
            "No breakdown components measurable from adapter."
        )
    return Measurement[int].available(
        sum(measured),
        source="Sum of measured embedded breakdown components",
    )
