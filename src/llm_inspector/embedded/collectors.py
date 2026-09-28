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
      - torch.cuda.memory_* per visible device (on demand)
      - StreamingMetrics peak counters (continuous)

    Per-device queries use ``torch.cuda.memory_allocated(device)`` etc.
    These are only valid inside the inference process (attach). Process-level
    totals are the measured sum across visible devices — never estimated.
    """
    del request  # memory collector has no options yet
    from llm_inspector.models.results import GpuDeviceMemory  # noqa: PLC0415

    gpu_allocated = Measurement[int].unavailable("CUDA not available or torch not installed.")
    gpu_reserved = Measurement[int].unavailable("CUDA not available or torch not installed.")
    peak = Measurement[int].unavailable("CUDA not available or torch not installed.")
    device_rows: list[GpuDeviceMemory] = []

    if _torch_available():
        import torch  # noqa: PLC0415

        n_devices = int(torch.cuda.device_count())
        total_allocated = 0
        total_reserved = 0
        total_peak = 0
        any_device = False

        for device in range(n_devices):
            try:
                allocated = int(torch.cuda.memory_allocated(device))
                reserved = int(torch.cuda.memory_reserved(device))
                max_alloc = int(torch.cuda.max_memory_allocated(device))
            except Exception as exc:  # noqa: BLE001
                device_rows.append(
                    GpuDeviceMemory(
                        device_index=device,
                        allocated=Measurement[int].unavailable(
                            f"torch.cuda.memory_allocated({device}) failed: {exc}"
                        ),
                        reserved=Measurement[int].unavailable(
                            f"torch.cuda.memory_reserved({device}) failed: {exc}"
                        ),
                        peak=Measurement[int].unavailable(
                            f"torch.cuda.max_memory_allocated({device}) failed: {exc}"
                        ),
                    )
                )
                continue

            any_device = True
            total_allocated += allocated
            total_reserved += reserved
            total_peak = max(total_peak, max_alloc)

            device_rows.append(
                GpuDeviceMemory(
                    device_index=device,
                    allocated=Measurement[int].available(
                        allocated,
                        source=f"torch.cuda.memory_allocated({device})",
                    ),
                    reserved=Measurement[int].available(
                        reserved,
                        source=f"torch.cuda.memory_reserved({device})",
                    ),
                    peak=Measurement[int].available(
                        max_alloc,
                        source=f"torch.cuda.max_memory_allocated({device})",
                    )
                    if max_alloc > 0
                    else Measurement[int].unavailable(f"No peak recorded yet on device {device}."),
                )
            )

        if any_device:
            gpu_allocated = Measurement[int].available(
                total_allocated,
                source=(
                    f"sum of torch.cuda.memory_allocated(device) for device in 0..{n_devices - 1}"
                ),
            )
            gpu_reserved = Measurement[int].available(
                total_reserved,
                source=(
                    f"sum of torch.cuda.memory_reserved(device) for device in 0..{n_devices - 1}"
                ),
            )

            stream_peak = streaming.peak_allocated_bytes
            if stream_peak > 0 or total_peak > 0:
                peak_val = max(stream_peak, total_peak)
                peak = Measurement[int].available(
                    peak_val,
                    source=(
                        "max(StreamingMetrics.peak_allocated_bytes, "
                        "sum/max of torch.cuda.max_memory_allocated(device))"
                    ),
                )
            else:
                peak = Measurement[int].unavailable(
                    "No peak recorded yet — run inference to populate streaming counters."
                )

            # Update streaming with process-wide allocated/reserved totals
            streaming.record_allocated(total_allocated)
            streaming.record_reserved(total_reserved)

    # process_ram and gpu_used are filled by external collectors (psutil/NVML)
    result = MemoryResult(
        process_ram=Measurement[int].unavailable("process_ram collected externally via psutil."),
        gpu_used=Measurement[int].unavailable("gpu_used collected externally via NVML."),
        gpu_allocated=gpu_allocated,
        gpu_reserved=gpu_reserved,
        peak=peak,
        devices=device_rows,
    )
    return model_to_dict(result)


def collect_model_embedded(
    request: CollectorRequest,
    adapter_ctx: tuple[Any, AdapterContext] | None,
) -> dict[str, Any]:
    """Snapshot model identity + tokenizer/architecture details from adapter."""
    del request
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

    # Phase 5 details — prefer adapter methods, else shared extractors
    detail_fn = getattr(adapter, "model_details", None)
    if callable(detail_fn):
        details = detail_fn(ctx)
    else:
        from llm_inspector.embedded.model_details import (  # noqa: PLC0415
            collect_model_detail_fields,
        )

        resolve = getattr(adapter, "_resolve_model", None)
        details = collect_model_detail_fields(
            engine=ctx.engine,
            model=ctx.model,
            resolve_model=resolve if callable(resolve) else None,
        )

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
        tokenizer_class=details.get(
            "tokenizer_class",
            Measurement[str].unavailable("Tokenizer not exposed by adapter."),
        ),
        vocab_size=details.get(
            "vocab_size",
            Measurement[int].unavailable("Vocab size not exposed by adapter."),
        ),
        chat_template=details.get(
            "chat_template",
            Measurement[str].unavailable("Chat template not exposed by adapter."),
        ),
        bos_token=details.get(
            "bos_token",
            Measurement[str].unavailable("BOS token not exposed by adapter."),
        ),
        eos_token=details.get(
            "eos_token",
            Measurement[str].unavailable("EOS token not exposed by adapter."),
        ),
        num_layers=details.get(
            "num_layers",
            Measurement[int].unavailable("num_layers not exposed by adapter."),
        ),
        hidden_size=details.get(
            "hidden_size",
            Measurement[int].unavailable("hidden_size not exposed by adapter."),
        ),
        num_attention_heads=details.get(
            "num_attention_heads",
            Measurement[int].unavailable("num_attention_heads not exposed by adapter."),
        ),
        num_kv_heads=details.get(
            "num_kv_heads",
            Measurement[int].unavailable("num_kv_heads not exposed by adapter."),
        ),
        num_experts=details.get(
            "num_experts",
            Measurement[int].unavailable("num_experts not exposed by adapter."),
        ),
        embed_params=details.get(
            "embed_params",
            Measurement[int].unavailable("embed_params not exposed by adapter."),
        ),
        transformer_params=details.get(
            "transformer_params",
            Measurement[int].unavailable("transformer_params not exposed by adapter."),
        ),
        head_params=details.get(
            "head_params",
            Measurement[int].unavailable("head_params not exposed by adapter."),
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
            total=_sum_total([activations, workspace, other]),
        )
        return model_to_dict(result)

    adapter, ctx = adapter_ctx

    weights_fn = getattr(adapter, "weights_bytes", None)
    weights = (
        weights_fn(ctx) if callable(weights_fn) else _unavail("Weights not exposed by adapter.")
    )

    kv_fn = getattr(adapter, "kv_cache_bytes", None)
    kv = kv_fn(ctx) if callable(kv_fn) else _unavail("KV cache not exposed by adapter.")

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
        unavail = Measurement[int].unavailable("CUDA not available or torch not installed.")
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
        source = f"allocated − attach baseline (current; peak since attach: {peak_act}"
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
            source=("torch.cuda.memory_allocated() − weights − KV − current activations"),
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
    measured = [m.value for m in measurements if m.is_available and m.value is not None]
    if not measured:
        return Measurement[int].unavailable("No breakdown components measurable from adapter.")
    return Measurement[int].available(
        sum(measured),
        source="Sum of measured embedded breakdown components",
    )
