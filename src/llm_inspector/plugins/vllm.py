"""
vLLM runtime plugin.

Data sources
------------
- Cmdline args    → model name, dtype, tensor-parallel, port, host
- GET /v1/models  → confirmed model id, max_model_len
- GET /metrics    → KV cache usage (Prometheus format)

Works on both macOS (development, CPU vLLM) and Linux/DGX (GPU vLLM).
All REST calls have a 2 s timeout and degrade gracefully to Unavailable.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llm_inspector.models.enums import RuntimeKind
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import (
    COMPONENT_ORDER,
    ComponentName,
    MemoryBreakdownResult,
    MemoryComponent,
    ModelResult,
)
from llm_inspector.plugins.base import RuntimePlugin
from llm_inspector.utils.cmdline import detect_host, detect_port, parse_arg, parse_int_arg
from llm_inspector.utils.http import find_metric, get_json, get_text, parse_prometheus
from llm_inspector.utils.vllm_urls import is_engine_core_process, resolve_vllm_base_url

if TYPE_CHECKING:
    from llm_inspector.inspector.context import InspectionContext
    from llm_inspector.models.results import ProcessResult

_LOG = logging.getLogger(__name__)

# Prometheus metric names used by vLLM
_METRIC_KV_USAGE = "vllm:gpu_cache_usage_perc"
_METRIC_WEIGHTS_BYTES = "vllm:model_weights_memory_bytes"  # newer vLLM versions
_METRIC_KV_BYTES = "vllm:gpu_cache_memory_bytes"          # newer vLLM versions


class VLLMPlugin(RuntimePlugin):
    """
    Inspects live vLLM OpenAI-compatible inference servers.

    Phase 2 capabilities:
      ✓ Model name, dtype, context length, tensor-parallel (cmdline + API)
      ✓ KV cache usage (Prometheus /metrics)
      ✓ Runtime details: PagedAttention, scheduler, tensor-parallel
      ~ Weights: only if vLLM exposes the metric (newer builds)
      ~ Activations / Workspace / Other: via embedded attach (Phase 4)
    """

    kind = RuntimeKind.VLLM
    display_name = "vLLM"
    description = "OpenAI-compatible inference server with PagedAttention KV cache."

    def supports(self, process: "ProcessResult") -> bool:
        cmdline_str = " ".join(process.cmdline).lower()
        return "vllm" in cmdline_str or is_engine_core_process(process.cmdline)

    def _base_url(self, ctx: "InspectionContext") -> tuple[str, str]:
        """Resolve API base URL with provenance (cmdline, env, or port probe)."""
        from llm_inspector.utils import procfs  # noqa: PLC0415

        snap = procfs.snapshot(ctx.pid)
        return resolve_vllm_base_url(ctx.process.cmdline, snap.environ)

    # ── Capability methods ────────────────────────────────────────────────────

    def get_model_info(self, ctx: "InspectionContext") -> ModelResult | None:
        cmdline = ctx.process.cmdline
        base_url, _url_source = self._base_url(ctx)

        # Model name — try API first, fall back to cmdline.
        # Intentionally avoid -m (clashes with `python -m <module>`).
        name = self._api_model_name(base_url) or parse_arg(cmdline, "--model")
        name_m = (
            Measurement[str].available(name, source=f"vLLM GET {base_url}/v1/models")
            if name
            else Measurement[str].unavailable("--model not found in cmdline and /v1/models unreachable.")
        )

        # Context length — from /v1/models API
        ctx_len = self._api_context_length(base_url)
        ctx_len_m = (
            Measurement[int].available(ctx_len, source=f"vLLM GET {base_url}/v1/models → max_model_len")
            if ctx_len
            else Measurement[int].unavailable("max_model_len not returned by /v1/models.")
        )

        # Precision / dtype — from cmdline only
        dtype_raw = parse_arg(cmdline, "--dtype", "--quantization", "-q")
        precision_m = (
            Measurement[str].available(self._normalise_dtype(dtype_raw), source="cmdline --dtype")
            if dtype_raw
            else Measurement[str].unavailable("--dtype not specified in cmdline.")
        )

        # Tensor parallel
        tp = parse_int_arg(cmdline, "--tensor-parallel-size", "-tp")
        tp_m = (
            Measurement[int].available(tp, source="cmdline --tensor-parallel-size")
            if tp is not None
            else Measurement[int].unavailable("--tensor-parallel-size not specified (default: 1).")
        )

        # Pipeline parallel
        pp = parse_int_arg(cmdline, "--pipeline-parallel-size", "-pp")
        pp_m = (
            Measurement[int].available(pp, source="cmdline --pipeline-parallel-size")
            if pp is not None
            else Measurement[int].unavailable("--pipeline-parallel-size not specified.")
        )

        return ModelResult(
            name=name_m,
            architecture=Measurement[str].unavailable(
                "Architecture not exposed by vLLM API."
            ),
            parameter_count=Measurement[int].unavailable(
                "Parameter count not exposed by vLLM API."
            ),
            precision=precision_m,
            context_length=ctx_len_m,
            tensor_parallel=tp_m,
            pipeline_parallel=pp_m,
        )

    def get_memory_breakdown(
        self, ctx: "InspectionContext"
    ) -> MemoryBreakdownResult | None:
        base_url, _ = self._base_url(ctx)
        metrics_url = f"{base_url}/metrics"

        raw_text = get_text(metrics_url)
        if raw_text is None:
            return MemoryBreakdownResult(
                components=[
                    MemoryComponent(
                        name=name,
                        measurement=Measurement[int].unavailable(
                            f"vLLM /metrics endpoint unreachable at {metrics_url}."
                        ),
                        order=COMPONENT_ORDER.get(name, 99),
                    )
                    for name in (
                        ComponentName.WEIGHTS,
                        ComponentName.KV_CACHE,
                        ComponentName.ACTIVATIONS,
                        ComponentName.WORKSPACE,
                        ComponentName.OTHER,
                    )
                ],
                total=Measurement[int].unavailable(
                    f"/metrics unreachable at {metrics_url}."
                ),
            )

        metrics = parse_prometheus(raw_text)
        components: list[MemoryComponent] = []

        # ── Weights ──────────────────────────────────────────────────────────
        weights_bytes = find_metric(metrics, _METRIC_WEIGHTS_BYTES)
        if weights_bytes is not None:
            weights_m = Measurement[int].available(
                int(weights_bytes),
                source=f"vLLM /metrics → {_METRIC_WEIGHTS_BYTES}",
            )
        else:
            weights_m = Measurement[int].unavailable(
                f"vLLM /metrics does not expose {_METRIC_WEIGHTS_BYTES}. "
                "vLLM 0.15+ reports KV usage % only — weight/KV byte totals "
                "require embedded engine introspection (future release)."
            )
        components.append(MemoryComponent(
            name=ComponentName.WEIGHTS,
            measurement=weights_m,
            description="Model weight tensors loaded into GPU VRAM.",
            order=COMPONENT_ORDER[ComponentName.WEIGHTS],
        ))

        # ── KV Cache ─────────────────────────────────────────────────────────
        kv_bytes = find_metric(metrics, _METRIC_KV_BYTES)
        if kv_bytes is not None:
            kv_m = Measurement[int].available(
                int(kv_bytes),
                source=f"vLLM /metrics → {_METRIC_KV_BYTES}",
            )
        else:
            # Try to compute from usage % × device VRAM (approximation warning)
            # Actually — no estimation. Mark unavailable.
            kv_m = Measurement[int].unavailable(
                f"vLLM /metrics does not expose {_METRIC_KV_BYTES}. "
                f"KV cache usage fraction: {self._kv_usage_pct(metrics)} "
                "(byte totals require embedded engine introspection)."
            )
        components.append(MemoryComponent(
            name=ComponentName.KV_CACHE,
            measurement=kv_m,
            description="PagedAttention KV cache blocks allocated in GPU VRAM.",
            order=COMPONENT_ORDER[ComponentName.KV_CACHE],
        ))

        # ── Activations / Workspace / Other ───────────────────────────────────
        # Filled by embedded attach (Phase 4). External /metrics has no byte totals.
        for name, reason in (
            (
                ComponentName.ACTIVATIONS,
                "vLLM /metrics has no activation bytes — use embedded attach().",
            ),
            (
                ComponentName.WORKSPACE,
                "vLLM /metrics has no workspace bytes — use embedded attach().",
            ),
            (
                ComponentName.OTHER,
                "vLLM /metrics has no residual breakdown — use embedded attach().",
            ),
        ):
            components.append(MemoryComponent(
                name=name,
                measurement=Measurement[int].unavailable(reason),
                order=COMPONENT_ORDER[name],
            ))

        # Total — sum of available components only
        measured = [c.measurement.value for c in components if c.measurement.is_available and c.measurement.value]
        total_m = (
            Measurement[int].available(sum(measured), source="Sum of measured breakdown components")
            if measured
            else Measurement[int].unavailable("No breakdown components measurable.")
        )

        return MemoryBreakdownResult(components=components, total=total_m)

    def get_runtime_details(
        self, ctx: "InspectionContext"
    ) -> dict[str, Measurement[str]]:
        cmdline = ctx.process.cmdline
        details: dict[str, Measurement[str]] = {}

        details["PagedAttention"] = Measurement[str].available(
            "Enabled", source="vLLM always uses PagedAttention"
        )

        # Scheduler policy
        scheduler = parse_arg(cmdline, "--scheduler-policy") or "fcfs"
        details["Scheduler"] = Measurement[str].available(
            scheduler.upper(), source="cmdline --scheduler-policy (default: FCFS)"
        )

        # Tensor parallel
        tp = parse_int_arg(cmdline, "--tensor-parallel-size", "-tp", default=1)
        details["Tensor Parallel"] = Measurement[str].available(
            str(tp), source="cmdline --tensor-parallel-size"
        )

        # Max sequences
        max_seqs = parse_arg(cmdline, "--max-num-seqs")
        if max_seqs:
            details["Max Sequences"] = Measurement[str].available(
                max_seqs, source="cmdline --max-num-seqs"
            )

        # GPU memory utilization
        gpu_util = parse_arg(cmdline, "--gpu-memory-utilization")
        if gpu_util:
            details["GPU Memory Utilization"] = Measurement[str].available(
                gpu_util, source="cmdline --gpu-memory-utilization"
            )

        return details

    def get_version(self, ctx: "InspectionContext") -> Measurement[str]:
        base_url, _ = self._base_url(ctx)
        # Try the /version or /v1/openai endpoint
        data = get_json(f"{base_url}/version")
        if isinstance(data, dict) and "version" in data:
            return Measurement[str].available(
                data["version"],
                source=f"vLLM GET {base_url}/version",
            )
        return Measurement[str].unavailable(
            "vLLM version endpoint not reachable. Consider upgrading vLLM."
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _api_model_name(base_url: str) -> str | None:
        data = get_json(f"{base_url}/v1/models")
        if isinstance(data, dict):
            models = data.get("data", [])
            if models and isinstance(models[0], dict):
                return models[0].get("id")
        return None

    @staticmethod
    def _api_context_length(base_url: str) -> int | None:
        data = get_json(f"{base_url}/v1/models")
        if isinstance(data, dict):
            models = data.get("data", [])
            if models and isinstance(models[0], dict):
                raw = models[0].get("max_model_len")
                if raw is not None:
                    return int(raw)
        return None

    @staticmethod
    def _normalise_dtype(raw: str) -> str:
        mapping = {
            "float16": "FP16",
            "float32": "FP32",
            "bfloat16": "BF16",
            "half": "FP16",
            "auto": "auto",
            "fp8": "FP8",
            "int8": "INT8",
            "int4": "INT4",
            "awq": "AWQ",
            "gptq": "GPTQ",
        }
        return mapping.get(raw.lower(), raw.upper())

    @staticmethod
    def _kv_usage_pct(metrics: dict[str, float]) -> str:
        val = find_metric(metrics, _METRIC_KV_USAGE)
        if val is not None:
            return f"{val * 100:.1f}%"
        return "unknown"
