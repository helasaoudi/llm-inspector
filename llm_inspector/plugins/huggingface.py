"""
HuggingFace Transformers runtime plugin.

Data sources (Phase 2 — external observation only)
----------------------------------------------------
- Cmdline args → model path, dtype, trust-remote-code flag
- No REST API: vanilla Transformers inference scripts do not start a server.
- No cross-process torch inspection: requires Phase 3 (torch hooks / RPC).

Memory breakdown is fully Unavailable in Phase 2.
Phase 3 will add torch.cuda.memory_stats() via a companion sidecar.
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
from llm_inspector.utils.cmdline import flag_present, parse_arg, parse_int_arg

if TYPE_CHECKING:
    from llm_inspector.inspector.context import InspectionContext
    from llm_inspector.models.results import ProcessResult

_LOG = logging.getLogger(__name__)

# Common HuggingFace Transformers cmdline flags across scripts / accelerate / deepspeed
_MODEL_FLAGS = (
    "--model_name_or_path",
    "--model-name-or-path",
    "--model",
    "--pretrained_model_name_or_path",
    "--base_model",
)
_DTYPE_FLAGS = ("--dtype", "--torch_dtype", "--torch-dtype", "--bf16", "--fp16")

_DTYPE_MAP = {
    "float16": "FP16",
    "fp16": "FP16",
    "float32": "FP32",
    "fp32": "FP32",
    "bfloat16": "BF16",
    "bf16": "BF16",
    "auto": "auto",
    "8bit": "INT8",
    "4bit": "INT4",
}


class HuggingFacePlugin(RuntimePlugin):
    """
    Inspects Python processes that use HuggingFace Transformers for inference.

    Phase 2: cmdline argument extraction only.
    Phase 3 will add real memory attribution via torch hooks.
    """

    kind = RuntimeKind.HUGGING_FACE
    display_name = "HuggingFace Transformers"
    description = "Direct Transformers inference via AutoModel/pipeline."

    def supports(self, process: "ProcessResult") -> bool:
        cmdline_str = " ".join(process.cmdline).lower()
        return any(
            kw in cmdline_str
            for kw in ("transformers", "from_pretrained", "run_clm", "run_mlm", "text-generation")
        )

    # ── Capability methods ────────────────────────────────────────────────────

    def get_model_info(self, ctx: "InspectionContext") -> ModelResult | None:
        cmdline = ctx.process.cmdline

        # Model name / path
        name: str | None = None
        name_source: str | None = None
        for flag in _MODEL_FLAGS:
            val = parse_arg(cmdline, flag)
            if val:
                name = val
                name_source = f"cmdline {flag}"
                break

        name_m = (
            Measurement[str].available(name, source=name_source)  # type: ignore[arg-type]
            if name
            else Measurement[str].unavailable(
                "Model path not found in cmdline. "
                "Try --model_name_or_path or --model."
            )
        )

        # Precision / dtype
        precision_m = self._detect_precision(cmdline)

        # Tensor parallel (Accelerate / DeepSpeed)
        tp = (
            parse_int_arg(cmdline, "--num_processes", "--num-processes")
            or parse_int_arg(cmdline, "--nproc_per_node")
        )
        tp_m = (
            Measurement[int].available(tp, source="cmdline --num_processes / --nproc_per_node")
            if tp
            else Measurement[int].unavailable("Not specified — assuming single-process.")
        )

        return ModelResult(
            name=name_m,
            architecture=Measurement[str].unavailable(
                "Architecture requires torch model introspection (Phase 3)."
            ),
            parameter_count=Measurement[int].unavailable(
                "Parameter count requires torch model introspection (Phase 3)."
            ),
            precision=precision_m,
            context_length=Measurement[int].unavailable(
                "Context length not exposed without model access (Phase 3)."
            ),
            tensor_parallel=tp_m,
            pipeline_parallel=Measurement[int].unavailable(
                "Pipeline parallel not detectable from cmdline."
            ),
        )

    def get_memory_breakdown(
        self, ctx: "InspectionContext"
    ) -> MemoryBreakdownResult | None:
        reason = (
            "HuggingFace Transformers does not expose a metrics API. "
            "Per-component memory attribution requires torch hooks (Phase 3)."
        )
        return MemoryBreakdownResult(
            components=[
                MemoryComponent(
                    name=name,
                    measurement=Measurement[int].unavailable(reason),
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
            total=Measurement[int].unavailable(reason),
        )

    def get_runtime_details(
        self, ctx: "InspectionContext"
    ) -> dict[str, Measurement[str]]:
        cmdline = ctx.process.cmdline
        details: dict[str, Measurement[str]] = {}

        # Trust remote code
        if flag_present(cmdline, "--trust_remote_code", "--trust-remote-code"):
            details["Trust Remote Code"] = Measurement[str].available(
                "True", source="cmdline --trust_remote_code"
            )

        # Flash Attention
        if flag_present(cmdline, "--use_flash_attention_2", "--attn_implementation"):
            details["Flash Attention"] = Measurement[str].available(
                "Requested", source="cmdline --use_flash_attention_2"
            )

        # Low CPU mem usage
        if flag_present(cmdline, "--low_cpu_mem_usage"):
            details["Low CPU Mem Usage"] = Measurement[str].available(
                "True", source="cmdline --low_cpu_mem_usage"
            )

        return details

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _detect_precision(cmdline: list[str]) -> Measurement[str]:
        # Bare flags like --fp16 / --bf16 take priority
        if flag_present(cmdline, "--fp16"):
            return Measurement[str].available("FP16", source="cmdline --fp16")
        if flag_present(cmdline, "--bf16"):
            return Measurement[str].available("BF16", source="cmdline --bf16")

        # Named flags like --torch_dtype float16
        raw = parse_arg(cmdline, "--torch_dtype", "--torch-dtype", "--dtype")
        if raw:
            pretty = _DTYPE_MAP.get(raw.lower(), raw.upper())
            return Measurement[str].available(pretty, source=f"cmdline --torch_dtype={raw}")

        # load_in_8bit / load_in_4bit
        if flag_present(cmdline, "--load_in_8bit"):
            return Measurement[str].available("INT8 (bitsandbytes)", source="cmdline --load_in_8bit")
        if flag_present(cmdline, "--load_in_4bit"):
            return Measurement[str].available("INT4 (bitsandbytes)", source="cmdline --load_in_4bit")

        return Measurement[str].unavailable(
            "Dtype not specified in cmdline — typically defaults to FP32 or model config."
        )
