"""Built-in quantization methods (v0.6 — weights only)."""

from __future__ import annotations

from llm_inspector.models.enums import RuntimeKind
from llm_inspector.optimization.base import OptimizationContext
from llm_inspector.optimization.models import QualityBand
from llm_inspector.optimization.quantization.base import QuantizationMethod


class FP8Method(QuantizationMethod):
    name = "FP8"
    bits = 8.0
    description = "8-bit floating point weights (E4M3/E5M2 family)."

    def bytes_per_parameter(self) -> float:
        return 1.0

    def expected_quality(self) -> QualityBand:
        return QualityBand.EXCELLENT

    def supports_runtime(self, runtime: RuntimeKind) -> bool:
        return runtime in (
            RuntimeKind.VLLM,
            RuntimeKind.HUGGING_FACE,
            RuntimeKind.SGLANG,
            RuntimeKind.TENSORRT,
            RuntimeKind.UNKNOWN,
            RuntimeKind.FASTAPI,
        )


class INT8Method(QuantizationMethod):
    name = "INT8"
    bits = 8.0
    description = "8-bit integer weight quantization."

    def bytes_per_parameter(self) -> float:
        return 1.0

    def expected_quality(self) -> QualityBand:
        return QualityBand.EXCELLENT


class AWQ4Method(QuantizationMethod):
    name = "AWQ 4-bit"
    bits = 4.0
    description = "Activation-aware Weight Quantization (4-bit) with group scales."

    def bytes_per_parameter(self) -> float:
        return 0.5

    def overhead_bytes(self, ctx: OptimizationContext) -> int:
        # Rough scale overhead: ~2 bytes per group of 128 params
        if ctx.param_count:
            groups = max(1, ctx.param_count // 128)
            return groups * 2
        return 0

    def expected_quality(self) -> QualityBand:
        return QualityBand.VERY_GOOD

    def supports_runtime(self, runtime: RuntimeKind) -> bool:
        return runtime in (
            RuntimeKind.VLLM,
            RuntimeKind.HUGGING_FACE,
            RuntimeKind.SGLANG,
            RuntimeKind.UNKNOWN,
            RuntimeKind.FASTAPI,
        )


class GPTQ4Method(QuantizationMethod):
    name = "GPTQ 4-bit"
    bits = 4.0
    description = "GPTQ 4-bit weight quantization."

    def bytes_per_parameter(self) -> float:
        return 0.5

    def overhead_bytes(self, ctx: OptimizationContext) -> int:
        if ctx.param_count:
            groups = max(1, ctx.param_count // 128)
            return groups * 2
        return 0

    def expected_quality(self) -> QualityBand:
        return QualityBand.GOOD

    def supports_runtime(self, runtime: RuntimeKind) -> bool:
        return runtime in (
            RuntimeKind.VLLM,
            RuntimeKind.HUGGING_FACE,
            RuntimeKind.SGLANG,
            RuntimeKind.UNKNOWN,
            RuntimeKind.FASTAPI,
        )


class GGUFQ4KMMethod(QuantizationMethod):
    name = "GGUF Q4_K_M"
    bits = 4.5  # K-quants are slightly above pure 4-bit on average
    description = "llama.cpp GGUF Q4_K_M k-quant (typical ~4.5 bits effective)."

    def bytes_per_parameter(self) -> float:
        return 4.5 / 8.0

    def expected_quality(self) -> QualityBand:
        return QualityBand.GOOD

    def supports_runtime(self, runtime: RuntimeKind) -> bool:
        # Educational for vLLM: shown as unsupported
        return runtime in (RuntimeKind.OLLAMA, RuntimeKind.LLAMA_CPP)


def builtin_quantization_methods() -> list[QuantizationMethod]:
    return [
        FP8Method(),
        INT8Method(),
        AWQ4Method(),
        GPTQ4Method(),
        GGUFQ4KMMethod(),
    ]
