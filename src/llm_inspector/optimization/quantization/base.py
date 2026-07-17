"""QuantizationMethod — weight-memory projection plugins."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

from llm_inspector.models.enums import RuntimeKind
from llm_inspector.models.measurement import Measurement
from llm_inspector.optimization.base import OptimizationContext
from llm_inspector.optimization.models import QualityBand


def _dtype_bytes(precision: str | None) -> float | None:
    if not precision:
        return None
    p = precision.lower().replace("torch.", "").replace(" ", "")
    mapping = {
        "float32": 4.0,
        "fp32": 4.0,
        "float16": 2.0,
        "fp16": 2.0,
        "half": 2.0,
        "bfloat16": 2.0,
        "bf16": 2.0,
        "float8": 1.0,
        "fp8": 1.0,
        "int8": 1.0,
        "uint8": 1.0,
        "int4": 0.5,
        "fp4": 0.5,
    }
    return mapping.get(p)


class QuantizationMethod(ABC):
    """Single quantization scheme. Projects weight bytes only (v0.6)."""

    name: str
    bits: float
    description: str = ""

    @abstractmethod
    def bytes_per_parameter(self) -> float:
        """Average storage bytes per parameter after quantization."""

    def overhead_bytes(self, ctx: OptimizationContext) -> int:
        """Extra bytes (scales, zeros, packing). Default: none."""
        del ctx
        return 0

    def affects_weights(self) -> bool:
        return True

    def affects_kv_cache(self) -> bool:
        return False

    def supports_runtime(self, runtime: RuntimeKind) -> bool:
        return True

    def expected_quality(self) -> QualityBand:
        return QualityBand.UNKNOWN

    def project_weights(self, ctx: OptimizationContext) -> Measurement[int]:
        """Simulate weight memory from measured params or scale from current weights."""
        if not self.affects_weights():
            if ctx.weights_bytes is None:
                return Measurement[int].unavailable("Current weights not measured.")
            return Measurement[int].available(
                ctx.weights_bytes, source="pass-through (method does not alter weights)"
            )

        bpp = self.bytes_per_parameter()
        overhead = self.overhead_bytes(ctx)

        if ctx.param_count is not None and ctx.param_count > 0:
            raw = math.ceil(ctx.param_count * bpp) + overhead
            return Measurement[int].simulated(
                raw,
                source=(
                    f"projected: {self.name} — "
                    f"ceil(params={ctx.param_count} × {bpp} B) + overhead={overhead}"
                ),
            )

        cur_bpp = _dtype_bytes(ctx.precision)
        if ctx.weights_bytes is not None and cur_bpp and cur_bpp > 0:
            scaled = int(round(ctx.weights_bytes * (bpp / cur_bpp))) + overhead
            return Measurement[int].simulated(
                scaled,
                source=(
                    f"projected: {self.name} — "
                    f"weights×({bpp}/{cur_bpp}) from precision={ctx.precision}"
                ),
            )

        return Measurement[int].unavailable(
            f"{self.name}: need measured parameter_count or (weights + precision)."
        )
