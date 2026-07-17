"""Recommendation text from measured breakdown + projected scenarios."""

from __future__ import annotations

from llm_inspector.optimization.base import OptimizationContext
from llm_inspector.optimization.models import (
    OptimizationGroup,
    OptimizationRecommendation,
    OptimizationScenario,
)


def _best_scenario(scenarios: list[OptimizationScenario]) -> OptimizationScenario | None:
    """Prefer largest savings among scenarios with a projected total."""
    ranked = [
        s
        for s in scenarios
        if s.saved_bytes.has_value
        and s.saved_bytes.value is not None
        and s.new_total.has_value
    ]
    if not ranked:
        return None
    return max(ranked, key=lambda s: s.saved_bytes.value or 0)


def build_recommendation(
    ctx: OptimizationContext,
    groups: list[OptimizationGroup],
) -> OptimizationRecommendation:
    warnings: list[str] = []
    bottleneck: str | None = None

    components = [
        ("Weights", ctx.weights_bytes),
        ("KV Cache", ctx.kv_bytes),
        ("Workspace", ctx.workspace_bytes),
        ("Other", ctx.other_bytes),
        ("Activations", ctx.activations_bytes),
    ]
    present = [(n, v) for n, v in components if v is not None and v > 0]
    if present:
        bottleneck_name, bottleneck_val = max(present, key=lambda x: x[1])
        bottleneck = bottleneck_name
        total = ctx.current_total or sum(v for _, v in present)
        if total > 0 and bottleneck_val / total >= 0.40:
            warnings.append(
                f"Your largest memory consumer is {bottleneck_name} "
                f"({bottleneck_val / total:.0%} of measured breakdown). "
                "Weight quantization alone may not remove the bottleneck."
            )

    quant = next((g for g in groups if g.title == "Quantization"), None)
    scenarios = quant.scenarios if quant else []
    best = _best_scenario(scenarios)

    if best is None:
        return OptimizationRecommendation(
            summary=(
                "Not enough measured data to recommend a quantization method. "
                "Use embedded attach() for Weights and parameter_count."
            ),
            warnings=warnings,
            bottleneck=bottleneck,
        )

    saved = best.saved_bytes.value or 0
    new_total = best.new_total.value
    summary = (
        f"{best.method_name} is the best trade-off "
        f"(saves {saved / 1024**3:.1f} GB"
        + (
            f", projected total {new_total / 1024**3:.1f} GB"
            if new_total is not None
            else ""
        )
        + f", typical quality: {best.quality.value})."
    )

    if (
        bottleneck == "KV Cache"
        and ctx.kv_bytes is not None
        and new_total is not None
        and ctx.kv_bytes / max(new_total, 1) >= 0.45
    ):
        warnings.append(
            "After weight quantization, KV Cache would still dominate. "
            "Consider KV-cache quantization if your runtime supports it."
        )

    if ctx.vram_total is not None and new_total is not None:
        if new_total <= ctx.vram_total:
            warnings.append(
                f"Projected total fits device VRAM "
                f"({ctx.vram_total / 1024**3:.1f} GB)."
            )
        else:
            warnings.append(
                f"Projected total still exceeds device VRAM "
                f"({ctx.vram_total / 1024**3:.1f} GB)."
            )

    return OptimizationRecommendation(
        summary=summary,
        warnings=warnings,
        suggested_method=best.method_name,
        bottleneck=bottleneck,
    )
