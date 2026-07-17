"""Quantization optimization strategy family."""

from __future__ import annotations

from llm_inspector.models.measurement import Measurement
from llm_inspector.optimization.base import OptimizationContext, OptimizationStrategy
from llm_inspector.optimization.models import (
    OptimizationGroup,
    OptimizationKind,
    OptimizationScenario,
)
from llm_inspector.optimization.quantization.base import QuantizationMethod
from llm_inspector.optimization.quantization.methods import builtin_quantization_methods


class QuantizationStrategy(OptimizationStrategy):
    kind = OptimizationKind.QUANTIZATION
    title = "Quantization"

    def __init__(self, methods: list[QuantizationMethod] | None = None) -> None:
        self._methods = methods or builtin_quantization_methods()

    def analyze(self, ctx: OptimizationContext) -> OptimizationGroup:
        scenarios: list[OptimizationScenario] = []
        note: str | None = None

        if ctx.weights_bytes is None and ctx.param_count is None:
            return OptimizationGroup(
                kind=self.kind,
                title=self.title,
                scenarios=[],
                note=(
                    "Quantization projections require measured Weights or "
                    "parameter_count (embedded attach recommended)."
                ),
            )

        for method in self._methods:
            if not method.supports_runtime(ctx.runtime):
                scenarios.append(
                    OptimizationScenario(
                        kind=self.kind,
                        method_name=method.name,
                        description=method.description,
                        quality=method.expected_quality(),
                        new_total=Measurement[int].unavailable(
                            f"{method.name} is not typically used with "
                            f"{ctx.runtime.value}."
                        ),
                        saved_bytes=Measurement[int].unavailable(
                            f"Unsupported for runtime {ctx.runtime.value}."
                        ),
                    )
                )
                continue

            weights = method.project_weights(ctx)
            if not weights.has_value or weights.value is None:
                scenarios.append(
                    OptimizationScenario(
                        kind=self.kind,
                        method_name=method.name,
                        description=method.description,
                        quality=method.expected_quality(),
                        new_total=Measurement[int].unavailable(
                            weights.reason or "Could not project weights."
                        ),
                        saved_bytes=Measurement[int].unavailable(
                            weights.reason or "Could not project weights."
                        ),
                        details={"weights": weights},
                    )
                )
                continue

            # Pass-through measured non-weight components
            new_total_val = (
                int(weights.value)
                + (ctx.kv_bytes or 0)
                + (ctx.workspace_bytes or 0)
                + (ctx.other_bytes or 0)
                + (ctx.activations_bytes or 0)
            )
            new_total = Measurement[int].simulated(
                new_total_val,
                source=(
                    f"projected: {method.name} weights + measured "
                    "KV/Workspace/Other/Activations"
                ),
            )

            if ctx.current_total is not None:
                saved = max(0, ctx.current_total - new_total_val)
                saved_m = Measurement[int].simulated(
                    saved,
                    source=f"projected: current_total − new_total ({method.name})",
                )
            else:
                saved_m = Measurement[int].unavailable(
                    "Current breakdown total unavailable — cannot compute savings."
                )

            scenarios.append(
                OptimizationScenario(
                    kind=self.kind,
                    method_name=method.name,
                    description=method.description,
                    quality=method.expected_quality(),
                    new_total=new_total,
                    saved_bytes=saved_m,
                    details={"weights": weights},
                )
            )

        return OptimizationGroup(
            kind=self.kind,
            title=self.title,
            scenarios=scenarios,
            note=note,
        )
