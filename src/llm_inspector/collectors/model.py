"""
ModelCollector — Phase B.

Delegates entirely to the runtime plugin.  The collector is responsible
for error handling and timing; the plugin is responsible for knowing how
to extract model identity from its specific runtime.

Fallback: if the plugin returns None (e.g. UnknownPlugin), all fields
are marked Unavailable with a clear reason.
"""

from __future__ import annotations

from llm_inspector.collectors.base import Collector, CollectorResult
from llm_inspector.models.enums import CollectorPhase
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import ModelResult


class ModelCollector(Collector[ModelResult]):
    """
    Collect model identity from the active runtime plugin.

    Phase B — requires a built InspectionContext with a selected plugin.
    """

    name = "model"
    phase = CollectorPhase.B

    def collect(self, ctx: object) -> CollectorResult[ModelResult]:  # type: ignore[override]
        return super().collect(ctx)

    def _collect(self, ctx: object) -> ModelResult:  # type: ignore[override]
        from llm_inspector.inspector.context import InspectionContext  # noqa: PLC0415

        assert isinstance(ctx, InspectionContext)

        result = ctx.plugin.get_model_info(ctx)
        if result is not None:
            return result

        # Plugin returned None — mark everything unavailable
        reason = (
            f"No model info available from {ctx.plugin.display_name} runtime. "
            "The server does not expose a recognised model endpoint."
        )
        return ModelResult(
            name=Measurement[str].unavailable(reason),
            architecture=Measurement[str].unavailable(reason),
            parameter_count=Measurement[int].unavailable(reason),
            precision=Measurement[str].unavailable(reason),
            context_length=Measurement[int].unavailable(reason),
            tensor_parallel=Measurement[int].unavailable(reason),
            pipeline_parallel=Measurement[int].unavailable(reason),
        )
