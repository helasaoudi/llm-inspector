"""
ModelCollector — Phase B.

Resolution order for model identity:
  1. EmbeddedSource (in-process adapter — HF, PyTorch, vLLM engine)
  2. Runtime plugin API (Ollama /api/ps, vLLM /v1/models, openapi…)
  3. /proc/<pid>/maps + environ
  4. Fully Unavailable with an honest explanation

This means any process that has model weights memory-mapped will have its
model name extracted even if the runtime exposes no API — no estimation,
just OS observation.
"""

from __future__ import annotations

from llm_inspector.collectors.base import Collector, CollectorResult
from llm_inspector.models.enums import CollectorPhase
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import ModelResult
from llm_inspector.serialization import model_from_dict
from llm_inspector.sources.resolver import get_resolver


def merge_model_results(
    primary: ModelResult,
    secondary: ModelResult | None,
) -> ModelResult:
    """
    Merge two ModelResults field-by-field.

    ``primary`` wins when a field is available; otherwise ``secondary`` fills
    the gap.  Used to combine embedded adapter data with runtime API data.
    """
    if secondary is None:
        return primary

    kwargs: dict = {}
    for field_name in ModelResult.model_fields:
        first = getattr(primary, field_name)
        second = getattr(secondary, field_name)
        kwargs[field_name] = first if first.is_available else second
    return ModelResult(**kwargs)


def _all_unavailable(reason: str) -> ModelResult:
    kwargs = {
        name: Measurement.unavailable(reason) for name in ModelResult.model_fields
    }
    return ModelResult(**kwargs)


class ModelCollector(Collector[ModelResult]):
    """
    Collect model identity from the active runtime plugin or /proc inspection.

    Phase B — requires a built InspectionContext with a selected plugin.
    """

    name = "model"
    phase = CollectorPhase.B

    def collect(self, ctx: object) -> CollectorResult[ModelResult]:  # type: ignore[override]
        return super().collect(ctx)

    def _collect(self, ctx: object) -> ModelResult:  # type: ignore[override]
        from llm_inspector.inspector.context import InspectionContext  # noqa: PLC0415
        from llm_inspector.utils import procfs  # noqa: PLC0415

        assert isinstance(ctx, InspectionContext)

        # 1. Embedded inspector (highest fidelity when attach() was called)
        embedded: ModelResult | None = None
        embedded_data = get_resolver().fetch("model", ctx)
        if embedded_data is not None:
            embedded = model_from_dict(ModelResult, embedded_data)

        # 2. Runtime plugin API — merged with embedded so API fills gaps
        api_result = ctx.plugin.get_model_info(ctx)
        if embedded is not None and embedded.name.is_available:
            return merge_model_results(embedded, api_result)
        if api_result is not None:
            return api_result

        # 3. /proc/<pid> inspection — works for any process on Linux.
        snap = procfs.snapshot(ctx.process.pid)
        hf_name = snap.hf_model_name
        weights_bytes = snap.total_weights_bytes
        env_model = snap.model_from_env  # (name, env_var) or None

        if (
            not snap.maps_readable
            and not snap.fd_readable
            and not snap.environ_readable
        ):
            reason = (
                f"Cannot read /proc/{ctx.process.pid} (permission denied). "
                "Try running with: sudo $(which llminspect) inspect "
                f"{ctx.process.pid}"
            )
            return _all_unavailable(reason)

        if hf_name or weights_bytes or env_model:
            if hf_name:
                name_meas = Measurement[str].available(
                    hf_name,
                    source=f"/proc/{ctx.process.pid}/maps → HuggingFace cache path",
                )
            elif env_model:
                model_name, env_var = env_model
                name_meas = Measurement[str].available(
                    model_name,
                    source=f"/proc/{ctx.process.pid}/environ → ${env_var}",
                )
            else:
                name_meas = Measurement[str].unavailable(
                    "Model weight files found in /proc maps but no HuggingFace "
                    "cache path pattern or model env var detected."
                )

            frameworks = snap.detected_frameworks
            arch_meas = (
                Measurement[str].available(
                    ", ".join(frameworks),
                    source=f"/proc/{ctx.process.pid}/maps → loaded .so libraries",
                )
                if frameworks
                else Measurement[str].unavailable(
                    "No known framework signature found in /proc maps."
                )
            )

            result = _all_unavailable(
                "Requires embedded attach() or runtime API for tokenizer/config details."
            )
            return result.model_copy(
                update={
                    "name": name_meas,
                    "architecture": arch_meas,
                    "parameter_count": Measurement[int].unavailable(
                        "Parameter count requires runtime API. "
                        "See Memory Breakdown → Weights for raw mapped size."
                    ),
                    "precision": Measurement[str].unavailable(
                        "Precision not detectable from /proc — requires runtime API."
                    ),
                }
            )

        reason = (
            f"No model info available: {ctx.plugin.display_name} runtime exposes no API, "
            "and no model weight files found in /proc maps or fd. "
            "If the model was loaded >minutes ago, file handles may be closed — "
            "the runtime must expose an API to identify it."
        )
        return _all_unavailable(reason)
