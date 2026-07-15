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
        embedded_data = get_resolver().fetch("model", ctx)
        if embedded_data is not None:
            embedded = model_from_dict(ModelResult, embedded_data)
            if embedded.name.is_available:
                return embedded

        # 2. Runtime plugin API
        result = ctx.plugin.get_model_info(ctx)
        if result is not None:
            return result

        # 3. /proc/<pid> inspection — works for any process on Linux.
        #    Model name can come from either:
        #      a) memory-mapped weight files (HuggingFace cache path), or
        #      b) environment variables (MODEL_NAME, WHISPER_MODEL, etc.)
        snap = procfs.snapshot(ctx.process.pid)
        hf_name = snap.hf_model_name
        weights_bytes = snap.total_weights_bytes
        env_model = snap.model_from_env  # (name, env_var) or None

        # Only treat as permission error if we could read *nothing* at all.
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
            return ModelResult(
                name=Measurement[str].unavailable(reason),
                architecture=Measurement[str].unavailable(reason),
                parameter_count=Measurement[int].unavailable(reason),
                precision=Measurement[str].unavailable(reason),
                context_length=Measurement[int].unavailable(reason),
                tensor_parallel=Measurement[int].unavailable(reason),
                pipeline_parallel=Measurement[int].unavailable(reason),
            )

        if hf_name or weights_bytes or env_model:
            # Model name — prefer HF cache path, then env var
            name_meas: Measurement[str]
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

            param_meas = Measurement[int].unavailable(
                "Parameter count requires runtime API. "
                "See Memory Breakdown → Weights for raw mapped size."
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

            return ModelResult(
                name=name_meas,
                architecture=arch_meas,
                precision=Measurement[str].unavailable(
                    "Precision not detectable from /proc — requires runtime API."
                ),
                parameter_count=param_meas,
                context_length=Measurement[int].unavailable(
                    "Context length not detectable from /proc — requires runtime API."
                ),
                tensor_parallel=Measurement[int].unavailable(
                    "Tensor parallel not detectable from /proc — requires runtime API."
                ),
                pipeline_parallel=Measurement[int].unavailable(
                    "Pipeline parallel not detectable from /proc — requires runtime API."
                ),
            )

        # 3. Nothing found anywhere — /proc was readable but no weight files present
        #    This is common when model was loaded long ago: torch.load() reads
        #    the file into GPU memory and closes the handle. After that, no
        #    trace remains in /proc/maps or /proc/fd.
        reason = (
            f"No model info available: {ctx.plugin.display_name} runtime exposes no API, "
            "and no model weight files found in /proc maps or fd. "
            "If the model was loaded >minutes ago, file handles may be closed — "
            "the runtime must expose an API to identify it."
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
