"""
ModelCollector — Phase B.

Resolution order for model identity:
  1. Runtime plugin (API call — highest fidelity)
  2. /proc/<pid>/maps  — model weight files memory-mapped by the process
                         (HuggingFace cache paths reveal model name + weights size)
  3. Fully Unavailable with an honest explanation

This means any process that has model weights memory-mapped will have its
model name extracted even if the runtime exposes no API — no estimation,
just OS observation.
"""

from __future__ import annotations

from llm_inspector.collectors.base import Collector, CollectorResult
from llm_inspector.models.enums import CollectorPhase
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import ModelResult


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

        # 1. Try plugin (API-based, highest fidelity)
        result = ctx.plugin.get_model_info(ctx)
        if result is not None:
            return result

        # 2. Fallback: /proc/<pid>/maps — works for any process on Linux
        #    that has model weights memory-mapped (HuggingFace, llama.cpp, etc.)
        snap = procfs.snapshot(ctx.process.pid)
        hf_name = snap.hf_model_name
        weights_bytes = snap.total_weights_bytes

        if not snap.maps_readable and not snap.fd_readable:
            # /proc files unreadable — likely a different-user process
            reason = (
                f"Cannot read /proc/{ctx.process.pid}/maps (permission denied). "
                "Try running llminspect with sudo for cross-user inspection."
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

        if hf_name or weights_bytes:
            name_meas: Measurement[str]
            if hf_name:
                name_meas = Measurement[str].available(
                    hf_name,
                    source=f"/proc/{ctx.process.pid}/maps → HuggingFace cache path",
                )
            else:
                name_meas = Measurement[str].unavailable(
                    "Model weight files found in /proc maps but no HuggingFace "
                    "cache path pattern detected."
                )

            # Parameter count from weight file size (rough, but measured)
            # Typical: fp16 → 2 bytes/param, bf16 → 2, int4 → 0.5
            param_meas: Measurement[int]
            if weights_bytes:
                # We don't know precision yet — defer to Unavailable rather
                # than estimate. The bytes are shown in Memory Breakdown instead.
                param_meas = Measurement[int].unavailable(
                    "Parameter count requires knowing precision. "
                    "See Memory Breakdown → Weights for raw size."
                )
            else:
                param_meas = Measurement[int].unavailable(
                    "No model weight files found in /proc maps."
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
