"""
Ollama runtime plugin.

Data sources
------------
- GET /api/ps      → loaded models, VRAM usage per model
- GET /api/version → Ollama server version
- Cmdline args     → host / port overrides

Works identically on macOS (primary Ollama platform) and Linux/DGX.
On macOS, Ollama uses Metal — GPU memory appears via VRAM tracking in /api/ps.

Default Ollama address: http://127.0.0.1:11434
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
from llm_inspector.utils.cmdline import detect_host, detect_port
from llm_inspector.utils.http import get_json

if TYPE_CHECKING:
    from llm_inspector.inspector.context import InspectionContext
    from llm_inspector.models.results import ProcessResult

_LOG = logging.getLogger(__name__)

_DEFAULT_HOST = "127.0.0.1"
_DEFAULT_PORT = 11434


class OllamaPlugin(RuntimePlugin):
    """
    Inspects live Ollama servers via its REST management API.

    Phase 2 capabilities:
      ✓ Model name, size_vram from /api/ps
      ✓ Server version from /api/version
      ✓ VRAM used as "Weights" breakdown component (closest available metric)
      ✗ Activations, KV Cache, Workspace: not exposed by Ollama API
    """

    kind = RuntimeKind.OLLAMA
    display_name = "Ollama"
    description = "Local LLM server with Metal/CUDA/CPU backends."

    def supports(self, process: "ProcessResult") -> bool:
        cmdline_str = " ".join(process.cmdline).lower()
        return "ollama" in cmdline_str

    # ── Capability methods ────────────────────────────────────────────────────

    def get_model_info(self, ctx: "InspectionContext") -> ModelResult | None:
        base_url = self._base_url(ctx.process.cmdline)
        data = get_json(f"{base_url}/api/ps")
        model = self._first_model(data)

        if model:
            name = model.get("name") or model.get("model")
            name_m = (
                Measurement[str].available(name, source=f"Ollama GET {base_url}/api/ps → name")
                if name
                else Measurement[str].unavailable("Model name missing from /api/ps response.")
            )

            # Ollama reports details.parameter_size e.g. "8B", "70B"
            details = model.get("details", {})
            param_size = details.get("parameter_size")
            param_m = (
                Measurement[int].unavailable(
                    f"Ollama reports parameter_size='{param_size}' (not a byte count). "
                    "Exact count not available."
                )
            )

            # family → architecture approximation
            family = details.get("family", "")
            arch_m = (
                Measurement[str].available(family, source=f"Ollama GET {base_url}/api/ps → details.family")
                if family
                else Measurement[str].unavailable("Architecture family not returned by /api/ps.")
            )

            # quantization_level → precision
            quant = details.get("quantization_level")
            precision_m = (
                Measurement[str].available(
                    quant, source=f"Ollama GET {base_url}/api/ps → details.quantization_level"
                )
                if quant
                else Measurement[str].unavailable("Quantization level not returned by /api/ps.")
            )

            # context length
            ctx_size = model.get("context_window")
            ctx_len_m = (
                Measurement[int].available(
                    int(ctx_size),
                    source=f"Ollama GET {base_url}/api/ps → context_window",
                )
                if ctx_size is not None
                else Measurement[int].unavailable("context_window not returned by /api/ps.")
            )
        else:
            unavail_reason = (
                f"Ollama /api/ps returned no loaded models (url: {base_url}/api/ps). "
                "Is a model running? Try: ollama run <model>"
            )
            name_m = Measurement[str].unavailable(unavail_reason)
            param_m = Measurement[int].unavailable(unavail_reason)
            arch_m = Measurement[str].unavailable(unavail_reason)
            precision_m = Measurement[str].unavailable(unavail_reason)
            ctx_len_m = Measurement[int].unavailable(unavail_reason)

        return ModelResult(
            name=name_m,
            architecture=arch_m,
            parameter_count=param_m,
            precision=precision_m,
            context_length=ctx_len_m,
            tensor_parallel=Measurement[int].unavailable(
                "Ollama does not expose tensor-parallel configuration."
            ),
            pipeline_parallel=Measurement[int].unavailable(
                "Ollama does not expose pipeline-parallel configuration."
            ),
        )

    def get_memory_breakdown(
        self, ctx: "InspectionContext"
    ) -> MemoryBreakdownResult | None:
        base_url = self._base_url(ctx.process.cmdline)
        data = get_json(f"{base_url}/api/ps")
        model = self._first_model(data)

        components: list[MemoryComponent] = []

        # Ollama exposes size_vram — total VRAM for this model.
        # The closest conceptual match is Weights (it's the loaded model blob).
        # Not a breakdown — just total loaded model VRAM.
        if model and model.get("size_vram") is not None:
            size_vram = int(model["size_vram"])
            components.append(MemoryComponent(
                name=ComponentName.WEIGHTS,
                measurement=Measurement[int].available(
                    size_vram,
                    source=f"Ollama GET {base_url}/api/ps → size_vram",
                ),
                description="Total model VRAM as reported by Ollama (includes weights and runtime overhead).",
                order=COMPONENT_ORDER[ComponentName.WEIGHTS],
            ))
        else:
            components.append(MemoryComponent(
                name=ComponentName.WEIGHTS,
                measurement=Measurement[int].unavailable(
                    "size_vram not returned by /api/ps — is a model currently loaded?"
                ),
                order=COMPONENT_ORDER[ComponentName.WEIGHTS],
            ))

        # Other components: Ollama does not expose them
        for name, note in (
            (ComponentName.KV_CACHE, "Ollama does not expose KV cache allocation."),
            (ComponentName.ACTIVATIONS, "Ollama does not expose activation memory."),
            (ComponentName.WORKSPACE, "Ollama does not expose workspace memory."),
            (ComponentName.OTHER, "Not available from Ollama /api/ps."),
        ):
            components.append(MemoryComponent(
                name=name,
                measurement=Measurement[int].unavailable(note),
                order=COMPONENT_ORDER[name],
            ))

        measured = [c.measurement.value for c in components if c.measurement.is_available and c.measurement.value]
        total_m = (
            Measurement[int].available(sum(measured), source="Ollama /api/ps → size_vram")
            if measured
            else Measurement[int].unavailable("/api/ps did not return size_vram.")
        )

        return MemoryBreakdownResult(components=components, total=total_m)

    def get_runtime_details(
        self, ctx: "InspectionContext"
    ) -> dict[str, Measurement[str]]:
        base_url = self._base_url(ctx.process.cmdline)
        details: dict[str, Measurement[str]] = {}
        data = get_json(f"{base_url}/api/ps")
        model = self._first_model(data)

        if model:
            expires = model.get("expires_at")
            if expires:
                details["Model Expires"] = Measurement[str].available(
                    expires, source=f"Ollama GET {base_url}/api/ps → expires_at"
                )

            details_obj = model.get("details", {})
            if details_obj.get("families"):
                details["Model Family"] = Measurement[str].available(
                    ", ".join(details_obj["families"]),
                    source=f"Ollama GET {base_url}/api/ps → details.families",
                )

        return details

    def get_version(self, ctx: "InspectionContext") -> Measurement[str]:
        base_url = self._base_url(ctx.process.cmdline)
        data = get_json(f"{base_url}/api/version")
        if isinstance(data, dict) and "version" in data:
            return Measurement[str].available(
                data["version"],
                source=f"Ollama GET {base_url}/api/version",
            )
        return Measurement[str].unavailable(
            f"Ollama /api/version not reachable at {base_url}."
        )

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _base_url(cmdline: list[str]) -> str:
        host = detect_host(cmdline, _DEFAULT_HOST)
        port = detect_port(cmdline, _DEFAULT_PORT)
        return f"http://{host}:{port}"

    @staticmethod
    def _first_model(data: dict | list | None) -> dict | None:
        if not isinstance(data, dict):
            return None
        models = data.get("models", [])
        if models and isinstance(models[0], dict):
            return models[0]
        return None
