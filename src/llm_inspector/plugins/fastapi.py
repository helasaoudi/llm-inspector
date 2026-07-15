"""
FastAPI / uvicorn runtime plugin.

Targets any Python process serving an ASGI app via uvicorn (or gunicorn).
These are commonly custom inference servers wrapping HuggingFace, LiteLLM,
or other frameworks behind a FastAPI layer.

Data sources (probed in order — first success wins)
----------------------------------------------------
- GET /v1/models        → OpenAI-compatible model list (LiteLLM, vLLM, many others)
- GET /info             → HF Text Generation Inference info endpoint
- GET /model_info       → generic model info endpoint
- Cmdline               → --host / --port to build the base URL
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from llm_inspector.models.enums import RuntimeKind
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import ModelResult
from llm_inspector.plugins.base import RuntimePlugin
from llm_inspector.utils import cmdline as cl
from llm_inspector.utils import http

if TYPE_CHECKING:
    from llm_inspector.inspector.context import InspectionContext
    from llm_inspector.models.results import MemoryBreakdownResult, ProcessResult

_log = logging.getLogger(__name__)

_UNAVAIL_REASON = (
    "FastAPI plugin: no recognised API endpoint found. "
    "The server does not expose /v1/models, /info, or /model_info."
)


class FastAPIPlugin(RuntimePlugin):
    """
    Plugin for custom FastAPI / uvicorn inference servers.

    Probes well-known HTTP endpoints to extract model information.
    If the server does not expose any recognised endpoint, all fields
    are marked Unavailable with an explanation — never estimated.
    """

    kind = RuntimeKind.FASTAPI
    display_name = "FastAPI"
    description = "Custom ASGI inference server (uvicorn / gunicorn)"

    def supports(self, process: "ProcessResult") -> bool:
        return process.runtime_kind == RuntimeKind.FASTAPI

    # ── Public capability methods ─────────────────────────────────────────────

    def get_version(self, ctx: "InspectionContext") -> Measurement[str]:
        base = self._base_url(ctx.process.cmdline)

        # HF TGI exposes version in /info
        info = http.get_json(f"{base}/info")
        if isinstance(info, dict) and "version" in info:
            return Measurement[str].available(
                str(info["version"]),
                source=f"GET {base}/info → .version",
            )

        return Measurement[str].unavailable(
            "FastAPI plugin: server does not expose a version endpoint."
        )

    def get_model_info(self, ctx: "InspectionContext") -> ModelResult | None:
        base = self._base_url(ctx.process.cmdline)
        name = self._probe_model_name(base)
        if name is None:
            return None  # nothing found — UnknownPlugin fallback will mark all unavailable

        return ModelResult(
            name=Measurement[str].available(name, source=self._name_source),
            architecture=Measurement[str].unavailable(
                "FastAPI plugin: architecture not exposed by API."
            ),
            precision=Measurement[str].unavailable(
                "FastAPI plugin: precision not exposed by API."
            ),
            parameter_count=Measurement[int].unavailable(
                "FastAPI plugin: parameter count not exposed by API."
            ),
            context_length=Measurement[int].unavailable(
                "FastAPI plugin: context length not exposed by API."
            ),
            tensor_parallel=Measurement[int].unavailable(
                "FastAPI plugin: tensor parallel not exposed by API."
            ),
            pipeline_parallel=Measurement[int].unavailable(
                "FastAPI plugin: pipeline parallel not exposed by API."
            ),
        )

    def get_runtime_details(self, ctx: "InspectionContext") -> dict[str, Measurement[str]]:
        cmdline = ctx.process.cmdline
        host = cl.detect_host(cmdline, default="0.0.0.0")
        port = cl.detect_port(cmdline, default=8000)
        base = self._base_url(cmdline)

        details: dict[str, Measurement[str]] = {
            "Listening": Measurement[str].available(
                f"{host}:{port}",
                source="cmdline --host / --port",
            ),
        }

        # Try HF TGI /info for extra runtime details
        info = http.get_json(f"{base}/info")
        if isinstance(info, dict):
            for key in ("model_id", "dtype", "max_input_length", "max_total_tokens"):
                if key in info:
                    details[key.replace("_", " ").title()] = Measurement[str].available(
                        str(info[key]),
                        source=f"GET {base}/info → .{key}",
                    )

        return details

    # ── Private helpers ───────────────────────────────────────────────────────

    _name_source: str = ""  # set by _probe_model_name

    def _base_url(self, cmdline: list[str]) -> str:
        host = cl.detect_host(cmdline, default="127.0.0.1")
        # Remap 0.0.0.0 → 127.0.0.1 for local loopback queries
        if host in ("0.0.0.0", "::"):
            host = "127.0.0.1"
        port = cl.detect_port(cmdline, default=8000)
        return f"http://{host}:{port}"

    def _probe_model_name(self, base: str) -> str | None:
        """
        Try well-known endpoints and return the model name if found.
        Sets self._name_source as a side-effect for provenance tracking.
        """
        # 1. OpenAI-compatible /v1/models  (LiteLLM, vLLM, many custom servers)
        data = http.get_json(f"{base}/v1/models")
        if isinstance(data, dict) and "data" in data:
            models = data["data"]
            if models and isinstance(models, list):
                model_id = models[0].get("id")
                if model_id:
                    self._name_source = f"GET {base}/v1/models → .data[0].id"
                    return str(model_id)

        # 2. HF Text Generation Inference /info
        info = http.get_json(f"{base}/info")
        if isinstance(info, dict):
            for key in ("model_id", "model", "name"):
                if key in info:
                    self._name_source = f"GET {base}/info → .{key}"
                    return str(info[key])

        # 3. Generic /model_info
        model_info = http.get_json(f"{base}/model_info")
        if isinstance(model_info, dict):
            for key in ("model_id", "model", "name", "model_name"):
                if key in model_info:
                    self._name_source = f"GET {base}/model_info → .{key}"
                    return str(model_info[key])

        _log.debug("FastAPIPlugin: no model name found at %s", base)
        return None
