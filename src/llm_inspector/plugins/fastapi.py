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

        # OpenAPI schema exposes the app version (info.version)
        schema = http.get_json(f"{base}/openapi.json")
        if isinstance(schema, dict):
            app_info = schema.get("info")
            if isinstance(app_info, dict):
                title = app_info.get("title")
                ver = app_info.get("version")
                if ver:
                    label = f"{title} {ver}" if title else str(ver)
                    return Measurement[str].available(
                        label,
                        source=f"GET {base}/openapi.json → .info",
                    )

        # PyTorch version from process environment (needs read access to /proc)
        from llm_inspector.utils import procfs  # noqa: PLC0415

        snap = procfs.snapshot(ctx.process.pid)
        if snap.pytorch_version:
            return Measurement[str].available(
                f"PyTorch {snap.pytorch_version}",
                source=f"/proc/{ctx.process.pid}/environ → $PYTORCH_VERSION",
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

        # App title/description from OpenAPI schema
        schema = http.get_json(f"{base}/openapi.json")
        if isinstance(schema, dict):
            app_info = schema.get("info")
            if isinstance(app_info, dict):
                if app_info.get("title"):
                    details["Service"] = Measurement[str].available(
                        str(app_info["title"]),
                        source=f"GET {base}/openapi.json → .info.title",
                    )
            paths = schema.get("paths")
            if isinstance(paths, dict):
                details["Endpoints"] = Measurement[str].available(
                    str(len(paths)),
                    source=f"GET {base}/openapi.json → .paths",
                )

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

    # Keys that commonly hold a model name in a JSON response
    _MODEL_NAME_KEYS: tuple[str, ...] = (
        "model_id",
        "model_name",
        "model",
        "model_size",
        "name",
        "id",
        "size",
    )

    def _probe_model_name(self, base: str) -> str | None:
        """
        Try well-known endpoints and return the model name if found.
        Sets self._name_source as a side-effect for provenance tracking.

        Order:
          1. OpenAI-compatible /v1/models
          2. HF TGI /info
          3. Generic /model_info
          4. /openapi.json route discovery → call any model-info endpoint
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
            name = self._extract_name(info)
            if name:
                self._name_source = f"GET {base}/info"
                return name

        # 3. Generic /model_info
        model_info = http.get_json(f"{base}/model_info")
        if isinstance(model_info, dict):
            name = self._extract_name(model_info)
            if name:
                self._name_source = f"GET {base}/model_info"
                return name

        # 4. Discover model-info routes from the OpenAPI schema (FastAPI default).
        #    Any GET path containing 'model' and 'info' (e.g.
        #    /api/v1/whisper/model/info) is probed automatically.
        name = self._probe_via_openapi(base)
        if name:
            return name

        _log.debug("FastAPIPlugin: no model name found at %s", base)
        return None

    def _probe_via_openapi(self, base: str) -> str | None:
        """
        Fetch /openapi.json, find GET model-info routes, call them,
        and extract a model name. Works with any FastAPI server.
        """
        schema = http.get_json(f"{base}/openapi.json")
        if not isinstance(schema, dict):
            return None
        paths = schema.get("paths")
        if not isinstance(paths, dict):
            return None

        # Candidate GET routes that likely expose model identity
        candidates: list[str] = []
        for route, methods in paths.items():
            if not isinstance(methods, dict) or "get" not in methods:
                continue
            low = route.lower()
            has_param = "{" in route  # skip routes needing path params
            if has_param:
                continue
            if ("model" in low and "info" in low) or low.endswith("/model"):
                candidates.insert(0, route)  # highest priority
            elif "model" in low or low.endswith("/info"):
                candidates.append(route)

        for route in candidates:
            resp = http.get_json(f"{base}{route}")
            if isinstance(resp, dict):
                name = self._extract_name(resp)
                if name:
                    self._name_source = f"GET {base}{route} (via /openapi.json discovery)"
                    return name

        return None

    def _extract_name(self, obj: dict) -> str | None:
        """Extract a model name from a JSON dict using known keys."""
        for key in self._MODEL_NAME_KEYS:
            val = obj.get(key)
            if val and isinstance(val, (str, int)):
                return str(val)
        # Nested: some servers wrap under 'model' or 'data'
        for wrap in ("model", "data", "info"):
            inner = obj.get(wrap)
            if isinstance(inner, dict):
                nested = self._extract_name(inner)
                if nested:
                    return nested
        return None
