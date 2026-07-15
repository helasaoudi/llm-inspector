"""
ProcessCollector — Phase A.

Identifies the running inference process from OS primitives.
Always runs first.  If this collector fails, the inspection aborts.
"""

from __future__ import annotations

import re

from llm_inspector.collectors.base import Collector, CollectorResult
from llm_inspector.models.enums import CollectorPhase, RuntimeKind
from llm_inspector.models.results import ProcessResult
from llm_inspector.utils import proc as proc_utils

# ── Runtime detection heuristics ─────────────────────────────────────────────
# Each entry: (RuntimeKind, list_of_regex_patterns).
# Evaluated in order; first match wins.
_RUNTIME_RULES: list[tuple[RuntimeKind, list[str]]] = [
    (RuntimeKind.VLLM,          [r"\bvllm\b", r"vllm\.entrypoints", r"VLLM::EngineCore"]),
    (RuntimeKind.OLLAMA,        [r"\bollama\b"]),
    (RuntimeKind.TENSORRT,      [r"tritonserver", r"trtllm", r"tensorrt_llm"]),
    (RuntimeKind.SGLANG,        [r"\bsglang\b"]),
    (RuntimeKind.LLAMA_CPP,     [r"llama[-_]server", r"llama[-_]cli", r"llama_cpp"]),
    (RuntimeKind.HUGGING_FACE,  [r"\btransformers\b", r"from_pretrained"]),
    # uvicorn/FastAPI — custom inference API servers served via ASGI
    (RuntimeKind.FASTAPI,       [r"\buvicorn\b", r"\bfastapi\b", r"\bgunicorn\b"]),
]

_COMPILED_RULES: list[tuple[RuntimeKind, list[re.Pattern[str]]]] = [
    (kind, [re.compile(p, re.IGNORECASE) for p in patterns])
    for kind, patterns in _RUNTIME_RULES
]


def detect_runtime(cmdline: list[str]) -> RuntimeKind:
    """
    Apply heuristic rules to a process argv list to detect the runtime.

    Returns RuntimeKind.UNKNOWN if no rule matches.
    """
    joined = " ".join(cmdline)
    for kind, patterns in _COMPILED_RULES:
        if any(rx.search(joined) for rx in patterns):
            return kind
    return RuntimeKind.UNKNOWN


class ProcessCollector(Collector[ProcessResult]):
    """
    Read OS-level metadata for a given PID.

    Phase A — runs before InspectionContext is built.
    Inputs: pid (int) only.
    """

    name = "process"
    phase = CollectorPhase.A

    def collect(self, pid: int) -> CollectorResult[ProcessResult]:  # type: ignore[override]
        return super().collect(pid)

    def _collect(self, pid: int) -> ProcessResult:  # type: ignore[override]
        cmdline = proc_utils.read_cmdline(pid)
        if not cmdline:
            raise RuntimeError(
                f"Cannot read cmdline for PID {pid}. "
                "Process may have exited or access is denied."
            )

        return ProcessResult(
            pid=pid,
            ppid=proc_utils.read_ppid(pid),
            name=proc_utils.read_name(pid),
            exe=proc_utils.read_exe(pid),
            cmdline=cmdline,
            start_time=proc_utils.read_start_time(pid),
            uptime_seconds=proc_utils.read_uptime(pid),
            runtime_kind=detect_runtime(cmdline),
        )
