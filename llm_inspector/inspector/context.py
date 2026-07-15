"""
InspectionContext — the frozen shared input for all Phase B collectors.

Built after Phase A completes.  Never mutated.
"""

from __future__ import annotations

from dataclasses import dataclass

from llm_inspector.backends.base import HardwareBackend
from llm_inspector.models.results import HardwareResult, ProcessResult
from llm_inspector.plugins.base import RuntimePlugin


@dataclass(frozen=True)
class InspectionContext:
    """
    Read-only snapshot of everything known after Phase A.

    Passed to every Phase B collector.  Collectors read from this
    context — they never write to it.

    Attributes:
        pid:      The process being inspected.
        process:  Result from ProcessCollector (Phase A).
        hardware: Result from HardwareCollector (Phase A).
        plugin:   Selected plugin for this process's runtime.
        backend:  The active hardware backend (CUDA / CPU / etc.)
    """

    pid: int
    process: ProcessResult
    hardware: HardwareResult
    plugin: RuntimePlugin
    backend: HardwareBackend
