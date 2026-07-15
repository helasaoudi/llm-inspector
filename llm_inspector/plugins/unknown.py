"""UnknownPlugin — always-available fallback for unrecognised runtimes."""

from __future__ import annotations

from llm_inspector.models.enums import RuntimeKind
from llm_inspector.plugins.base import RuntimePlugin


class UnknownPlugin(RuntimePlugin):
    """
    Catch-all plugin for GPU-attached processes with no recognised runtime.

    All capability methods return None / empty — collectors mark their
    fields as Unavailable. This is the correct answer: we don't know
    the runtime, so we cannot measure its internals.
    """

    kind = RuntimeKind.UNKNOWN
    display_name = "Unknown"
    description = "Fallback for GPU processes with no recognised runtime signature."

    def supports(self, process: object) -> bool:
        return True  # Always accepts — it is the last resort.
