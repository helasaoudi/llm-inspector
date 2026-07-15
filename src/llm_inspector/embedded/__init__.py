"""Embedded inspector — in-process deep runtime observability."""

from llm_inspector.embedded.attach import attach, auto_attach, detach, is_attached

__all__ = ["attach", "auto_attach", "detach", "is_attached"]
