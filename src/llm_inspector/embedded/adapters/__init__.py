"""Framework adapters for embedded inspection."""

from llm_inspector.embedded.adapters.registry import bind_context, select_adapter

__all__ = ["bind_context", "select_adapter"]
