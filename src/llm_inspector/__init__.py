"""LLM Inspector — the htop for LLM inference. Measured. Not guessed."""

from llm_inspector.embedded import attach, auto_attach, detach, is_attached

__version__ = "0.6.0"
__repo__ = "https://github.com/helasaoudi/llm-inspector"

__all__ = [
    "__version__",
    "__repo__",
    "attach",
    "auto_attach",
    "detach",
    "is_attached",
]
