"""
PluginRegistry — selects the appropriate RuntimePlugin for a process.

The Inspector never uses if-else chains to pick a runtime.
It always asks the registry.

Registering a new plugin
-------------------------
::

    registry = PluginRegistry()
    registry.register(RuntimeKind.VLLM, VLLMPlugin)
    registry.register(RuntimeKind.OLLAMA, OllamaPlugin)

The registry always falls back to UnknownPlugin when no match is found.
"""

from __future__ import annotations

from llm_inspector.models.enums import RuntimeKind
from llm_inspector.plugins.base import RuntimePlugin
from llm_inspector.plugins.unknown import UnknownPlugin


class PluginRegistry:
    """
    Maps RuntimeKind → RuntimePlugin class.

    Thread-safe for read operations (select).  Registration should
    happen once at startup before any inspection begins.
    """

    def __init__(self) -> None:
        self._registry: dict[RuntimeKind, type[RuntimePlugin]] = {}
        # UnknownPlugin is always the fallback — pre-register it.
        self._registry[RuntimeKind.UNKNOWN] = UnknownPlugin

    def register(self, kind: RuntimeKind, plugin_cls: type[RuntimePlugin]) -> None:
        """
        Register *plugin_cls* as the handler for *kind*.

        Args:
            kind:       The RuntimeKind this plugin handles.
            plugin_cls: The plugin class (not an instance).
        """
        self._registry[kind] = plugin_cls

    def select(self, kind: RuntimeKind) -> RuntimePlugin:
        """
        Return an instantiated plugin for *kind*.

        Falls back to UnknownPlugin if *kind* is not registered.
        """
        plugin_cls = self._registry.get(kind, UnknownPlugin)
        return plugin_cls()

    def registered_kinds(self) -> list[RuntimeKind]:
        """Return all RuntimeKind values that have a registered plugin."""
        return list(self._registry.keys())

    def all_plugins(self) -> list[RuntimePlugin]:
        """Return one instance of every registered plugin (for llminspect runtimes)."""
        return [cls() for cls in self._registry.values()]


def build_default_registry() -> PluginRegistry:
    """
    Build a PluginRegistry pre-loaded with all Phase 2 plugins.

    Plugin loading is deferred (import inside function) so missing
    optional dependencies (requests, etc.) don't prevent the tool
    from starting when those runtimes aren't present.
    """
    from llm_inspector.plugins.huggingface import HuggingFacePlugin  # noqa: PLC0415
    from llm_inspector.plugins.ollama import OllamaPlugin  # noqa: PLC0415
    from llm_inspector.plugins.vllm import VLLMPlugin  # noqa: PLC0415

    registry = PluginRegistry()
    registry.register(RuntimeKind.VLLM, VLLMPlugin)
    registry.register(RuntimeKind.HUGGING_FACE, HuggingFacePlugin)
    registry.register(RuntimeKind.OLLAMA, OllamaPlugin)
    # Phase 3: registry.register(RuntimeKind.TENSORRT, TensorRTPlugin)
    # Phase 3: registry.register(RuntimeKind.SGLANG, SGLangPlugin)
    # Phase 3: registry.register(RuntimeKind.LLAMA_CPP, LlamaCppPlugin)
    return registry
