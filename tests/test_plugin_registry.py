"""Tests for PluginRegistry."""

from llm_inspector.models.enums import RuntimeKind
from llm_inspector.plugins.registry import PluginRegistry
from llm_inspector.plugins.unknown import UnknownPlugin


class TestPluginRegistry:
    def test_unknown_kind_returns_unknown_plugin(self):
        registry = PluginRegistry()
        plugin = registry.select(RuntimeKind.UNKNOWN)
        assert isinstance(plugin, UnknownPlugin)

    def test_unregistered_kind_falls_back_to_unknown(self):
        registry = PluginRegistry()
        plugin = registry.select(RuntimeKind.VLLM)
        assert isinstance(plugin, UnknownPlugin)

    def test_register_and_select(self):
        class FakePlugin(UnknownPlugin):
            kind = RuntimeKind.VLLM
            display_name = "Fake vLLM"

        registry = PluginRegistry()
        registry.register(RuntimeKind.VLLM, FakePlugin)
        plugin = registry.select(RuntimeKind.VLLM)
        assert isinstance(plugin, FakePlugin)

    def test_all_plugins_returns_instances(self):
        registry = PluginRegistry()
        plugins = registry.all_plugins()
        assert len(plugins) >= 1
        assert all(hasattr(p, "display_name") for p in plugins)

    def test_unknown_plugin_supports_everything(self):
        plugin = UnknownPlugin()
        assert plugin.supports(None) is True

    def test_unknown_plugin_capability_methods_return_empty(self) -> None:
        plugin = UnknownPlugin()
        assert plugin.get_model_info(None) is None  # type: ignore[arg-type]
        assert plugin.get_memory_breakdown(None) is None  # type: ignore[arg-type]
        assert plugin.get_runtime_details(None) == {}  # type: ignore[arg-type]
        assert not plugin.get_version(None).is_available  # type: ignore[arg-type]
