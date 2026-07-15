"""Tests for ModelCollector — delegates to plugin."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_inspector.backends.cpu import CPUBackend
from llm_inspector.collectors.model import ModelCollector
from llm_inspector.inspector.context import InspectionContext
from llm_inspector.models.enums import BackendKind, CollectorStatus, RuntimeKind
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import HardwareResult, ModelResult, ProcessResult
from llm_inspector.plugins.unknown import UnknownPlugin


def _make_ctx(plugin=None) -> InspectionContext:
    process = ProcessResult(pid=1, cmdline=[], runtime_kind=RuntimeKind.UNKNOWN)
    hardware = HardwareResult(backend=BackendKind.CPU)
    backend = CPUBackend()
    return InspectionContext(
        pid=1,
        process=process,
        hardware=hardware,
        plugin=plugin or UnknownPlugin(),
        backend=backend,
    )


class TestModelCollector:
    def test_unknown_plugin_returns_all_unavailable(self) -> None:
        ctx = _make_ctx(UnknownPlugin())
        result = ModelCollector().collect(ctx)
        assert result.succeeded
        assert result.data is not None
        m = result.data
        assert not m.name.is_available
        assert not m.precision.is_available
        assert not m.architecture.is_available

    def test_plugin_result_is_passed_through(self) -> None:
        plugin = MagicMock(spec=UnknownPlugin)
        plugin.get_model_info.return_value = ModelResult(
            name=Measurement[str].available("llama3", source="test"),
            architecture=Measurement[str].unavailable("n/a"),
            parameter_count=Measurement[int].unavailable("n/a"),
            precision=Measurement[str].available("BF16", source="test"),
            context_length=Measurement[int].unavailable("n/a"),
            tensor_parallel=Measurement[int].unavailable("n/a"),
            pipeline_parallel=Measurement[int].unavailable("n/a"),
        )
        plugin.display_name = "TestPlugin"
        ctx = _make_ctx(plugin)
        result = ModelCollector().collect(ctx)
        assert result.succeeded
        assert result.data is not None
        assert result.data.name.value == "llama3"
        assert result.data.precision.value == "BF16"

    def test_plugin_returns_none_falls_back_to_unavailable(self) -> None:
        plugin = MagicMock(spec=UnknownPlugin)
        plugin.get_model_info.return_value = None
        plugin.display_name = "NonePlugin"
        ctx = _make_ctx(plugin)
        result = ModelCollector().collect(ctx)
        assert result.succeeded
        assert result.data is not None
        assert not result.data.name.is_available
