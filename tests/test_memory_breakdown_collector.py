"""Tests for MemoryBreakdownCollector merge + embedded priority."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from llm_inspector.backends.cpu import CPUBackend
from llm_inspector.collectors.memory_breakdown import (
    MemoryBreakdownCollector,
    merge_memory_breakdown,
)
from llm_inspector.inspector.context import InspectionContext
from llm_inspector.models.enums import BackendKind, RuntimeKind
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import (
    COMPONENT_ORDER,
    ComponentName,
    HardwareResult,
    MemoryBreakdownResult,
    MemoryComponent,
    ProcessResult,
)
from llm_inspector.plugins.unknown import UnknownPlugin


def _comp(name: str, value: int | None, reason: str = "n/a") -> MemoryComponent:
    meas = (
        Measurement[int].available(value, source="test")
        if value is not None
        else Measurement[int].unavailable(reason)
    )
    return MemoryComponent(
        name=name,
        measurement=meas,
        order=COMPONENT_ORDER.get(name, 99),
    )


def _breakdown(
    weights: int | None = None,
    kv: int | None = None,
) -> MemoryBreakdownResult:
    comps = [
        _comp(ComponentName.WEIGHTS, weights),
        _comp(ComponentName.KV_CACHE, kv),
        _comp(ComponentName.ACTIVATIONS, None),
    ]
    measured = [c.measurement.value for c in comps if c.measurement.is_available]
    total = (
        Measurement[int].available(sum(measured), source="sum")  # type: ignore[arg-type]
        if measured
        else Measurement[int].unavailable("none")
    )
    return MemoryBreakdownResult(components=comps, total=total)


def _make_ctx(plugin=None) -> InspectionContext:
    return InspectionContext(
        pid=1,
        process=ProcessResult(pid=1, cmdline=[], runtime_kind=RuntimeKind.UNKNOWN),
        hardware=HardwareResult(backend=BackendKind.CPU),
        plugin=plugin or UnknownPlugin(),
        backend=CPUBackend(),
    )


class TestMergeMemoryBreakdown:
    def test_primary_wins_available_fields(self) -> None:
        primary = _breakdown(weights=100, kv=None)
        secondary = _breakdown(weights=999, kv=50)
        merged = merge_memory_breakdown(primary, secondary)
        assert merged.get(ComponentName.WEIGHTS).measurement.value == 100
        assert merged.get(ComponentName.KV_CACHE).measurement.value == 50


class TestMemoryBreakdownCollector:
    @patch("llm_inspector.collectors.memory_breakdown.get_resolver")
    def test_embedded_preferred_over_plugin(self, mock_resolver: MagicMock) -> None:
        embedded = _breakdown(weights=3_000_000_000, kv=5_000_000_000)
        mock_resolver.return_value.fetch.return_value = embedded.model_dump(mode="json")

        plugin = MagicMock(spec=UnknownPlugin)
        plugin.get_memory_breakdown.return_value = _breakdown(weights=1, kv=1)
        plugin.display_name = "Test"

        result = MemoryBreakdownCollector().collect(_make_ctx(plugin))
        assert result.succeeded
        assert result.data is not None
        assert result.data.get(ComponentName.WEIGHTS).measurement.value == 3_000_000_000
        assert result.data.get(ComponentName.KV_CACHE).measurement.value == 5_000_000_000
