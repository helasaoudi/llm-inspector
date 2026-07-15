"""Phase 4 — Activations / Workspace / Other accounting."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from llm_inspector.embedded.collectors import collect_memory_breakdown_embedded
from llm_inspector.embedded.hooks import _tensor_nbytes, install_activation_hooks
from llm_inspector.embedded.streaming import StreamingMetrics
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.results import ComponentName, MemoryBreakdownResult
from llm_inspector.rpc import CollectorRequest
from llm_inspector.serialization import model_from_dict


class _FakeAdapter:
    name = "vllm"

    def weights_bytes(self, ctx):  # noqa: ANN001
        return Measurement[int].available(3_000_000_000, source="test-weights")

    def kv_cache_bytes(self, ctx):  # noqa: ANN001
        return Measurement[int].available(10_000_000_000, source="test-kv")

    def model_name(self, ctx):  # noqa: ANN001
        return Measurement[str].unavailable("n/a")

    def parameter_count(self, ctx):  # noqa: ANN001
        return Measurement[int].unavailable("n/a")

    def precision(self, ctx):  # noqa: ANN001
        return Measurement[str].unavailable("n/a")


class TestStreamingActivationBaseline:
    def test_peak_activation_from_allocated(self) -> None:
        s = StreamingMetrics()
        s.set_baseline_allocated(10_000)
        s.record_allocated(10_000)
        s.record_allocated(15_000)
        s.record_allocated(12_000)
        assert s.peak_activation_bytes == 5_000
        assert s.current_activation_bytes(12_000) == 2_000


class TestPhase4Breakdown:
    def test_workspace_other_activations(self) -> None:
        streaming = StreamingMetrics()
        streaming.set_baseline_allocated(13_000_000_000)

        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        # allocated = weights(3) + kv(10) + activations(0.5) = 13.5
        # reserved = 14.0 → workspace = 0.5
        mock_torch.cuda.memory_allocated.return_value = 13_500_000_000
        mock_torch.cuda.memory_reserved.return_value = 14_000_000_000

        adapter_ctx = (_FakeAdapter(), MagicMock())
        with patch("llm_inspector.embedded.collectors._torch_available", return_value=True):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                data = collect_memory_breakdown_embedded(
                    CollectorRequest(collector="memory-breakdown"),
                    adapter_ctx,
                    streaming,
                )

        result = model_from_dict(MemoryBreakdownResult, data)
        weights = result.get(ComponentName.WEIGHTS)
        kv = result.get(ComponentName.KV_CACHE)
        act = result.get(ComponentName.ACTIVATIONS)
        ws = result.get(ComponentName.WORKSPACE)
        other = result.get(ComponentName.OTHER)

        assert weights is not None and weights.measurement.value == 3_000_000_000
        assert kv is not None and kv.measurement.value == 10_000_000_000
        assert act is not None and act.measurement.is_available
        assert act.measurement.value == 500_000_000  # 13.5 - 13.0 baseline
        assert ws is not None and ws.measurement.value == 500_000_000  # 14 - 13.5
        assert other is not None and other.measurement.is_available
        # other = allocated - W - KV - current_act = 13.5 - 3 - 10 - 0.5 = 0
        assert other.measurement.value == 0
        assert result.total.is_available
        assert result.total.value == 3_000_000_000 + 10_000_000_000 + 500_000_000 + 500_000_000 + 0

    def test_activations_unavailable_without_baseline(self) -> None:
        streaming = StreamingMetrics()  # no baseline
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 13_000_000_000
        mock_torch.cuda.memory_reserved.return_value = 14_000_000_000

        adapter_ctx = (_FakeAdapter(), MagicMock())
        with patch("llm_inspector.embedded.collectors._torch_available", return_value=True):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                data = collect_memory_breakdown_embedded(
                    CollectorRequest(collector="memory-breakdown"),
                    adapter_ctx,
                    streaming,
                )
        result = model_from_dict(MemoryBreakdownResult, data)
        act = result.get(ComponentName.ACTIVATIONS)
        assert act is not None
        assert not act.measurement.is_available


class TestHooks:
    def test_tensor_nbytes_nested(self) -> None:
        class _T:
            def numel(self) -> int:
                return 50

            def element_size(self) -> int:
                return 4

        assert _tensor_nbytes(None) == 0
        assert _tensor_nbytes([None, None]) == 0
        assert _tensor_nbytes(_T()) == 200
        assert _tensor_nbytes([_T(), _T()]) == 400

    def test_install_hooks_noop_without_module(self) -> None:
        streaming = StreamingMetrics()
        assert install_activation_hooks(None, streaming) == 0
        assert install_activation_hooks(object(), streaming) == 0
