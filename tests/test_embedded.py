"""Tests for embedded attach and memory collector."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from llm_inspector.embedded.attach import attach, detach, is_attached
from llm_inspector.embedded.collectors import collect_memory_embedded
from llm_inspector.embedded.streaming import StreamingMetrics
from llm_inspector.rpc import CollectorRequest
from llm_inspector.serialization import model_from_dict
from llm_inspector.models.results import MemoryResult


class FakeTensor:
    def __init__(self, numel: int, element_size: int = 2, dtype: str = "float16"):
        self.numel_value = numel
        self._element_size = element_size
        self.dtype = dtype

    def numel(self) -> int:
        return self.numel_value

    @property
    def element_size(self) -> int:
        return self._element_size


class FakeParameter:
    def __init__(self, numel: int):
        self._t = FakeTensor(numel)

    def numel(self) -> int:
        return self._t.numel()

    @property
    def dtype(self):
        return self._t.dtype

    @property
    def element_size(self) -> int:
        return self._t._element_size


class FakeModule:
    """Minimal nn.Module stand-in for adapter tests."""

    def __init__(self, params: int = 1000):
        self._params = [FakeParameter(params)]
        self.config = MagicMock()
        self.config._name_or_path = "meta-llama/Llama-3-8B"
        self.config.architectures = ["LlamaForCausalLM"]
        self.config.model_type = "llama"

    def parameters(self):
        return iter(self._params)


class TestStreamingMetrics:
    def test_peak_tracking(self):
        m = StreamingMetrics()
        m.record_allocated(1000)
        m.record_allocated(500)
        m.record_allocated(2000)
        snap = m.snapshot()
        assert snap["peak_allocated_bytes"] == 2000


class TestEmbeddedMemoryCollector:
    def test_no_cuda_returns_unavailable(self):
        with patch("llm_inspector.embedded.collectors._torch_available", return_value=False):
            data = collect_memory_embedded(CollectorRequest(collector="memory"), StreamingMetrics())
            result = model_from_dict(MemoryResult, data)
            assert not result.gpu_allocated.is_available

    def test_with_mock_torch(self):
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.memory_allocated.return_value = 1_000_000_000
        mock_torch.cuda.memory_reserved.return_value = 2_000_000_000
        mock_torch.cuda.max_memory_allocated.return_value = 1_500_000_000

        streaming = StreamingMetrics()
        streaming.record_allocated(1_800_000_000)

        with patch("llm_inspector.embedded.collectors._torch_available", return_value=True):
            with patch.dict("sys.modules", {"torch": mock_torch}):
                data = collect_memory_embedded(CollectorRequest(collector="memory"), streaming)
                result = model_from_dict(MemoryResult, data)
                assert result.gpu_allocated.is_available
                assert result.gpu_allocated.value == 1_000_000_000
                assert result.peak.is_available
                assert result.peak.value >= 1_800_000_000


class TestAttach:
    def test_attach_and_detach(self):
        if is_attached():
            detach()

        with tempfile.TemporaryDirectory() as tmp:
            sock = Path(tmp) / f"{os.getpid()}.sock"
            attach(model=FakeModule(), socket_path=str(sock))
            assert is_attached()
            assert sock.exists()
            detach()
            assert not is_attached()

    def test_double_attach_is_noop(self):
        if is_attached():
            detach()
        with tempfile.TemporaryDirectory() as tmp:
            sock = Path(tmp) / f"{os.getpid()}.sock"
            attach(model=FakeModule(), socket_path=str(sock))
            attach(model=FakeModule(), socket_path=str(sock))  # should not raise
            detach()


class TestHuggingFaceAdapter:
    def test_supports_pretrained_model(self):
        from llm_inspector.embedded.adapters.huggingface import HuggingFaceAdapter

        adapter = HuggingFaceAdapter()
        # Without transformers installed, supports() returns False
        assert adapter.supports(FakeModule()) is False

    def test_pytorch_adapter_supports_fake_module(self):
        from llm_inspector.embedded.adapters.pytorch import PyTorchAdapter

        # FakeModule is not nn.Module — adapter uses isinstance check
        adapter = PyTorchAdapter()
        assert adapter.supports(FakeModule()) is False
