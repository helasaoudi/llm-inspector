"""Tests for VLLMAdapter embedded introspection."""

from __future__ import annotations

from unittest.mock import MagicMock

from llm_inspector.embedded.adapters.base import AdapterContext
from llm_inspector.embedded.adapters.vllm import VLLMAdapter, _is_vllm_engine


class _FakeArchConfig:
    architectures = ["WhisperForConditionalGeneration"]


class _FakeHFConfig:
    architectures = ["WhisperForConditionalGeneration"]
    model_type = "whisper"
    num_parameters = 1_550_000_000


class _FakeModelConfig:
    model = "openai/whisper-large-v3"
    dtype = "torch.float16"
    max_model_len = 448
    hf_config = _FakeHFConfig()
    architecture = _FakeArchConfig()


class _FakeParallelConfig:
    tensor_parallel_size = 1
    pipeline_parallel_size = 1


class _FakeVllmConfig:
    model_config = _FakeModelConfig()
    parallel_config = _FakeParallelConfig()


class _FakeEngineCore:
    __module__ = "vllm.v1.engine.core"

    def __init__(self) -> None:
        self.vllm_config = _FakeVllmConfig()


class TestIsVllmEngine:
    def test_engine_core_detected(self) -> None:
        assert _is_vllm_engine(_FakeEngineCore())

    def test_non_vllm_rejected(self) -> None:
        assert not _is_vllm_engine(object())


class TestVLLMAdapter:
    def setup_method(self) -> None:
        self.adapter = VLLMAdapter()
        self.ctx = self.adapter.bind(_FakeEngineCore())

    def test_model_name_from_vllm_config(self) -> None:
        result = self.adapter.model_name(self.ctx)
        assert result.is_available
        assert result.value == "openai/whisper-large-v3"

    def test_architecture_from_hf_config(self) -> None:
        result = self.adapter.architecture(self.ctx)
        assert result.is_available
        assert "WhisperForConditionalGeneration" in result.value

    def test_context_length_from_model_config(self) -> None:
        result = self.adapter.context_length(self.ctx)
        assert result.is_available
        assert result.value == 448

    def test_tensor_parallel_from_parallel_config(self) -> None:
        result = self.adapter.tensor_parallel(self.ctx)
        assert result.is_available
        assert result.value == 1

    def test_parameter_count_from_hf_config(self) -> None:
        result = self.adapter.parameter_count(self.ctx)
        assert result.is_available
        assert result.value == 1_550_000_000

    def test_legacy_engine_model_config(self) -> None:
        engine = MagicMock()
        engine.model_config = _FakeModelConfig()
        engine.vllm_config = None
        ctx = AdapterContext(model=None, engine=engine, runtime_kind="vllm")
        result = self.adapter.precision(ctx)
        assert result.is_available
        assert result.value == "torch.float16"
