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


class _FakeCacheConfig:
    kv_cache_memory_bytes = None
    num_gpu_blocks = 128


class _FakeVllmConfig:
    model_config = _FakeModelConfig()
    parallel_config = _FakeParallelConfig()
    cache_config = _FakeCacheConfig()


class _FakeParam:
    def __init__(self, numel: int = 1_000_000, element_size: int = 2) -> None:
        self._numel = numel
        self._element_size = element_size

    def numel(self) -> int:
        return self._numel

    def element_size(self) -> int:
        return self._element_size


class _FakeModule:
    def parameters(self):
        return iter([_FakeParam(1_550_000_000, 2)])


class _FakeKVTensor:
    def __init__(self, size: int) -> None:
        self.size = size


class _FakeKVCacheConfig:
    def __init__(self) -> None:
        self.num_blocks = 128
        self.kv_cache_tensors = [_FakeKVTensor(4_000_000_000), _FakeKVTensor(1_000_000_000)]


class _FakeScheduler:
    kv_cache_config = _FakeKVCacheConfig()


class _FakeModelRunner:
    def __init__(self) -> None:
        self.model = _FakeModule()

    def get_model(self):
        return self.model


class _FakeWorker:
    def __init__(self) -> None:
        self.model_runner = _FakeModelRunner()

    def get_model(self):
        return self.model_runner.get_model()


class _FakeExecutor:
    def __init__(self) -> None:
        self.driver_worker = _FakeWorker()


class _FakeEngineCore:
    __module__ = "vllm.v1.engine.core"

    def __init__(self) -> None:
        self.vllm_config = _FakeVllmConfig()
        self.model_executor = _FakeExecutor()
        self.scheduler = _FakeScheduler()
        self.available_gpu_memory_for_kv_cache = 5_000_000_000


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

    def test_parameter_count_from_model_runner(self) -> None:
        result = self.adapter.parameter_count(self.ctx)
        assert result.is_available
        assert result.value == 1_550_000_000
        assert "model.parameters()" in (result.source or "")

    def test_weights_bytes_from_model_runner(self) -> None:
        result = self.adapter.weights_bytes(self.ctx)
        assert result.is_available
        assert result.value == 1_550_000_000 * 2

    def test_kv_cache_bytes_from_tensors(self) -> None:
        result = self.adapter.kv_cache_bytes(self.ctx)
        assert result.is_available
        assert result.value == 5_000_000_000
        assert "kv_cache_tensors" in (result.source or "")

    def test_kv_cache_bytes_falls_back_to_available_memory(self) -> None:
        engine = _FakeEngineCore()
        engine.scheduler.kv_cache_config.kv_cache_tensors = []
        ctx = self.adapter.bind(engine)
        result = self.adapter.kv_cache_bytes(ctx)
        assert result.is_available
        assert result.value == 5_000_000_000
        assert "available_gpu_memory_for_kv_cache" in (result.source or "")

    def test_legacy_engine_model_config(self) -> None:
        engine = MagicMock()
        engine.model_config = _FakeModelConfig()
        engine.vllm_config = None
        engine.model_executor = None
        engine.scheduler = None
        ctx = AdapterContext(model=None, engine=engine, runtime_kind="vllm")
        result = self.adapter.precision(ctx)
        assert result.is_available
        assert result.value == "torch.float16"
