"""Phase 5 — tokenizer / architecture / module breakdown extractors."""

from __future__ import annotations

from types import SimpleNamespace

from llm_inspector.embedded.adapters.vllm import VLLMAdapter
from llm_inspector.embedded.model_details import (
    collect_model_detail_fields,
    extract_architecture_fields,
    extract_module_param_breakdown,
    extract_tokenizer_fields,
)


class _FakeTokenizer:
    vocab_size = 128_256
    chat_template = "{% for m in messages %}{{ m }}{% endfor %}"
    bos_token = "<s>"
    eos_token = "</s>"


class _FakeHFConfig:
    architectures = ["LlamaForCausalLM"]
    num_hidden_layers = 32
    hidden_size = 4096
    num_attention_heads = 32
    num_key_value_heads = 8


class _FakeParam:
    def __init__(self, n: int) -> None:
        self._n = n

    def numel(self) -> int:
        return self._n


class _FakeModel:
    def named_parameters(self):
        return [
            ("model.embed_tokens.weight", _FakeParam(1000)),
            ("model.layers.0.self_attn.q_proj.weight", _FakeParam(5000)),
            ("lm_head.weight", _FakeParam(1000)),
        ]


class TestTokenizerFields:
    def test_extracts_vocab_and_template(self) -> None:
        fields = extract_tokenizer_fields(_FakeTokenizer())
        assert fields["tokenizer_class"].value == "_FakeTokenizer"
        assert fields["vocab_size"].value == 128_256
        assert fields["chat_template"].value == "yes"
        assert fields["bos_token"].value == "<s>"
        assert fields["eos_token"].value == "</s>"

    def test_missing_tokenizer(self) -> None:
        fields = extract_tokenizer_fields(None)
        assert not fields["tokenizer_class"].is_available


class TestArchitectureFields:
    def test_from_hf_config(self) -> None:
        fields = extract_architecture_fields(_FakeHFConfig())
        assert fields["num_layers"].value == 32
        assert fields["hidden_size"].value == 4096
        assert fields["num_attention_heads"].value == 32
        assert fields["num_kv_heads"].value == 8
        assert not fields["num_experts"].is_available


class TestModuleBreakdown:
    def test_buckets_by_name(self) -> None:
        fields = extract_module_param_breakdown(_FakeModel())
        assert fields["embed_params"].value == 1000
        assert fields["transformer_params"].value == 5000
        assert fields["head_params"].value == 1000


class _FakeEngineCore:
    """Class name must look like vLLM EngineCore for adapter detection."""

    __module__ = "vllm.v1.engine.core"

    def __init__(self) -> None:
        self.tokenizer = _FakeTokenizer()
        self.vllm_config = SimpleNamespace(
            model_config=SimpleNamespace(hf_config=_FakeHFConfig())
        )
        self.model_executor = SimpleNamespace(
            driver_worker=SimpleNamespace(
                model_runner=SimpleNamespace(model=_FakeModel())
            )
        )


class TestVLLMAdapterDetails:
    def test_model_details_from_engine(self) -> None:
        adapter = VLLMAdapter()
        ctx = adapter.bind(_FakeEngineCore())
        details = adapter.model_details(ctx)
        assert details["vocab_size"].value == 128_256
        assert details["num_layers"].value == 32
        assert details["embed_params"].value == 1000


class TestCollectAggregate:
    def test_aggregate(self) -> None:
        model = _FakeModel()
        model.config = _FakeHFConfig()  # type: ignore[attr-defined]
        fields = collect_model_detail_fields(
            engine=SimpleNamespace(tokenizer=_FakeTokenizer()),
            model=model,
        )
        assert fields["chat_template"].value == "yes"
        assert fields["hidden_size"].value == 4096
        assert fields["transformer_params"].value == 5000
