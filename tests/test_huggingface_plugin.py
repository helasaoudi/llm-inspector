"""Tests for HuggingFacePlugin — cmdline-only inspection."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from llm_inspector.models.enums import RuntimeKind
from llm_inspector.models.results import ProcessResult
from llm_inspector.plugins.huggingface import HuggingFacePlugin


def _make_process(cmdline: list[str]) -> ProcessResult:
    return ProcessResult(pid=1111, cmdline=cmdline, runtime_kind=RuntimeKind.HUGGING_FACE)  # noqa: E501


def _make_context(cmdline: list[str]) -> MagicMock:
    ctx = MagicMock()
    ctx.process = _make_process(cmdline)
    return ctx


class TestHuggingFacePluginSupports:
    def test_matches_transformers(self) -> None:
        p = _make_process(["python", "run.py", "--model_name_or_path", "llama"])
        # support is detected via cmdline content, not just --model_name_or_path
        # We need 'transformers' or related keywords in cmdline
        p2 = _make_process(["python", "transformers/trainer.py"])
        assert HuggingFacePlugin().supports(p2) is True

    def test_no_match_for_vllm(self) -> None:
        p = _make_process(["python", "-m", "vllm.entrypoints.api_server"])
        assert HuggingFacePlugin().supports(p) is False


class TestHuggingFaceGetModelInfo:
    def test_model_name_from_model_name_or_path(self) -> None:
        ctx = _make_context([
            "python", "train.py",
            "--model_name_or_path", "/models/llama-3-8b",
        ])
        result = HuggingFacePlugin().get_model_info(ctx)
        assert result is not None
        assert result.name.is_available
        assert result.name.value == "/models/llama-3-8b"

    def test_model_name_from_model_flag(self) -> None:
        ctx = _make_context(["python", "run.py", "--model", "gpt2"])
        result = HuggingFacePlugin().get_model_info(ctx)
        assert result is not None
        assert result.name.is_available
        assert result.name.value == "gpt2"

    def test_model_name_missing_returns_unavailable(self) -> None:
        ctx = _make_context(["python", "run.py"])
        result = HuggingFacePlugin().get_model_info(ctx)
        assert result is not None
        assert not result.name.is_available

    def test_fp16_flag(self) -> None:
        ctx = _make_context(["python", "run.py", "--fp16"])
        result = HuggingFacePlugin().get_model_info(ctx)
        assert result is not None
        assert result.precision.is_available
        assert result.precision.value == "FP16"

    def test_bf16_flag(self) -> None:
        ctx = _make_context(["python", "run.py", "--bf16"])
        result = HuggingFacePlugin().get_model_info(ctx)
        assert result is not None
        assert result.precision.is_available
        assert result.precision.value == "BF16"

    def test_torch_dtype_float16(self) -> None:
        ctx = _make_context(["python", "run.py", "--torch_dtype", "float16"])
        result = HuggingFacePlugin().get_model_info(ctx)
        assert result is not None
        assert result.precision.is_available
        assert result.precision.value == "FP16"

    def test_load_in_8bit(self) -> None:
        ctx = _make_context(["python", "run.py", "--load_in_8bit"])
        result = HuggingFacePlugin().get_model_info(ctx)
        assert result is not None
        assert result.precision.is_available
        assert "INT8" in (result.precision.value or "")

    def test_unknown_precision_returns_unavailable(self) -> None:
        ctx = _make_context(["python", "run.py"])
        result = HuggingFacePlugin().get_model_info(ctx)
        assert result is not None
        assert not result.precision.is_available

    def test_architecture_always_unavailable(self) -> None:
        ctx = _make_context(["python", "run.py", "--model", "gpt2"])
        result = HuggingFacePlugin().get_model_info(ctx)
        assert result is not None
        assert not result.architecture.is_available

    def test_tensor_parallel_from_num_processes(self) -> None:
        ctx = _make_context(["python", "run.py", "--num_processes", "4"])
        result = HuggingFacePlugin().get_model_info(ctx)
        assert result is not None
        assert result.tensor_parallel.is_available
        assert result.tensor_parallel.value == 4


class TestHuggingFaceGetMemoryBreakdown:
    def test_all_components_unavailable(self) -> None:
        ctx = _make_context(["python", "run.py"])
        result = HuggingFacePlugin().get_memory_breakdown(ctx)
        assert result is not None
        assert all(not c.measurement.is_available for c in result.components)
        assert not result.total.is_available


class TestHuggingFaceGetRuntimeDetails:
    def test_trust_remote_code_detected(self) -> None:
        ctx = _make_context(["python", "run.py", "--trust_remote_code"])
        details = HuggingFacePlugin().get_runtime_details(ctx)
        assert "Trust Remote Code" in details
        assert details["Trust Remote Code"].value == "True"

    def test_no_details_for_minimal_cmdline(self) -> None:
        ctx = _make_context(["python", "run.py"])
        details = HuggingFacePlugin().get_runtime_details(ctx)
        assert len(details) == 0
