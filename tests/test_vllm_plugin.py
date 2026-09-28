"""Tests for VLLMPlugin — mocking REST API responses."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_inspector.models.enums import RuntimeKind
from llm_inspector.models.results import ProcessResult
from llm_inspector.plugins.vllm import VLLMPlugin


def _make_process(cmdline: list[str]) -> ProcessResult:
    return ProcessResult(pid=1234, cmdline=cmdline, runtime_kind=RuntimeKind.VLLM)


def _make_context(cmdline: list[str]) -> MagicMock:
    ctx = MagicMock()
    ctx.process = _make_process(cmdline)
    return ctx


class TestVLLMPluginSupports:
    def test_matches_vllm_in_cmdline(self) -> None:
        p = _make_process(["python", "-m", "vllm.entrypoints.openai.api_server"])
        assert VLLMPlugin().supports(p) is True

    def test_no_match_for_ollama(self) -> None:
        p = _make_process(["ollama", "serve"])
        assert VLLMPlugin().supports(p) is False


class TestVLLMGetModelInfo:
    def _cmdline(self) -> list[str]:
        return [
            "python",
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            "meta-llama/Llama-3-8B-Instruct",
            "--dtype",
            "bfloat16",
            "--tensor-parallel-size",
            "2",
            "--port",
            "8000",
        ]

    @patch("llm_inspector.plugins.vllm.get_json")
    def test_model_name_prefers_cmdline_over_api(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"data": [{"id": "wrong-api-model", "max_model_len": 8192}]}
        ctx = _make_context(self._cmdline())
        result = VLLMPlugin().get_model_info(ctx)
        assert result is not None
        assert result.name.is_available
        assert result.name.value == "meta-llama/Llama-3-8B-Instruct"
        assert result.name.source == "cmdline --model"

    @patch("llm_inspector.plugins.vllm.get_json")
    def test_model_name_from_api_when_no_cmdline(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {
            "data": [{"id": "meta-llama/Llama-3-8B-Instruct", "max_model_len": 8192}]
        }
        ctx = _make_context(
            ["python", "-m", "vllm.entrypoints.openai.api_server", "--port", "8000"]
        )
        result = VLLMPlugin().get_model_info(ctx)
        assert result is not None
        assert result.name.is_available
        assert result.name.value == "meta-llama/Llama-3-8B-Instruct"
        assert "v1/models" in (result.name.source or "")

    @patch("llm_inspector.plugins.vllm.get_json")
    def test_model_name_falls_back_to_cmdline(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None  # API unreachable
        ctx = _make_context(self._cmdline())
        result = VLLMPlugin().get_model_info(ctx)
        assert result is not None
        assert result.name.is_available
        assert result.name.value == "meta-llama/Llama-3-8B-Instruct"

    @patch("llm_inspector.plugins.vllm.get_json")
    def test_context_length_from_api(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"data": [{"id": "llama", "max_model_len": 8192}]}
        ctx = _make_context(self._cmdline())
        result = VLLMPlugin().get_model_info(ctx)
        assert result is not None
        assert result.context_length.is_available
        assert result.context_length.value == 8192

    @patch("llm_inspector.plugins.vllm.get_json")
    def test_precision_from_cmdline(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        ctx = _make_context(self._cmdline())
        result = VLLMPlugin().get_model_info(ctx)
        assert result is not None
        assert result.precision.is_available
        assert result.precision.value == "BF16"

    @patch("llm_inspector.plugins.vllm.get_json")
    def test_tensor_parallel_from_cmdline(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        ctx = _make_context(self._cmdline())
        result = VLLMPlugin().get_model_info(ctx)
        assert result is not None
        assert result.tensor_parallel.is_available
        assert result.tensor_parallel.value == 2

    @patch("llm_inspector.plugins.vllm.get_json")
    def test_no_model_flag_returns_unavailable(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        ctx = _make_context(["python", "-m", "vllm.entrypoints.openai.api_server"])
        result = VLLMPlugin().get_model_info(ctx)
        assert result is not None
        assert not result.name.is_available


class TestVLLMGetMemoryBreakdown:
    @patch("llm_inspector.plugins.vllm.get_text")
    def test_api_unreachable_returns_unavailable_components(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        ctx = _make_context(
            ["python", "-m", "vllm.entrypoints.openai.api_server", "--port", "8000"]
        )
        result = VLLMPlugin().get_memory_breakdown(ctx)
        assert result is not None
        assert all(not c.measurement.is_available for c in result.components)

    @patch("llm_inspector.plugins.vllm.get_text")
    def test_kv_bytes_metric_populates_kv_cache(self, mock_get: MagicMock) -> None:
        mock_get.return_value = (
            "vllm:gpu_cache_memory_bytes 2147483648.0\n"
            "vllm:model_weights_memory_bytes 13958643712.0\n"
        )
        ctx = _make_context(
            ["python", "-m", "vllm.entrypoints.openai.api_server", "--port", "8000"]
        )
        result = VLLMPlugin().get_memory_breakdown(ctx)
        assert result is not None
        kv = result.get("KV Cache")
        assert kv is not None
        assert kv.measurement.is_available
        assert kv.measurement.value == 2147483648

        weights = result.get("Weights")
        assert weights is not None
        assert weights.measurement.is_available
        assert weights.measurement.value == 13958643712

    @patch("llm_inspector.plugins.vllm.get_text")
    def test_total_sums_available_components(self, mock_get: MagicMock) -> None:
        mock_get.return_value = (
            "vllm:gpu_cache_memory_bytes 1000000.0\nvllm:model_weights_memory_bytes 2000000.0\n"
        )
        ctx = _make_context(["python", "-m", "vllm.entrypoints.openai.api_server"])
        result = VLLMPlugin().get_memory_breakdown(ctx)
        assert result is not None
        assert result.total.is_available
        assert result.total.value == 3000000


class TestVLLMGetRuntimeDetails:
    def test_pageattn_always_enabled(self) -> None:
        ctx = _make_context(["python", "-m", "vllm.entrypoints.openai.api_server"])
        details = VLLMPlugin().get_runtime_details(ctx)
        assert "PagedAttention" in details
        assert details["PagedAttention"].is_available
        assert details["PagedAttention"].value == "Enabled"

    def test_tensor_parallel_in_details(self) -> None:
        ctx = _make_context(["vllm", "--tensor-parallel-size", "4"])
        details = VLLMPlugin().get_runtime_details(ctx)
        assert details["Tensor Parallel"].value == "4"

    def test_default_scheduler(self) -> None:
        ctx = _make_context(["vllm"])
        details = VLLMPlugin().get_runtime_details(ctx)
        assert details["Scheduler"].value == "FCFS"


class TestVLLMNormaliseDtype:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("float16", "FP16"),
            ("bfloat16", "BF16"),
            ("float32", "FP32"),
            ("half", "FP16"),
            ("auto", "auto"),
            ("awq", "AWQ"),
            ("gptq", "GPTQ"),
            ("fp8", "FP8"),
            ("BFLOAT16", "BF16"),  # case-insensitive
        ],
    )
    def test_dtype_normalisation(self, raw: str, expected: str) -> None:
        assert VLLMPlugin._normalise_dtype(raw) == expected
