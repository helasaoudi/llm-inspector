"""Tests for OllamaPlugin — mocking REST API responses."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from llm_inspector.models.enums import RuntimeKind
from llm_inspector.models.results import ProcessResult
from llm_inspector.plugins.ollama import OllamaPlugin


def _make_process(cmdline: list[str]) -> ProcessResult:
    return ProcessResult(pid=9999, cmdline=cmdline, runtime_kind=RuntimeKind.OLLAMA)


def _make_context(cmdline: list[str]) -> MagicMock:
    ctx = MagicMock()
    ctx.process = _make_process(cmdline)
    return ctx


_OLLAMA_PS_RESPONSE = {
    "models": [
        {
            "name": "llama3:8b",
            "model": "llama3:8b",
            "size_vram": 5368709120,
            "context_window": 8192,
            "expires_at": "2025-01-01T00:00:00Z",
            "details": {
                "family": "llama",
                "families": ["llama"],
                "parameter_size": "8B",
                "quantization_level": "Q4_0",
            },
        }
    ]
}


class TestOllamaPluginSupports:
    def test_matches_ollama(self) -> None:
        p = _make_process(["ollama", "serve"])
        assert OllamaPlugin().supports(p) is True

    def test_no_match_for_vllm(self) -> None:
        p = _make_process(["python", "-m", "vllm.entrypoints.openai.api_server"])
        assert OllamaPlugin().supports(p) is False


class TestOllamaGetModelInfo:
    @patch("llm_inspector.plugins.ollama.get_json")
    def test_model_name_from_api(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _OLLAMA_PS_RESPONSE
        ctx = _make_context(["ollama", "serve"])
        result = OllamaPlugin().get_model_info(ctx)
        assert result is not None
        assert result.name.is_available
        assert result.name.value == "llama3:8b"

    @patch("llm_inspector.plugins.ollama.get_json")
    def test_architecture_from_family(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _OLLAMA_PS_RESPONSE
        ctx = _make_context(["ollama", "serve"])
        result = OllamaPlugin().get_model_info(ctx)
        assert result is not None
        assert result.architecture.is_available
        assert result.architecture.value == "llama"

    @patch("llm_inspector.plugins.ollama.get_json")
    def test_precision_from_quantization_level(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _OLLAMA_PS_RESPONSE
        ctx = _make_context(["ollama", "serve"])
        result = OllamaPlugin().get_model_info(ctx)
        assert result is not None
        assert result.precision.is_available
        assert result.precision.value == "Q4_0"

    @patch("llm_inspector.plugins.ollama.get_json")
    def test_context_length_from_api(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _OLLAMA_PS_RESPONSE
        ctx = _make_context(["ollama", "serve"])
        result = OllamaPlugin().get_model_info(ctx)
        assert result is not None
        assert result.context_length.is_available
        assert result.context_length.value == 8192

    @patch("llm_inspector.plugins.ollama.get_json")
    def test_api_unreachable_returns_unavailable(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        ctx = _make_context(["ollama", "serve"])
        result = OllamaPlugin().get_model_info(ctx)
        assert result is not None
        assert not result.name.is_available

    @patch("llm_inspector.plugins.ollama.get_json")
    def test_no_models_loaded_returns_unavailable(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"models": []}
        ctx = _make_context(["ollama", "serve"])
        result = OllamaPlugin().get_model_info(ctx)
        assert result is not None
        assert not result.name.is_available


class TestOllamaGetMemoryBreakdown:
    @patch("llm_inspector.plugins.ollama.get_json")
    def test_size_vram_becomes_weights_component(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _OLLAMA_PS_RESPONSE
        ctx = _make_context(["ollama", "serve"])
        result = OllamaPlugin().get_memory_breakdown(ctx)
        assert result is not None
        weights = result.get("Weights")
        assert weights is not None
        assert weights.measurement.is_available
        assert weights.measurement.value == 5368709120

    @patch("llm_inspector.plugins.ollama.get_json")
    def test_kv_cache_always_unavailable(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _OLLAMA_PS_RESPONSE
        ctx = _make_context(["ollama", "serve"])
        result = OllamaPlugin().get_memory_breakdown(ctx)
        assert result is not None
        kv = result.get("KV Cache")
        assert kv is not None
        assert not kv.measurement.is_available

    @patch("llm_inspector.plugins.ollama.get_json")
    def test_total_equals_size_vram(self, mock_get: MagicMock) -> None:
        mock_get.return_value = _OLLAMA_PS_RESPONSE
        ctx = _make_context(["ollama", "serve"])
        result = OllamaPlugin().get_memory_breakdown(ctx)
        assert result is not None
        assert result.total.is_available
        assert result.total.value == 5368709120


class TestOllamaGetVersion:
    @patch("llm_inspector.plugins.ollama.get_json")
    def test_version_from_api(self, mock_get: MagicMock) -> None:
        mock_get.return_value = {"version": "0.3.14"}
        ctx = _make_context(["ollama", "serve"])
        version = OllamaPlugin().get_version(ctx)
        assert version.is_available
        assert version.value == "0.3.14"

    @patch("llm_inspector.plugins.ollama.get_json")
    def test_api_unreachable_returns_unavailable(self, mock_get: MagicMock) -> None:
        mock_get.return_value = None
        ctx = _make_context(["ollama", "serve"])
        version = OllamaPlugin().get_version(ctx)
        assert not version.is_available
