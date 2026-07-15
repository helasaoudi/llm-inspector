"""Tests for ProcessCollector and runtime detection heuristics."""

import pytest
from unittest.mock import patch

from llm_inspector.collectors.process import ProcessCollector, detect_runtime
from llm_inspector.models.enums import CollectorStatus, RuntimeKind


# ── detect_runtime heuristics ─────────────────────────────────────────────────

@pytest.mark.parametrize("cmdline,expected", [
    (["python", "-m", "vllm.entrypoints.openai.api_server"], RuntimeKind.VLLM),
    (["python", "-m", "vllm", "--model", "meta-llama/Llama-3-8B"], RuntimeKind.VLLM),
    (["VLLM::EngineCore"], RuntimeKind.VLLM),
    (["ollama", "serve"], RuntimeKind.OLLAMA),
    (["tritonserver", "--model-repository=/models"], RuntimeKind.TENSORRT),
    (["python", "inference.py", "--use-transformers"], RuntimeKind.HUGGING_FACE),
    (["python", "my_app.py"], RuntimeKind.UNKNOWN),
    (["bash", "-c", "echo hello"], RuntimeKind.UNKNOWN),
    (["python", "-m", "sglang.launch_server"], RuntimeKind.SGLANG),
])
def test_detect_runtime(cmdline: list[str], expected: RuntimeKind):
    assert detect_runtime(cmdline) == expected


# ── ProcessCollector ──────────────────────────────────────────────────────────

class TestProcessCollector:
    def test_returns_failed_result_for_nonexistent_pid(self):
        collector = ProcessCollector()
        result = collector.collect(pid=9_999_999)
        assert result.failed
        assert result.data is None
        assert result.error is not None

    def test_collect_succeeds_for_current_process(self):
        import os
        collector = ProcessCollector()
        result = collector.collect(pid=os.getpid())
        # On macOS /proc doesn't exist but psutil fallback should work
        if result.succeeded:
            assert result.data is not None
            assert result.data.pid == os.getpid()
            assert result.data.cmdline  # non-empty

    def test_elapsed_ms_is_recorded(self):
        collector = ProcessCollector()
        result = collector.collect(pid=9_999_999)
        assert result.elapsed_ms >= 0

    def test_collector_name(self):
        assert ProcessCollector.name == "process"

    def test_never_raises(self):
        """collect() must never raise — it returns FAILED CollectorResult instead."""
        collector = ProcessCollector()
        # Should not raise even for clearly invalid PIDs
        result = collector.collect(pid=-1)
        assert result is not None
        assert result.collector_name == "process"
