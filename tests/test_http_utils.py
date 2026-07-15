"""Tests for utils/http.py — safe HTTP client and Prometheus parser."""

import pytest

from llm_inspector.utils.http import find_metric, parse_prometheus


class TestParsePrometheus:
    def test_simple_metric(self) -> None:
        text = "vllm:gpu_cache_usage_perc 0.423\n"
        result = parse_prometheus(text)
        assert result["vllm:gpu_cache_usage_perc"] == pytest.approx(0.423)

    def test_metric_with_labels(self) -> None:
        text = 'vllm:gpu_cache_usage_perc{model_name="Llama-3-8B"} 0.55\n'
        result = parse_prometheus(text)
        assert 'vllm:gpu_cache_usage_perc{model_name="Llama-3-8B"}' in result

    def test_comment_lines_skipped(self) -> None:
        text = "# HELP vllm:gpu_cache_usage_perc\n# TYPE gauge\nvllm:foo 1.0\n"
        result = parse_prometheus(text)
        assert "vllm:foo" in result
        assert all(not k.startswith("#") for k in result)

    def test_empty_lines_skipped(self) -> None:
        text = "\n\nvllm:bar 2.0\n\n"
        result = parse_prometheus(text)
        assert result["vllm:bar"] == pytest.approx(2.0)

    def test_invalid_value_skipped(self) -> None:
        text = "vllm:bad not_a_float\nvllm:good 1.0\n"
        result = parse_prometheus(text)
        assert "vllm:bad" not in result
        assert result["vllm:good"] == pytest.approx(1.0)

    def test_empty_text(self) -> None:
        assert parse_prometheus("") == {}

    def test_multiple_metrics(self) -> None:
        text = (
            "vllm:num_requests_running 5.0\n"
            "vllm:num_requests_waiting 2.0\n"
            'vllm:gpu_cache_usage_perc{model="foo"} 0.7\n'
        )
        result = parse_prometheus(text)
        assert len(result) == 3
        assert result["vllm:num_requests_running"] == pytest.approx(5.0)


class TestFindMetric:
    def test_exact_prefix_match(self) -> None:
        metrics = {"vllm:gpu_cache_usage_perc": 0.5}
        assert find_metric(metrics, "vllm:gpu_cache_usage_perc") == pytest.approx(0.5)

    def test_prefix_matches_with_labels(self) -> None:
        metrics = {'vllm:gpu_cache_usage_perc{model="foo"}': 0.42}
        assert find_metric(metrics, "vllm:gpu_cache_usage_perc") == pytest.approx(0.42)

    def test_no_match_returns_none(self) -> None:
        metrics = {"vllm:other_metric": 1.0}
        assert find_metric(metrics, "vllm:gpu_cache_usage_perc") is None

    def test_label_filter_match(self) -> None:
        metrics = {
            'vllm:metric{model="foo"}': 0.1,
            'vllm:metric{model="bar"}': 0.9,
        }
        result = find_metric(metrics, "vllm:metric", label_filter={"model": "bar"})
        assert result == pytest.approx(0.9)

    def test_label_filter_no_match(self) -> None:
        metrics = {'vllm:metric{model="foo"}': 0.1}
        result = find_metric(metrics, "vllm:metric", label_filter={"model": "baz"})
        assert result is None

    def test_empty_metrics(self) -> None:
        assert find_metric({}, "vllm:anything") is None
