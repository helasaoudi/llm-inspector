"""Tests for pure formatting helpers in ui/format.py."""

from __future__ import annotations

from llm_inspector.models.measurement import Measurement
from llm_inspector.ui.format import (
    fmt_bytes,
    fmt_measurement_bytes,
    fmt_measurement_int,
    fmt_measurement_str,
    fmt_params,
    fmt_projected_bytes,
    fmt_uptime,
    truncate,
)


class TestFmtBytes:
    def test_none_is_unavailable(self) -> None:
        assert fmt_bytes(None) == "Unavailable"

    def test_gb_range(self) -> None:
        assert fmt_bytes(2 * 1024**3) == "2.0 GB"

    def test_mb_range(self) -> None:
        assert fmt_bytes(5 * 1024**2) == "5.0 MB"

    def test_kb_range(self) -> None:
        assert fmt_bytes(3 * 1024) == "3.0 KB"

    def test_bytes_range(self) -> None:
        assert fmt_bytes(42) == "42 B"

    def test_zero_bytes(self) -> None:
        assert fmt_bytes(0) == "0 B"


class TestFmtMeasurementBytes:
    def test_available_formats_value(self) -> None:
        m = Measurement.available(1024**3, "NVML")
        assert fmt_measurement_bytes(m) == "1.0 GB"

    def test_unavailable_returns_placeholder(self) -> None:
        m = Measurement.unavailable("no source")
        assert fmt_measurement_bytes(m) == "Unavailable"


class TestFmtProjectedBytes:
    def test_simulated_value_has_value(self) -> None:
        m = Measurement.simulated(512, "FP8 projection")
        assert fmt_projected_bytes(m) == "512 B"

    def test_unavailable_returns_placeholder(self) -> None:
        m = Measurement.unavailable("not projected")
        assert fmt_projected_bytes(m) == "Unavailable"


class TestFmtMeasurementStr:
    def test_available(self) -> None:
        m = Measurement.available("llama3", "cmdline")
        assert fmt_measurement_str(m) == "llama3"

    def test_unavailable(self) -> None:
        m = Measurement.unavailable("not exposed")
        assert fmt_measurement_str(m) == "Unavailable"


class TestFmtMeasurementInt:
    def test_available_with_suffix(self) -> None:
        m = Measurement.available(8192, "config.json")
        assert fmt_measurement_int(m, suffix=" tokens") == "8,192 tokens"

    def test_unavailable(self) -> None:
        m = Measurement.unavailable("not exposed")
        assert fmt_measurement_int(m) == "Unavailable"


class TestFmtParams:
    def test_none(self) -> None:
        assert fmt_params(None) == "Unavailable"

    def test_trillion(self) -> None:
        assert fmt_params(2_000_000_000_000) == "2 Trillion"

    def test_billion(self) -> None:
        assert fmt_params(7_000_000_000) == "7 Billion"

    def test_million(self) -> None:
        assert fmt_params(350_000_000) == "350 Million"

    def test_small_count(self) -> None:
        assert fmt_params(42) == "42"


class TestFmtUptime:
    def test_none(self) -> None:
        assert fmt_uptime(None) == "unknown"

    def test_hours_and_minutes(self) -> None:
        assert fmt_uptime(3600 + 120) == "1h 2m"

    def test_minutes_only(self) -> None:
        assert fmt_uptime(150) == "2m"

    def test_seconds_only(self) -> None:
        assert fmt_uptime(45) == "45s"


class TestTruncate:
    def test_short_string_unchanged(self) -> None:
        assert truncate("hello", max_len=10) == "hello"

    def test_long_string_truncated_with_ellipsis(self) -> None:
        result = truncate("x" * 100, max_len=10)
        assert result == "x" * 9 + "…"
        assert len(result) == 10

    def test_exact_length_unchanged(self) -> None:
        assert truncate("x" * 10, max_len=10) == "x" * 10
