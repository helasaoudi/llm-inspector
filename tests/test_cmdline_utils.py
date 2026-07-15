"""Tests for utils/cmdline.py — argument parsing helpers."""

import pytest

from llm_inspector.utils.cmdline import (
    detect_host,
    detect_port,
    flag_present,
    parse_arg,
    parse_int_arg,
)


class TestParseArg:
    def test_space_separated(self) -> None:
        assert parse_arg(["--model", "meta-llama/Llama-3-8B"], "--model") == "meta-llama/Llama-3-8B"

    def test_equals_separated(self) -> None:
        assert parse_arg(["--model=meta-llama/Llama-3-8B"], "--model") == "meta-llama/Llama-3-8B"

    def test_missing_flag(self) -> None:
        assert parse_arg(["--dtype", "float16"], "--model") is None

    def test_multiple_flags_first_match(self) -> None:
        cmdline = ["--model_name_or_path", "/models/llama"]
        assert parse_arg(cmdline, "--model", "--model_name_or_path") == "/models/llama"

    def test_flag_at_end_no_value(self) -> None:
        assert parse_arg(["--model"], "--model") is None

    def test_empty_cmdline(self) -> None:
        assert parse_arg([], "--model") is None

    def test_equals_with_path_containing_equals(self) -> None:
        # Only first '=' is split
        assert parse_arg(["--arg=val=ue"], "--arg") == "val=ue"


class TestParseIntArg:
    def test_valid_int(self) -> None:
        assert parse_int_arg(["--tensor-parallel-size", "4"], "--tensor-parallel-size") == 4

    def test_invalid_int_returns_default(self) -> None:
        assert parse_int_arg(["--port", "abc"], "--port", default=8000) == 8000

    def test_missing_returns_default(self) -> None:
        assert parse_int_arg([], "--port", default=8000) == 8000

    def test_missing_no_default_returns_none(self) -> None:
        assert parse_int_arg([], "--port") is None


class TestFlagPresent:
    def test_present(self) -> None:
        assert flag_present(["--fp16", "--trust_remote_code"], "--fp16")

    def test_absent(self) -> None:
        assert not flag_present(["--bf16"], "--fp16")

    def test_multiple_flags_any_present(self) -> None:
        assert flag_present(["--bf16"], "--fp16", "--bf16")

    def test_empty_cmdline(self) -> None:
        assert not flag_present([], "--fp16")


class TestDetectPort:
    def test_explicit_port(self) -> None:
        assert detect_port(["--port", "9000"]) == 9000

    def test_short_flag(self) -> None:
        assert detect_port(["-p", "9001"]) == 9001

    def test_default(self) -> None:
        assert detect_port([]) == 8000

    def test_custom_default(self) -> None:
        assert detect_port([], default=11434) == 11434


class TestDetectHost:
    def test_explicit_host(self) -> None:
        assert detect_host(["--host", "0.0.0.0"]) == "0.0.0.0"

    def test_default(self) -> None:
        assert detect_host([]) == "127.0.0.1"
