"""Tests for vLLM API URL resolution."""

from __future__ import annotations

from unittest.mock import patch

from llm_inspector.utils.vllm_urls import is_engine_core_process, resolve_vllm_base_url


class TestResolveVllmBaseUrl:
    def test_cmdline_port_wins(self) -> None:
        base, src = resolve_vllm_base_url(
            ["python", "-m", "vllm", "--port", "8011"],
            {},
        )
        assert base == "http://127.0.0.1:8011"
        assert "cmdline" in src

    def test_sibling_port_before_probe(self) -> None:
        # EngineCore pid 20, parent API server 10 has --port 8011;
        # a wrong job also answers on :8000 — sibling must win.
        def cmdline(pid: int) -> list[str]:
            return {
                10: ["python", "-m", "vllm", "--port", "8011", "--model", "AceMath"],
                20: ["VLLM::EngineCore"],
            }.get(pid, [])

        def ppid(pid: int) -> int | None:
            return {20: 10, 10: 1}.get(pid)

        with (
            patch(
                "llm_inspector.utils.proc.read_cmdline",
                side_effect=cmdline,
            ),
            patch(
                "llm_inspector.utils.proc.read_ppid",
                side_effect=ppid,
            ),
            patch(
                "llm_inspector.utils.proc.list_all_pids",
                return_value=[10, 20],
            ),
            patch(
                "llm_inspector.utils.vllm_urls._api_reachable",
                side_effect=lambda url: url.endswith(":8000"),
            ),
        ):
            base, src = resolve_vllm_base_url(
                ["VLLM::EngineCore"],
                {},
                pid=20,
            )
        assert base == "http://127.0.0.1:8011"
        assert "parent" in src
        assert "8011" in src

    def test_probe_includes_8011(self) -> None:
        with (
            patch(
                "llm_inspector.utils.proc.read_ppid",
                return_value=None,
            ),
            patch(
                "llm_inspector.utils.proc.list_all_pids",
                return_value=[],
            ),
            patch(
                "llm_inspector.utils.vllm_urls._api_reachable",
                side_effect=lambda url: url.endswith(":8011"),
            ),
        ):
            base, src = resolve_vllm_base_url(
                ["VLLM::EngineCore"],
                {},
                pid=99,
            )
        assert base == "http://127.0.0.1:8011"
        assert "port probe" in src

    def test_is_engine_core(self) -> None:
        assert is_engine_core_process(["VLLM::EngineCore"]) is True
        assert is_engine_core_process(["python", "app.py"]) is False
