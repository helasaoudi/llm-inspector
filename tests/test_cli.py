"""Tests for the CLI commands — cli.py delegates only, so we mock collaborators."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from llm_inspector.cli import app
from llm_inspector.models.enums import BackendKind, RuntimeKind
from llm_inspector.models.results import HardwareResult, ProcessResult

runner = CliRunner()


def _make_report(pid: int = 123):
    from llm_inspector.models.report import InspectionReport

    process = ProcessResult(pid=pid, cmdline=["python"], runtime_kind=RuntimeKind.UNKNOWN)
    hardware = HardwareResult(backend=BackendKind.CPU)
    return InspectionReport(pid=pid, process=process, hardware=hardware)


class TestCmdPs:
    def test_no_processes_prints_hint(self) -> None:
        with patch("llm_inspector.inspector.core.Inspector") as mock_inspector_cls:
            mock_inspector_cls.return_value.scan.return_value = []
            result = runner.invoke(app, ["ps"])

        assert result.exit_code == 0
        assert "No LLM processes found" in result.stdout

    def test_processes_render_table(self) -> None:
        report = _make_report()
        with patch("llm_inspector.inspector.core.Inspector") as mock_inspector_cls:
            mock_inspector_cls.return_value.scan.return_value = [report]
            result = runner.invoke(app, ["ps"])

        assert result.exit_code == 0
        assert "123" in result.stdout


class TestCmdInspect:
    def test_success_renders_report(self) -> None:
        report = _make_report(pid=456)
        with patch("llm_inspector.inspector.core.Inspector") as mock_inspector_cls:
            mock_inspector_cls.return_value.inspect.return_value = report
            result = runner.invoke(app, ["inspect", "456"])

        assert result.exit_code == 0
        assert "456" in result.stdout
        assert "Process" in result.stdout

    def test_runtime_error_exits_nonzero(self) -> None:
        with patch("llm_inspector.inspector.core.Inspector") as mock_inspector_cls:
            mock_inspector_cls.return_value.inspect.side_effect = RuntimeError("No such process")
            result = runner.invoke(app, ["inspect", "999999"])

        assert result.exit_code == 1
        assert "Error" in result.stdout
        assert "No such process" in result.stdout

    def test_passes_collect_and_verbose_flags(self) -> None:
        report = _make_report(pid=1)
        with patch("llm_inspector.inspector.core.Inspector") as mock_inspector_cls:
            instance = mock_inspector_cls.return_value
            instance.inspect.return_value = report
            runner.invoke(app, ["inspect", "1", "--collect", "hardware", "--verbose"])

        instance.inspect.assert_called_once_with(pid=1, collect_filter="hardware")


class TestCmdGpu:
    def test_no_devices_prints_hint(self) -> None:
        with patch("llm_inspector.backends.registry.BackendRegistry") as mock_registry_cls:
            backend = mock_registry_cls.return_value.best_available.return_value
            backend.list_devices.return_value = []
            result = runner.invoke(app, ["gpu"])

        assert result.exit_code == 0
        assert "No GPU devices detected" in result.stdout

    def test_devices_render_table(self) -> None:
        from llm_inspector.backends.base import DeviceInfo

        device = DeviceInfo(
            index=0,
            name="NVIDIA RTX 4090",
            vram_total_bytes=24 * 1024**3,
            vram_used_bytes=12 * 1024**3,
            gpu_utilization_pct=50,
            driver_version="575.64",
            cuda_version="12.4",
        )
        with patch("llm_inspector.backends.registry.BackendRegistry") as mock_registry_cls:
            mock_registry_cls.return_value.best_available.return_value.list_devices.return_value = [
                device
            ]
            result = runner.invoke(app, ["gpu"])

        assert result.exit_code == 0
        assert "NVIDIA RTX 4090" in result.stdout
        assert "12.0 GB" in result.stdout
        assert "50%" in result.stdout


class TestCmdRuntimes:
    def test_lists_registered_plugins(self) -> None:
        plugin = MagicMock()
        plugin.display_name = "vLLM"
        plugin.description = "Runs vLLM inference"
        with patch("llm_inspector.inspector.core.Inspector") as mock_inspector_cls:
            mock_inspector_cls.return_value.registered_plugins.return_value = [plugin]
            result = runner.invoke(app, ["runtimes"])

        assert result.exit_code == 0
        assert "vLLM" in result.stdout
        assert "Runs vLLM inference" in result.stdout
