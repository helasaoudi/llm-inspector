"""Tests for ui/tables.py — the llminspect ps table renderer."""

from __future__ import annotations

import io

from rich.console import Console

from llm_inspector.models.enums import BackendKind, RuntimeKind
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.report import InspectionReport
from llm_inspector.models.results import HardwareResult, MemoryResult, ModelResult, ProcessResult
from llm_inspector.ui.tables import render_ps_table


def _console() -> Console:
    return Console(file=io.StringIO(), width=200, no_color=True)


def _report(
    pid: int,
    backend: BackendKind,
    *,
    model_name: str | None = None,
    process_ram: int | None = None,
) -> InspectionReport:
    process = ProcessResult(
        pid=pid, cmdline=["python"], runtime_kind=RuntimeKind.VLLM, uptime_seconds=125
    )
    hardware = HardwareResult(
        backend=backend,
        gpu_utilization_pct=77 if backend != BackendKind.CPU else None,
        vram_used_bytes=4 * 1024**3 if backend != BackendKind.CPU else None,
    )
    model = None
    if model_name is not None:
        model = ModelResult(name=Measurement.available(model_name, "cmdline"))
    memory = None
    if process_ram is not None:
        memory = MemoryResult(process_ram=Measurement.available(process_ram, "psutil"))
    return InspectionReport(
        pid=pid, process=process, hardware=hardware, model=model, memory=memory
    )


class TestRenderPsTable:
    def test_empty_reports_prints_hint(self) -> None:
        console = _console()
        render_ps_table([], console)
        output = console.file.getvalue()
        assert "No LLM processes found" in output

    def test_gpu_report_shows_vram_and_util_columns(self) -> None:
        console = _console()
        report = _report(123, BackendKind.CUDA, model_name="llama3-8b")
        render_ps_table([report], console)
        output = console.file.getvalue()

        assert "123" in output
        assert "VRAM" in output
        assert "GPU%" in output
        assert "llama3-8b" in output
        assert "4.0 GB" in output
        assert "77%" in output

    def test_cpu_only_report_shows_ram_column(self) -> None:
        console = _console()
        report = _report(456, BackendKind.CPU, process_ram=512 * 1024**2)
        render_ps_table([report], console)
        output = console.file.getvalue()

        assert "456" in output
        assert "RAM" in output
        assert "VRAM" not in output
        assert "512.0 MB" in output

    def test_model_name_is_truncated(self) -> None:
        console = _console()
        long_name = "a" * 40
        report = _report(1, BackendKind.CUDA, model_name=long_name)
        render_ps_table([report], console)
        output = console.file.getvalue()

        assert long_name not in output
        assert "…" in output

    def test_missing_model_shows_placeholder(self) -> None:
        console = _console()
        report = _report(1, BackendKind.CUDA)
        render_ps_table([report], console)
        output = console.file.getvalue()

        assert "—" in output
