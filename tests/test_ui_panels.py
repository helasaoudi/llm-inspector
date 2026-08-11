"""Tests for ui/panels.py — Rich panel renderers for llminspect inspect."""

from __future__ import annotations

import io

from rich.console import Console

from llm_inspector.models.enums import BackendKind, RuntimeKind
from llm_inspector.models.measurement import Measurement
from llm_inspector.models.report import InspectionReport
from llm_inspector.models.results import (
    HardwareResult,
    MemoryBreakdownResult,
    MemoryResult,
    ModelResult,
    ProcessResult,
    RuntimeResult,
)
from llm_inspector.optimization.models import (
    OptimizationAnalysis,
    OptimizationGroup,
    OptimizationKind,
    OptimizationRecommendation,
    OptimizationScenario,
    QualityBand,
)
from llm_inspector.ui import panels


def _console() -> Console:
    return Console(file=io.StringIO(), width=200, no_color=True)


def _base_report(**overrides: object) -> InspectionReport:
    process = ProcessResult(
        pid=1,
        cmdline=["python", "-m", "vllm.entrypoints.openai.api_server"],
        runtime_kind=RuntimeKind.VLLM,
        uptime_seconds=90,
    )
    hardware = HardwareResult(backend=BackendKind.CUDA, gpu_name="NVIDIA A100")
    defaults: dict = {"pid": 1, "process": process, "hardware": hardware}
    defaults.update(overrides)
    return InspectionReport(**defaults)


class TestHeaderFooter:
    def test_render_header_includes_version_and_snapshot_time(self) -> None:
        console = _console()
        panels.render_header(console, captured_at="2026-07-23 10:00:00 UTC")
        output = console.file.getvalue()
        assert "LLM INSPECTOR" in output
        assert "2026-07-23 10:00:00 UTC" in output

    def test_render_footer_includes_repo_url(self) -> None:
        console = _console()
        panels.render_footer(console)
        output = console.file.getvalue()
        assert "Measured. Not guessed." in output
        assert "github.com" in output


class TestProcessSection:
    def test_shows_pid_runtime_and_truncated_command(self) -> None:
        console = _console()
        report = _base_report()
        panels.render_process_section(report, console)
        output = console.file.getvalue()
        assert "1" in output
        assert "vLLM" in output
        assert "CUDA" in output

    def test_verbose_shows_full_command(self) -> None:
        console = _console()
        process = ProcessResult(
            pid=1,
            cmdline=["python"] + ["--flag"] * 60,
            runtime_kind=RuntimeKind.VLLM,
        )
        report = _base_report(process=process)
        panels.render_process_section(report, console, verbose=True)
        output = console.file.getvalue()
        assert "…" not in output


class TestHardwareSection:
    def test_gpu_backend_shows_vram_and_driver(self) -> None:
        console = _console()
        hardware = HardwareResult(
            backend=BackendKind.CUDA,
            gpu_name="NVIDIA A100",
            driver_version="575.64",
            cuda_version="12.4",
            vram_total_bytes=40 * 1024**3,
            vram_used_bytes=10 * 1024**3,
        )
        report = _base_report(hardware=hardware)
        panels.render_hardware_section(report, console)
        output = console.file.getvalue()
        assert "NVIDIA A100" in output
        assert "575.64" in output
        assert "10.0 GB" in output

    def test_cpu_backend_shows_system_info(self) -> None:
        console = _console()
        hardware = HardwareResult(
            backend=BackendKind.CPU,
            cpu_name="Apple M2",
            cpu_physical_cores=8,
            cpu_logical_cores=8,
            system_ram_total_bytes=16 * 1024**3,
        )
        report = _base_report(hardware=hardware)
        panels.render_hardware_section(report, console)
        output = console.file.getvalue()
        assert "Apple M2" in output
        assert "8P / 8L" in output
        assert "16.0 GB" in output


class TestModelSection:
    def test_none_model_shows_unavailable_message(self) -> None:
        console = _console()
        report = _base_report(model=None)
        panels.render_model_section(report, console)
        output = console.file.getvalue()
        assert "unavailable" in output.lower()

    def test_populated_model_shows_name_and_params(self) -> None:
        console = _console()
        model = ModelResult(
            name=Measurement.available("llama3-8b", "cmdline"),
            parameter_count=Measurement.available(8_000_000_000, "config.json"),
            context_length=Measurement.available(8192, "config.json"),
        )
        report = _base_report(model=model)
        panels.render_model_section(report, console)
        output = console.file.getvalue()
        assert "llama3-8b" in output
        assert "8.0B" in output
        assert "8,192 tokens" in output

    def test_verbose_shows_source_annotation(self) -> None:
        console = _console()
        model = ModelResult(name=Measurement.available("llama3-8b", "cmdline"))
        report = _base_report(model=model)
        panels.render_model_section(report, console, verbose=True)
        output = console.file.getvalue()
        assert "cmdline" in output

    def test_ollama_parameter_size_hint_used_when_count_unavailable(self) -> None:
        console = _console()
        model = ModelResult(
            parameter_count=Measurement.unavailable(
                "Ollama reported parameter_size='4.5B' without exact count"
            )
        )
        report = _base_report(model=model)
        panels.render_model_section(report, console)
        output = console.file.getvalue()
        assert "4.5B" in output
        assert "approx, from Ollama" in output

    def test_details_section_only_shown_when_details_available(self) -> None:
        console = _console()
        model = ModelResult(
            tokenizer_class=Measurement.available("LlamaTokenizer", "tokenizer_config")
        )
        report = _base_report(model=model)
        panels.render_model_section(report, console)
        output = console.file.getvalue()
        assert "Model Details" in output
        assert "LlamaTokenizer" in output


class TestMemorySection:
    def test_none_memory_shows_unavailable_message(self) -> None:
        console = _console()
        report = _base_report(memory=None)
        panels.render_memory_section(report, console)
        output = console.file.getvalue()
        assert "unavailable" in output.lower()

    def test_gpu_backend_shows_gpu_fields(self) -> None:
        console = _console()
        memory = MemoryResult(
            gpu_used=Measurement.available(10 * 1024**3, "NVML"),
            process_ram=Measurement.available(1 * 1024**3, "psutil"),
        )
        report = _base_report(memory=memory)
        panels.render_memory_section(report, console)
        output = console.file.getvalue()
        assert "GPU Used" in output
        assert "10.0 GB" in output

    def test_cpu_backend_hides_gpu_fields(self) -> None:
        console = _console()
        hardware = HardwareResult(backend=BackendKind.CPU)
        memory = MemoryResult(process_ram=Measurement.available(2 * 1024**3, "psutil"))
        report = _base_report(hardware=hardware, memory=memory)
        panels.render_memory_section(report, console)
        output = console.file.getvalue()
        assert "GPU Used" not in output
        assert "Process RAM" in output


class TestMemoryBreakdownSection:
    def test_none_breakdown_shows_unavailable_message(self) -> None:
        console = _console()
        report = _base_report(memory_breakdown=None)
        panels.render_memory_breakdown_section(report, console)
        output = console.file.getvalue()
        assert "unavailable" in output.lower()

    def test_components_and_total_rendered(self) -> None:
        from llm_inspector.models.results import ComponentName, MemoryComponent

        console = _console()
        breakdown = MemoryBreakdownResult(
            components=[
                MemoryComponent(
                    name=ComponentName.WEIGHTS,
                    measurement=Measurement.available(16 * 1024**3, "vLLM metrics"),
                    order=0,
                ),
                MemoryComponent(
                    name=ComponentName.KV_CACHE,
                    measurement=Measurement.available(4 * 1024**3, "vLLM metrics"),
                    order=1,
                ),
            ],
            total=Measurement.available(20 * 1024**3, "sum of components"),
        )
        report = _base_report(memory_breakdown=breakdown)
        panels.render_memory_breakdown_section(report, console)
        output = console.file.getvalue()
        assert "Weights" in output
        assert "16.0 GB" in output
        assert "KV Cache" in output
        assert "Total" in output
        assert "20.0 GB" in output


class TestRuntimeSection:
    def test_none_runtime_shows_unavailable_message(self) -> None:
        console = _console()
        report = _base_report(runtime=None)
        panels.render_runtime_section(report, console)
        output = console.file.getvalue()
        assert "unavailable" in output.lower()

    def test_details_rendered(self) -> None:
        console = _console()
        runtime = RuntimeResult(
            runtime_kind=RuntimeKind.VLLM,
            version=Measurement.available("0.6.3", "vLLM /version"),
            details={"PagedAttention": Measurement.available("Enabled", "vLLM config")},
        )
        report = _base_report(runtime=runtime)
        panels.render_runtime_section(report, console)
        output = console.file.getvalue()
        assert "0.6.3" in output
        assert "PagedAttention" in output
        assert "Enabled" in output

    def test_expiry_key_formatted_as_countdown(self) -> None:
        console = _console()
        runtime = RuntimeResult(
            runtime_kind=RuntimeKind.VLLM,
            details={"LicenseExpires": Measurement.available("2099-01-01T00:00:00+00:00", "api")},
        )
        report = _base_report(runtime=runtime)
        panels.render_runtime_section(report, console)
        output = console.file.getvalue()
        assert "expires in" in output


class TestOptimizationSection:
    def test_none_optimization_shows_unavailable_message(self) -> None:
        console = _console()
        report = _base_report(optimization=None)
        panels.render_optimization_section(report, console)
        output = console.file.getvalue()
        assert "unavailable" in output.lower()

    def test_skipped_reason_shown(self) -> None:
        console = _console()
        analysis = OptimizationAnalysis(skipped_reason="Weights size not measured")
        report = _base_report(optimization=analysis)
        panels.render_optimization_section(report, console)
        output = console.file.getvalue()
        assert "Weights size not measured" in output

    def test_groups_and_recommendation_rendered(self) -> None:
        console = _console()
        analysis = OptimizationAnalysis(
            groups=[
                OptimizationGroup(
                    kind=OptimizationKind.QUANTIZATION,
                    title="Quantization",
                    scenarios=[
                        OptimizationScenario(
                            kind=OptimizationKind.QUANTIZATION,
                            method_name="FP8",
                            quality=QualityBand.VERY_GOOD,
                            new_total=Measurement.simulated(8 * 1024**3, "FP8 formula"),
                            saved_bytes=Measurement.simulated(8 * 1024**3, "FP8 formula"),
                        )
                    ],
                )
            ],
            recommendation=OptimizationRecommendation(
                summary="FP8 saves ~50% with minimal quality loss.",
                warnings=["Requires Hopper or newer GPU."],
            ),
        )
        report = _base_report(optimization=analysis)
        panels.render_optimization_section(report, console)
        output = console.file.getvalue()
        assert "Quantization" in output
        assert "FP8" in output
        assert "saves ~50%" in output
        assert "Requires Hopper" in output


class TestRenderFullReport:
    def test_collect_filter_all_renders_every_section(self) -> None:
        console = _console()
        report = _base_report()
        panels.render_full_report(report, console, collect_filter="all")
        output = console.file.getvalue()
        assert "Process" in output
        assert "Hardware" in output
        assert "Model" in output
        assert "Memory" in output
        assert "Memory Breakdown" in output
        assert "Runtime Details" in output
        assert "Optimization Analysis" in output

    def test_collect_filter_single_section_only_renders_that_section(self) -> None:
        console = _console()
        report = _base_report()
        panels.render_full_report(report, console, collect_filter="hardware")
        output = console.file.getvalue()
        assert "Hardware" in output
        assert "Memory Breakdown" not in output
        assert "Optimization Analysis" not in output
