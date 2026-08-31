"""
Rich table renderers for llminspect ps and related list views.
"""

from __future__ import annotations

from rich import box
from rich.console import Console
from rich.table import Table

from llm_inspector.models.enums import BackendKind
from llm_inspector.models.report import InspectionReport
from llm_inspector.ui.format import fmt_bytes, fmt_uptime, truncate


def render_ps_table(reports: list[InspectionReport], console: Console) -> None:
    """
    Render a summary table of all detected LLM processes.

    Works on both GPU machines (CUDA/ROCm columns filled) and macOS
    (CPU/Metal backend — GPU columns show '—' gracefully).

    Used by ``llminspect ps``.
    """
    if not reports:
        console.print(
            "[dim]No LLM processes found. "
            "Make sure a model is loaded (e.g. 'ollama run llama3') "
            "and try again.[/dim]"
        )
        return

    # Decide which columns to show based on what's available
    has_gpu = any(r.hardware.backend != BackendKind.CPU for r in reports)

    table = Table(
        box=box.SIMPLE_HEAD,
        show_header=True,
        header_style="bold",
        expand=False,
        padding=(0, 1),
    )

    table.add_column("PID", style="cyan", no_wrap=True)
    table.add_column("Runtime")
    table.add_column("Model")
    table.add_column("Backend", no_wrap=True)
    if has_gpu:
        table.add_column("GPUs", no_wrap=True)
        table.add_column("GPU%", justify="right", no_wrap=True)
        table.add_column("VRAM", justify="right", no_wrap=True)
    else:
        table.add_column("RAM", justify="right", no_wrap=True)
    table.add_column("Up", no_wrap=True)

    for r in reports:
        runtime = r.process.runtime_kind.value

        # Model name — from plugin or fallback to exe basename
        model_str = "—"
        if r.model and r.model.name.is_available and r.model.name.value:
            model_str = truncate(r.model.name.value, 20)

        uptime = fmt_uptime(r.process.uptime_seconds)
        backend = r.hardware.backend.value

        if has_gpu:
            if r.hardware.gpu_indices:
                gpus = ",".join(str(i) for i in r.hardware.gpu_indices)
            elif r.hardware.device_index is not None:
                gpus = str(r.hardware.device_index)
            else:
                gpus = "—"

            gpu_util = (
                f"{r.hardware.gpu_utilization_pct}%"
                if r.hardware.gpu_utilization_pct is not None
                else "—"
            )
            # Prefer summed process VRAM across GPUs when available
            vram_bytes = r.hardware.process_vram_total_bytes
            if vram_bytes is None:
                vram_bytes = r.hardware.vram_used_bytes
            vram = fmt_bytes(vram_bytes)
            table.add_row(str(r.pid), runtime, model_str, backend, gpus, gpu_util, vram, uptime)
        else:
            # CPU/Metal backend — show process RAM instead
            ram = "—"
            if r.memory and r.memory.process_ram.is_available and r.memory.process_ram.value:
                ram = fmt_bytes(r.memory.process_ram.value)
            table.add_row(str(r.pid), runtime, model_str, backend, ram, uptime)

    console.print(table)
