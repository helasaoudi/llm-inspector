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
from llm_inspector.utils.vllm_group import (
    collapse_tp_reports_for_ps,
    worker_label,
    worker_tp_count,
)


def _short_model_name(r: InspectionReport, width: int = 24) -> str:
    if r.model and r.model.name.is_available and r.model.name.value:
        raw = r.model.name.value
        # Prefer last path segment / HF repo leaf over long paths
        leaf = raw.rstrip("/").split("/")[-1] if "/" in raw else raw
        return truncate(leaf, width)
    return "—"


def _vram_bytes(r: InspectionReport) -> int | None:
    if r.hardware.process_vram_total_bytes is not None:
        return r.hardware.process_vram_total_bytes
    return r.hardware.vram_used_bytes


def render_ps_table(reports: list[InspectionReport], console: Console) -> None:
    """
    Render a summary table of all detected LLM processes.

    vLLM EngineCore + Worker_TP* siblings collapse into one TP job row.
    Works on GPU machines and macOS (CPU/Metal — GPU columns show '—').
    """
    if not reports:
        console.print(
            "[dim]No LLM processes found. "
            "Make sure a model is loaded (e.g. 'ollama run llama3') "
            "and try again.[/dim]"
        )
        return

    groups = collapse_tp_reports_for_ps(reports)
    has_gpu = any(r.hardware.backend != BackendKind.CPU for r, _ in groups)

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

    # Prefer jobs with measured VRAM so TP rows are not buried
    def sort_key(item: tuple[InspectionReport, list[InspectionReport]]) -> tuple[int, int]:
        rep, members = item
        vrams = [_vram_bytes(m) for m in members]
        total = sum(v for v in vrams if v is not None)
        return (0 if total else 1, -total)

    for rep, members in sorted(groups, key=sort_key):
        runtime = rep.process.runtime_kind.value
        # Best model name among members (workers often have attach data)
        model_str = "—"
        for m in members:
            candidate = _short_model_name(m)
            if candidate != "—":
                model_str = candidate
                break

        uptime = fmt_uptime(rep.process.uptime_seconds)
        backend = rep.hardware.backend.value

        tp_n = worker_tp_count([m.pid for m in members])
        if len(members) > 1 and tp_n >= 1:
            # Lead with EngineCore PID when present, else representative
            lead = next(
                (m for m in members if worker_label(m.pid) == "EngineCore"),
                rep,
            )
            pid_str = f"{lead.pid} TP×{tp_n}"
        else:
            pid_str = str(rep.pid)

        if has_gpu:
            gpu_ids: list[int] = []
            for m in members:
                gpu_ids.extend(m.hardware.gpu_indices)
                if not m.hardware.gpu_indices and m.hardware.device_index is not None:
                    gpu_ids.append(m.hardware.device_index)
            gpus = ",".join(str(i) for i in sorted(set(gpu_ids))) if gpu_ids else "—"

            utils = [
                m.hardware.gpu_utilization_pct
                for m in members
                if m.hardware.gpu_utilization_pct is not None
            ]
            gpu_util = f"{max(utils)}%" if utils else "—"

            vrams = [_vram_bytes(m) for m in members]
            measured = [v for v in vrams if v is not None]
            # Sum VRAM across TP workers (each holds its shard); single process: as-is
            if len(members) > 1 and measured:
                vram = fmt_bytes(sum(measured))
            elif measured:
                vram = fmt_bytes(measured[0])
            else:
                vram = "—"

            table.add_row(pid_str, runtime, model_str, backend, gpus, gpu_util, vram, uptime)
        else:
            ram = "—"
            if rep.memory and rep.memory.process_ram.is_available and rep.memory.process_ram.value:
                ram = fmt_bytes(rep.memory.process_ram.value)
            table.add_row(pid_str, runtime, model_str, backend, ram, uptime)

    console.print(table)
