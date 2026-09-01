"""
Rich panel renderers for llminspect inspect.

Each section of the report is rendered as an independent function so
the --collect flag can selectively render only matching sections.
"""

from __future__ import annotations

from datetime import UTC

from rich.console import Console
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from llm_inspector.models.measurement import Measurement
from llm_inspector.models.report import InspectionReport
from llm_inspector.ui.format import (
    fmt_bytes,
    fmt_measurement_bytes,
    fmt_measurement_str,
    fmt_uptime,
    truncate,
)

_SECTION_STYLE = "bold white"
_LABEL_STYLE = "dim"
_UNAVAIL_STYLE = "dim italic"


def _source(val_str: str, source: str) -> Text:
    """Render a measured value with its dim source annotation."""
    t = Text()
    t.append(val_str)
    t.append(f"  [{source}]", style="dim")
    return t


def _reason(reason: str) -> Text:
    """Render an unavailable value with its dim reason annotation."""
    t = Text()
    t.append("Unavailable", style="dim italic")
    t.append(f"  ({reason})", style="dim")
    return t


def _fmt_expiry(iso_str: str) -> str:
    """
    Convert an ISO-8601 expiry timestamp to a human-readable countdown.

    E.g. "2026-07-15T13:33:32+01:00" → "expires in 4m 52s"
    Falls back to the raw string if parsing fails.
    """
    try:
        from datetime import datetime  # noqa: PLC0415

        from dateutil import parser as dtparser  # type: ignore[import-untyped]  # noqa: PLC0415

        expiry = dtparser.parse(iso_str)
        now = datetime.now(tz=UTC)
        if expiry.tzinfo is None:
            expiry = expiry.replace(tzinfo=UTC)
        delta = expiry - now
        secs = int(delta.total_seconds())
        if secs < 0:
            return f"expired {abs(secs) // 60}m ago"
        h, rem = divmod(secs, 3600)
        m, s = divmod(rem, 60)
        if h:
            return f"expires in {h}h {m}m"
        if m:
            return f"expires in {m}m {s}s"
        return f"expires in {s}s"
    except Exception:  # noqa: BLE001
        return iso_str


def _kv_table() -> Table:
    """A borderless two-column key/value table."""
    t = Table(box=None, show_header=False, padding=(0, 2, 0, 0), expand=False)
    t.add_column("key", style=_LABEL_STYLE, min_width=20)
    t.add_column("value", min_width=24)
    return t


def _val(v: str) -> Text:
    if v == "Unavailable":
        return Text(v, style=_UNAVAIL_STYLE)
    return Text(v)


def render_header(console: Console, captured_at: str | None = None) -> None:
    from llm_inspector import __version__  # noqa: PLC0415

    console.print()
    console.rule(
        f"[bold]LLM INSPECTOR[/bold] [dim]v{__version__}[/dim]",
        style="bright_blue",
        characters="═",
    )
    if captured_at:
        console.print(f"  [dim]Snapshot taken at {captured_at}[/dim]")
    console.print()


def render_footer(console: Console) -> None:
    from llm_inspector import __repo__  # noqa: PLC0415

    console.print()
    console.rule(style="dim", characters="─")
    console.print(f"[dim]Measured. Not guessed.  {__repo__}[/dim]")
    console.print()


def render_process_section(
    report: InspectionReport, console: Console, verbose: bool = False
) -> None:
    console.print(Rule("Process", style=_SECTION_STYLE, align="left"))
    t = _kv_table()
    t.add_row("PID", str(report.pid))
    t.add_row("Runtime", report.process.runtime_kind.value)
    t.add_row("Backend", report.hardware.backend.value)
    raw_cmd = report.process.cmdline_str or ""
    cmd = raw_cmd if verbose else truncate(raw_cmd, 120)
    t.add_row("Command", cmd or "Unavailable")
    t.add_row(
        "Started",
        fmt_uptime(report.process.uptime_seconds) + " ago"
        if report.process.uptime_seconds
        else "Unavailable",
    )
    console.print(t)
    console.print()


def render_hardware_section(
    report: InspectionReport, console: Console, verbose: bool = False
) -> None:
    from llm_inspector.models.enums import BackendKind  # noqa: PLC0415

    console.print(Rule("Hardware", style=_SECTION_STYLE, align="left"))
    hw = report.hardware
    t = _kv_table()
    t.add_row("Backend", hw.backend.value)

    if hw.backend == BackendKind.CPU:
        # CPU / macOS — show system info instead of GPU fields
        t.add_row("CPU", _val(hw.cpu_name or "Unavailable"))
        cores = (
            f"{hw.cpu_physical_cores}P / {hw.cpu_logical_cores}L"
            if hw.cpu_physical_cores and hw.cpu_logical_cores
            else None
        )
        t.add_row("CPU Cores", _val(cores or "Unavailable"))
        cpu_util = (
            f"{hw.cpu_utilization_pct:.1f}%"
            if hw.cpu_utilization_pct is not None
            else "Unavailable"
        )
        t.add_row("CPU Utilization", _val(cpu_util))
        t.add_row("System RAM Total", _val(fmt_bytes(hw.system_ram_total_bytes)))
        t.add_row("System RAM Used", _val(fmt_bytes(hw.system_ram_used_bytes)))
        console.print(t)
        console.print()
        return

    # GPU backend
    attachments = hw.devices
    if len(attachments) <= 1:
        # Single-GPU path — same shape as before (+ free + index)
        gpu_label = hw.gpu_name or "Unavailable"
        if hw.device_index is not None and hw.gpu_name:
            gpu_label = f"{hw.gpu_name}  (GPU {hw.device_index})"
        t.add_row("GPU", _val(gpu_label))
        t.add_row("Driver", _val(hw.driver_version or "Unavailable"))
        t.add_row("CUDA", _val(hw.cuda_version or "Unavailable"))
        t.add_row("VRAM Total", _val(fmt_bytes(hw.vram_total_bytes)))
        t.add_row("VRAM Used", _val(fmt_bytes(hw.vram_used_bytes)))
        free = attachments[0].vram_free_bytes if attachments else None
        if free is None and hw.vram_total_bytes is not None and hw.vram_used_bytes is not None:
            free = max(0, hw.vram_total_bytes - hw.vram_used_bytes)
        t.add_row("VRAM Free", _val(fmt_bytes(free)))
        gpu_util = (
            f"{hw.gpu_utilization_pct}%" if hw.gpu_utilization_pct is not None else "Unavailable"
        )
        t.add_row("GPU Utilization", _val(gpu_util))
        t.add_row("System RAM Total", _val(fmt_bytes(hw.system_ram_total_bytes)))
        console.print(t)
        console.print()
        return

    # Multi-GPU — process-level summary, then one measured block per GPU
    ids = ",".join(str(d.index) for d in attachments)
    t.add_row("GPUs", _val(f"{len(attachments)} devices  ({ids})"))
    t.add_row("Driver", _val(hw.driver_version or "Unavailable"))
    t.add_row("CUDA", _val(hw.cuda_version or "Unavailable"))
    t.add_row("Process VRAM Total", _val(fmt_bytes(hw.process_vram_total_bytes)))
    if verbose:
        t.add_row(
            "",
            Text(
                "  [NVML sum of process usedGpuMemory across attached GPUs]",
                style="dim",
            ),
        )
    t.add_row("System RAM Total", _val(fmt_bytes(hw.system_ram_total_bytes)))
    console.print(t)
    console.print()

    torch_by_index = {}
    if report.memory is not None:
        torch_by_index = {d.device_index: d for d in report.memory.devices}

    for att in attachments:
        console.print(Rule(f"GPU {att.index}", style=_SECTION_STYLE, align="left"))
        gt = _kv_table()
        gt.add_row("Name", _val(att.name or "Unavailable"))
        device_line = (
            f"{fmt_bytes(att.vram_used_bytes)} / {fmt_bytes(att.vram_total_bytes)}"
            if att.vram_total_bytes is not None
            else fmt_bytes(att.vram_used_bytes)
        )
        free_s = fmt_bytes(att.vram_free_bytes)
        util_s = (
            f"{att.gpu_utilization_pct}%" if att.gpu_utilization_pct is not None else "Unavailable"
        )
        gt.add_row("Device VRAM", _val(f"{device_line}  (free {free_s})"))
        gt.add_row("GPU Utilization", _val(util_s))
        gt.add_row("Process VRAM", _val(fmt_bytes(att.process_vram_bytes)))
        if verbose:
            gt.add_row(
                "",
                Text(
                    f"  [NVML compute process on GPU {att.index}]",
                    style="dim",
                ),
            )

        torch_m = torch_by_index.get(att.index)
        if torch_m is not None:
            for label, meas in (
                ("GPU Allocated", torch_m.allocated),
                ("GPU Reserved", torch_m.reserved),
                ("Peak GPU", torch_m.peak),
            ):
                val_str = fmt_measurement_bytes(meas)
                if verbose and meas.is_available and meas.source:
                    gt.add_row(label, _source(val_str, meas.source))
                elif verbose and not meas.is_available and meas.reason:
                    gt.add_row(label, _reason(meas.reason))
                else:
                    gt.add_row(label, _val(val_str))
        elif verbose:
            gt.add_row(
                "GPU Allocated",
                _reason(
                    "Per-device torch metrics require embedded attach() in the inference process."
                ),
            )

        console.print(gt)
        console.print()


def render_model_section(report: InspectionReport, console: Console, verbose: bool = False) -> None:
    console.print(Rule("Model", style=_SECTION_STYLE, align="left"))
    if report.model is None:
        console.print("[dim]  Model information unavailable (no runtime plugin active).[/dim]")
        console.print()
        return

    m = report.model
    t = _kv_table()

    def mrow_str(label: str, meas: Measurement[str]) -> None:
        val_str = fmt_measurement_str(meas)
        if verbose and meas.is_available and meas.source:
            t.add_row(label, _source(val_str, meas.source))
        elif verbose and not meas.is_available and meas.reason:
            t.add_row(label, _reason(meas.reason))
        else:
            t.add_row(label, _val(val_str))

    def mrow_int(label: str, meas: Measurement[int], formatter=str) -> None:
        val_str = (
            formatter(meas.value) if meas.is_available and meas.value is not None else "Unavailable"
        )
        if verbose and meas.is_available and meas.source:
            t.add_row(label, _source(val_str, meas.source))
        elif verbose and not meas.is_available and meas.reason:
            t.add_row(label, _reason(meas.reason))
        else:
            t.add_row(label, _val(val_str))

    mrow_str("Name", m.name)
    mrow_str("Architecture", m.architecture)
    mrow_str("Precision", m.precision)

    # Parameters — show raw string from Ollama (e.g. "4.5B") when int count unavailable
    if m.parameter_count.is_available and m.parameter_count.value is not None:
        n = m.parameter_count.value
        param_str = f"{n / 1e9:.1f}B" if n >= 1_000_000_000 else f"{n / 1e6:.0f}M"
        if verbose and m.parameter_count.source:
            t.add_row("Parameters", _source(param_str, m.parameter_count.source))
        else:
            t.add_row("Parameters", _val(param_str))
    else:
        # Check if Ollama gave us a human-readable string (e.g. "4.5B")
        raw_hint = ""
        if m.parameter_count.reason and "parameter_size='" in m.parameter_count.reason:
            import re  # noqa: PLC0415

            match = re.search(r"parameter_size='([^']+)'", m.parameter_count.reason)
            if match:
                raw_hint = match.group(1)
        if raw_hint:
            label_val = Text()
            label_val.append(raw_hint)
            label_val.append("  (approx, from Ollama)", style="dim")
            t.add_row("Parameters", label_val)
        elif verbose and m.parameter_count.reason:
            t.add_row("Parameters", _reason(m.parameter_count.reason))
        else:
            t.add_row("Parameters", _val("Unavailable"))

    mrow_int("Context Length", m.context_length, lambda n: f"{n:,} tokens")
    mrow_int("Tensor Parallel", m.tensor_parallel, lambda n: f"×{n}")
    mrow_int("Pipeline Parallel", m.pipeline_parallel, lambda n: f"×{n}")

    console.print(t)
    console.print()

    # Phase 5 — tokenizer / architecture / module buckets
    t2 = _kv_table()
    has_details = False

    def detail_str(label: str, meas: Measurement[str]) -> None:
        nonlocal has_details
        if not meas.is_available and not verbose:
            return
        has_details = True
        val_str = str(meas.value) if meas.is_available and meas.value is not None else "Unavailable"
        if verbose and meas.is_available and meas.source:
            t2.add_row(label, _source(val_str, meas.source))
        elif verbose and not meas.is_available and meas.reason:
            t2.add_row(label, _reason(meas.reason))
        elif meas.is_available:
            t2.add_row(label, _val(val_str))

    def detail_int(label: str, meas: Measurement[int], formatter=str) -> None:
        nonlocal has_details
        if not meas.is_available and not verbose:
            return
        has_details = True
        if meas.is_available and meas.value is not None:
            val_str = formatter(meas.value)
            if verbose and meas.source:
                t2.add_row(label, _source(val_str, meas.source))
            else:
                t2.add_row(label, _val(val_str))
        elif verbose and meas.reason:
            t2.add_row(label, _reason(meas.reason))

    def fmt_params(n: int) -> str:
        if n >= 1_000_000_000:
            return f"{n / 1e9:.2f}B"
        if n >= 1_000_000:
            return f"{n / 1e6:.1f}M"
        return f"{n:,}"

    detail_str("Tokenizer", m.tokenizer_class)
    detail_int("Vocab Size", m.vocab_size, lambda n: f"{n:,}")
    detail_str("Chat Template", m.chat_template)
    detail_str("BOS Token", m.bos_token)
    detail_str("EOS Token", m.eos_token)
    detail_int("Layers", m.num_layers, lambda n: f"{n:,}")
    detail_int("Hidden Size", m.hidden_size, lambda n: f"{n:,}")
    detail_int("Attention Heads", m.num_attention_heads, lambda n: f"{n:,}")
    detail_int("KV Heads", m.num_kv_heads, lambda n: f"{n:,}")
    detail_int("Experts (MoE)", m.num_experts, lambda n: f"{n:,}")
    detail_int("Embed Params", m.embed_params, fmt_params)
    detail_int("Transformer Params", m.transformer_params, fmt_params)
    detail_int("Head Params", m.head_params, fmt_params)

    if has_details:
        console.print(Rule("Model Details", style=_SECTION_STYLE, align="left"))
        console.print(t2)
        console.print()


def render_memory_section(
    report: InspectionReport, console: Console, verbose: bool = False
) -> None:
    from llm_inspector.models.enums import BackendKind  # noqa: PLC0415

    console.print(Rule("Memory", style=_SECTION_STYLE, align="left"))
    if report.memory is None:
        console.print("[dim]  Memory data unavailable.[/dim]")
        console.print()
        return

    m = report.memory
    t = _kv_table()
    is_cpu = report.hardware.backend == BackendKind.CPU

    def row(label: str, measurement: Measurement[int]) -> None:
        val_str = fmt_measurement_bytes(measurement)
        if verbose and measurement.is_available and measurement.source:
            t.add_row(label, _source(val_str, measurement.source))
        elif verbose and not measurement.is_available and measurement.reason:
            t.add_row(label, _reason(measurement.reason))
        else:
            t.add_row(label, _val(val_str))

    # GPU fields — only shown when a GPU backend is active
    if not is_cpu:
        row("GPU Used", m.gpu_used)
        row("GPU Allocated", m.gpu_allocated)
        row("GPU Reserved", m.gpu_reserved)
        row("Peak GPU", m.peak)

    row("Process RAM", m.process_ram)
    console.print(t)
    console.print()


def render_memory_breakdown_section(
    report: InspectionReport, console: Console, verbose: bool = False
) -> None:
    console.print(Rule("Memory Breakdown", style=_SECTION_STYLE, align="left"))
    if report.memory_breakdown is None:
        console.print("[dim]  Memory breakdown unavailable (requires runtime plugin).[/dim]")
        console.print()
        return

    bd = report.memory_breakdown
    t = _kv_table()

    for component in bd.sorted_components():
        val_str = fmt_measurement_bytes(component.measurement)
        if verbose and component.measurement.is_available and component.measurement.source:
            t.add_row(component.name, _source(val_str, component.measurement.source))
        elif verbose and not component.measurement.is_available and component.measurement.reason:
            t.add_row(component.name, _reason(component.measurement.reason))
        else:
            t.add_row(component.name, _val(val_str))

    console.print(t)

    total_str = fmt_measurement_bytes(bd.total)
    console.print(Rule(style="dim", characters="─"))
    t2 = _kv_table()
    t2.add_row("[bold]Total[/bold]", Text(total_str, style="bold"))
    console.print(t2)
    console.print()


def render_runtime_section(
    report: InspectionReport, console: Console, verbose: bool = False
) -> None:
    console.print(Rule("Runtime Details", style=_SECTION_STYLE, align="left"))
    if report.runtime is None:
        console.print("[dim]  Runtime details unavailable.[/dim]")
        console.print()
        return

    rt = report.runtime
    t = _kv_table()
    t.add_row("Runtime", rt.runtime_kind.value)

    version_str = fmt_measurement_str(rt.version)
    if verbose and rt.version.is_available and rt.version.source:
        t.add_row("Version", _source(version_str, rt.version.source))
    else:
        t.add_row("Version", _val(version_str))

    for key, measurement in rt.details.items():
        val_str = fmt_measurement_str(measurement)
        # Special-case: format ISO expiry timestamps as human-readable
        if "expires" in key.lower() and measurement.is_available and measurement.value:
            val_str = _fmt_expiry(measurement.value)
        if verbose and measurement.is_available and measurement.source:
            t.add_row(key, _source(val_str, measurement.source))
        else:
            t.add_row(key, _val(val_str))

    console.print(t)
    console.print()


def render_optimization_section(
    report: InspectionReport, console: Console, verbose: bool = False
) -> None:
    """Render Optimization Analysis (Projected) — continuation of inspect."""
    from llm_inspector.ui.format import fmt_projected_bytes  # noqa: PLC0415

    opt = report.optimization
    console.print(Rule("Optimization Analysis (Projected)", style=_SECTION_STYLE, align="left"))

    if opt is None:
        console.print(
            "[dim]  Optimization Analysis unavailable (insufficient measured data).[/dim]"
        )
        console.print()
        return

    if opt.skipped_reason:
        console.print(f"[dim]  {opt.skipped_reason}[/dim]")
        console.print()
        return

    for group in opt.groups:
        console.print(f"  [bold]{group.title}[/bold]")
        if group.note:
            console.print(f"  [dim]{group.note}[/dim]")

        t = Table(box=None, show_header=True, padding=(0, 2))
        t.add_column("Method", style="bold", min_width=16)
        t.add_column("New Total", min_width=12)
        t.add_column("Saved", min_width=12)
        if verbose:
            t.add_column("Quality", min_width=12)
            t.add_column("Source", style="dim")

        for s in group.scenarios:
            total_s = fmt_projected_bytes(s.new_total)
            saved_s = fmt_projected_bytes(s.saved_bytes)
            if verbose:
                src = s.new_total.source or s.new_total.reason or ""
                t.add_row(
                    s.method_name,
                    total_s,
                    saved_s,
                    s.quality.value,
                    src,
                )
            else:
                # Hide unsupported/unavailable rows in compact view
                if not s.new_total.has_value:
                    continue
                t.add_row(s.method_name, total_s, saved_s)

        console.print(t)
        console.print()

    if opt.recommendation is not None:
        rec = opt.recommendation
        console.print("  [bold]Recommendation[/bold]")
        console.print(f"  ✓ {rec.summary}")
        for warning in rec.warnings:
            console.print(f"  ⚠ {warning}")
        console.print()


def render_full_report(
    report: InspectionReport,
    console: Console,
    collect_filter: str = "all",
    verbose: bool = False,
) -> None:
    """
    Render a complete InspectionReport.

    If collect_filter is set to a specific collector name, only that
    section is rendered (plus the process header which always shows).
    """
    captured_at = report.captured_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    render_header(console, captured_at=captured_at)

    show_all = collect_filter == "all"

    if show_all or collect_filter == "process":
        render_process_section(report, console, verbose=verbose)

    if show_all or collect_filter == "hardware":
        render_hardware_section(report, console, verbose=verbose)

    if show_all or collect_filter == "model":
        render_model_section(report, console, verbose=verbose)

    if show_all or collect_filter == "memory":
        render_memory_section(report, console, verbose=verbose)

    if show_all or collect_filter == "memory-breakdown":
        render_memory_breakdown_section(report, console, verbose=verbose)

    if show_all or collect_filter == "runtime":
        render_runtime_section(report, console, verbose=verbose)

    # Continuation: projected optimizations (not a separate command)
    if show_all:
        render_optimization_section(report, console, verbose=verbose)

    render_footer(console)


def _pick_primary_report(reports: list[InspectionReport]) -> InspectionReport:
    """Prefer EngineCore / report with model data for process-level sections."""
    from llm_inspector.utils.vllm_group import worker_label  # noqa: PLC0415

    for r in reports:
        if worker_label(r.pid) == "EngineCore":
            return r
    for r in reports:
        if r.model is not None and r.model.name.is_available:
            return r
    return reports[0]


def render_tp_group_report(
    reports: list[InspectionReport],
    console: Console,
    collect_filter: str = "all",
    verbose: bool = False,
) -> None:
    """
    Render one vLLM TP job spanning multiple Worker_TP* / EngineCore PIDs.

    Process-level Model / Runtime / Optimization come from the primary member.
    Each GPU-attached worker gets its own measured Hardware + Memory block.
    """
    from llm_inspector.models.enums import BackendKind  # noqa: PLC0415
    from llm_inspector.utils.vllm_group import worker_label  # noqa: PLC0415

    if len(reports) == 1:
        render_full_report(reports[0], console, collect_filter=collect_filter, verbose=verbose)
        return

    primary = _pick_primary_report(reports)
    captured_at = primary.captured_at.strftime("%Y-%m-%d %H:%M:%S UTC")
    render_header(console, captured_at=captured_at)

    show_all = collect_filter == "all"

    # ── Process group (once) ────────────────────────────────────────────────
    console.print(Rule("Process Group (vLLM TP)", style=_SECTION_STYLE, align="left"))
    gt = _kv_table()
    gt.add_row("Members", _val(str(len(reports))))
    pids = ", ".join(str(r.pid) for r in reports)
    gt.add_row("PIDs", _val(pids))
    console.print(gt)
    console.print()

    for r in reports:
        label = worker_label(r.pid)
        gpus = (
            ",".join(str(i) for i in r.hardware.gpu_indices)
            if r.hardware.gpu_indices
            else (str(r.hardware.device_index) if r.hardware.device_index is not None else "—")
        )
        console.print(f"  {label:<14} PID {r.pid:<10} GPU(s) {gpus}")
    console.print()

    if show_all or collect_filter == "model":
        render_model_section(primary, console, verbose=verbose)

    # ── Per-member GPU / memory (measured) ──────────────────────────────────
    if show_all or collect_filter in ("hardware", "memory", "memory-breakdown", "all"):
        for r in reports:
            label = worker_label(r.pid)
            # Skip EngineCore if it has no GPU attachment (CPU coordinator)
            if (
                label == "EngineCore"
                and r.hardware.backend == BackendKind.CPU
                and not r.hardware.devices
            ):
                continue

            gpu_ids = (
                ",".join(str(i) for i in r.hardware.gpu_indices)
                if r.hardware.gpu_indices
                else (str(r.hardware.device_index) if r.hardware.device_index is not None else "?")
            )
            console.print(
                Rule(
                    f"{label}  ·  PID {r.pid}  ·  GPU {gpu_ids}",
                    style=_SECTION_STYLE,
                    align="left",
                )
            )
            if show_all or collect_filter == "hardware":
                render_hardware_section(r, console, verbose=verbose)
            if show_all or collect_filter == "memory":
                render_memory_section(r, console, verbose=verbose)
            if show_all or collect_filter == "memory-breakdown":
                render_memory_breakdown_section(r, console, verbose=verbose)

    # ── Totals across GPU workers ───────────────────────────────────────────
    if show_all or collect_filter in ("hardware", "memory", "all"):
        console.print(Rule("TP Job Totals (Measured)", style=_SECTION_STYLE, align="left"))
        tt = _kv_table()
        vram_parts = [
            r.hardware.process_vram_total_bytes
            if r.hardware.process_vram_total_bytes is not None
            else r.hardware.vram_used_bytes
            for r in reports
            if worker_label(r.pid) != "EngineCore"
            or r.hardware.devices
            or r.hardware.device_index is not None
        ]
        measured = [v for v in vram_parts if v is not None]
        if measured:
            tt.add_row("Process VRAM Total", _val(fmt_bytes(sum(measured))))
            if verbose:
                tt.add_row(
                    "",
                    Text(
                        "  [sum of NVML process VRAM across TP worker PIDs]",
                        style="dim",
                    ),
                )
        gpu_set: list[int] = []
        for r in reports:
            gpu_set.extend(r.hardware.gpu_indices)
            if not r.hardware.gpu_indices and r.hardware.device_index is not None:
                gpu_set.append(r.hardware.device_index)
        if gpu_set:
            uniq = sorted(set(gpu_set))
            tt.add_row("GPUs used", _val(",".join(str(i) for i in uniq)))
        console.print(tt)
        console.print()

    if show_all or collect_filter == "runtime":
        render_runtime_section(primary, console, verbose=verbose)

    if show_all:
        render_optimization_section(primary, console, verbose=verbose)

    render_footer(console)
