"""
LLM Inspector CLI — entry point only. Zero business logic.

Every command delegates to Inspector and passes results to the UI layer.
No data processing, no formatting, no fallback logic lives here.
"""

from __future__ import annotations

from typing import Annotated

import typer
from rich.console import Console

app = typer.Typer(
    name="llminspect",
    help="The htop for LLM inference. Measured. Not guessed.",
    no_args_is_help=True,
    pretty_exceptions_enable=False,
)

console = Console()


@app.command("ps")
def cmd_ps() -> None:
    """List all GPU-attached LLM inference processes."""
    from llm_inspector.inspector.core import Inspector  # noqa: PLC0415
    from llm_inspector.ui.tables import render_ps_table  # noqa: PLC0415

    inspector = Inspector()
    reports = inspector.scan()
    render_ps_table(reports, console)


@app.command("inspect")
def cmd_inspect(
    pid: Annotated[int, typer.Argument(help="PID of the process to inspect.")],
    collect: Annotated[
        str,
        typer.Option(
            "--collect",
            help=(
                "Run only one section: "
                "process | hardware | model | memory | memory-breakdown | runtime | all"
            ),
        ),
    ] = "all",
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Show data source for every field."),
    ] = False,
) -> None:
    """Inspect a single LLM inference process."""
    from llm_inspector.inspector.core import Inspector  # noqa: PLC0415
    from llm_inspector.ui.panels import render_full_report  # noqa: PLC0415

    inspector = Inspector()
    try:
        report = inspector.inspect(pid=pid, collect_filter=collect)
    except RuntimeError as exc:
        console.print(f"[red]Error:[/red] {exc}")
        raise typer.Exit(1) from exc

    render_full_report(report, console, collect_filter=collect, verbose=verbose)


@app.command("gpu")
def cmd_gpu() -> None:
    """Show GPU device summary (VRAM, utilisation, driver version)."""
    from rich import box as rbox  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    from llm_inspector.backends.registry import BackendRegistry  # noqa: PLC0415
    from llm_inspector.ui.format import fmt_bytes  # noqa: PLC0415

    registry = BackendRegistry()
    backend = registry.best_available()
    devices = backend.list_devices()

    if not devices:
        console.print("[dim]No GPU devices detected.[/dim]")
        return

    n = len(devices)
    console.print(f"[bold]GPUs[/bold]  {n} device{'s' if n != 1 else ''} ({backend.kind.value})")

    t = Table(box=rbox.SIMPLE_HEAD, header_style="bold", padding=(0, 1))
    t.add_column("IDX", style="cyan", min_width=3)
    t.add_column("Name")
    t.add_column("Total", justify="right")
    t.add_column("Used", justify="right")
    t.add_column("Free", justify="right")
    t.add_column("Util", justify="right")
    t.add_column("Driver")
    t.add_column("CUDA")

    for d in devices:
        free = None
        if d.vram_total_bytes is not None and d.vram_used_bytes is not None:
            free = max(0, d.vram_total_bytes - d.vram_used_bytes)
        t.add_row(
            str(d.index),
            d.name,
            fmt_bytes(d.vram_total_bytes),
            fmt_bytes(d.vram_used_bytes),
            fmt_bytes(free),
            f"{d.gpu_utilization_pct}%" if d.gpu_utilization_pct is not None else "—",
            d.driver_version or "—",
            d.cuda_version or "—",
        )

    console.print(t)

    driver = devices[0].driver_version or backend.driver_version()
    cuda = devices[0].cuda_version or backend.runtime_version()
    if driver or cuda:
        console.print(f"[dim]Driver {driver or '—'} · CUDA {cuda or '—'}[/dim]")


@app.command("runtimes")
def cmd_runtimes() -> None:
    """List registered runtime plugins and their capabilities."""
    from rich import box as rbox  # noqa: PLC0415
    from rich.table import Table  # noqa: PLC0415

    from llm_inspector.inspector.core import Inspector  # noqa: PLC0415

    inspector = Inspector()
    plugins = inspector.registered_plugins()

    t = Table(box=rbox.SIMPLE_HEAD, header_style="bold", padding=(0, 1))
    t.add_column("Runtime", style="cyan")
    t.add_column("Status")
    t.add_column("Description")

    for plugin in plugins:
        status = "[green]Active[/green]"
        t.add_row(plugin.display_name, status, plugin.description)  # type: ignore[attr-defined]

    console.print(t)
