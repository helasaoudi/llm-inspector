"""
Typed, immutable result classes returned by each collector.

Design rules
------------
- All result classes are frozen Pydantic models (value objects).
- Phase A results (ProcessResult, HardwareResult) use primitive types —
  their provenance is always the same well-known source.
- Phase B results (MemoryResult, ModelResult, RuntimeResult,
  MemoryBreakdownResult) use Measurement[T] — provenance matters because
  different runtimes expose the same data through different APIs.
- ``None`` is the only sentinel for "not available".
  No sentinel strings like "N/A" or "unknown" are ever used.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from llm_inspector.models.enums import BackendKind, RuntimeKind
from llm_inspector.models.measurement import Measurement


# ── Phase A results — primitive types ─────────────────────────────────────────


class ProcessResult(BaseModel):
    """
    OS-level metadata about the inference process.

    Collected by ProcessCollector in Phase A using /proc/<pid>/cmdline,
    /proc/<pid>/exe, and psutil.  All fields are plain types because the
    data source is unambiguous — always /proc or psutil.
    """

    model_config = ConfigDict(frozen=True)

    pid: int
    ppid: int | None = None
    name: str | None = None
    exe: str | None = None
    cmdline: list[str] = Field(default_factory=list)
    start_time: datetime | None = None
    uptime_seconds: float | None = None
    runtime_kind: RuntimeKind = RuntimeKind.UNKNOWN

    @property
    def cmdline_str(self) -> str:
        return " ".join(self.cmdline)

    @property
    def uptime_human(self) -> str:
        """Human-readable uptime, e.g. '14m', '2h 3m'."""
        if self.uptime_seconds is None:
            return "unknown"
        secs = int(self.uptime_seconds)
        h, rem = divmod(secs, 3600)
        m = rem // 60
        if h:
            return f"{h}h {m}m"
        if m:
            return f"{m}m"
        return f"{secs}s"


class HardwareResult(BaseModel):
    """
    Hardware device snapshot for the GPU (or CPU fallback) running inference.

    Collected by HardwareCollector in Phase A via the HardwareBackend
    abstraction (NVML for CUDA, sysfs for ROCm, IOKit for Metal).

    GPU fields are None on CPU-only machines.
    CPU/system fields are always populated via psutil.
    Plain types — the backend is always the same well-known source.
    """

    model_config = ConfigDict(frozen=True)

    backend: BackendKind
    device_index: int | None = None

    # GPU fields — populated on CUDA/ROCm/Metal backends
    gpu_name: str | None = None
    vram_total_bytes: int | None = None
    vram_used_bytes: int | None = None
    gpu_utilization_pct: int | None = None
    driver_version: str | None = None
    cuda_version: str | None = None

    # CPU/system fields — always populated
    cpu_name: str | None = None
    cpu_physical_cores: int | None = None
    cpu_logical_cores: int | None = None
    cpu_utilization_pct: float | None = None
    system_ram_total_bytes: int | None = None
    system_ram_used_bytes: int | None = None

    @property
    def vram_total_gb(self) -> float | None:
        return self.vram_total_bytes / (1024**3) if self.vram_total_bytes else None

    @property
    def vram_used_gb(self) -> float | None:
        return self.vram_used_bytes / (1024**3) if self.vram_used_bytes else None


# ── Phase B results — Measurement[T] types ────────────────────────────────────


class MemoryResult(BaseModel):
    """
    Raw, measurable memory facts about the inference process.

    Does NOT explain where memory is going — that is MemoryBreakdownResult.
    Uses Measurement[int] because provenance matters:
      - gpu_used comes from NVML (always available)
      - gpu_allocated / gpu_reserved come from PyTorch (requires plugin)
      - peak comes from PyTorch max tracker (requires plugin)
    All values are in bytes.
    """

    model_config = ConfigDict(frozen=True)

    process_ram: Measurement[int] = Field(
        default_factory=lambda: Measurement.unavailable("Not collected"),
        description="Process RSS from psutil.memory_info()",
    )
    gpu_used: Measurement[int] = Field(
        default_factory=lambda: Measurement.unavailable("Not collected"),
        description="GPU memory used — NVML nvmlDeviceGetMemoryInfo()",
    )
    gpu_allocated: Measurement[int] = Field(
        default_factory=lambda: Measurement.unavailable("Not collected"),
        description="PyTorch allocated memory — torch.cuda.memory_allocated()",
    )
    gpu_reserved: Measurement[int] = Field(
        default_factory=lambda: Measurement.unavailable("Not collected"),
        description="PyTorch reserved memory — torch.cuda.memory_reserved()",
    )
    peak: Measurement[int] = Field(
        default_factory=lambda: Measurement.unavailable("Not collected"),
        description="Peak GPU memory — torch.cuda.max_memory_allocated()",
    )


class ModelResult(BaseModel):
    """
    Identity of the loaded model.

    Uses Measurement[T] because source matters:
      - name might come from cmdline, /v1/models API, or config.json
      - precision might come from cmdline --dtype or dtype inspection
      - parameter_count might come from config.json or named_parameters()
    """

    model_config = ConfigDict(frozen=True)

    name: Measurement[str] = Field(
        default_factory=lambda: Measurement.unavailable("Not collected"),
    )
    architecture: Measurement[str] = Field(
        default_factory=lambda: Measurement.unavailable("Not collected"),
    )
    parameter_count: Measurement[int] = Field(
        default_factory=lambda: Measurement.unavailable("Not collected"),
    )
    precision: Measurement[str] = Field(
        default_factory=lambda: Measurement.unavailable("Not collected"),
    )
    context_length: Measurement[int] = Field(
        default_factory=lambda: Measurement.unavailable("Not collected"),
    )
    tensor_parallel: Measurement[int] = Field(
        default_factory=lambda: Measurement.unavailable("Not collected"),
    )
    pipeline_parallel: Measurement[int] = Field(
        default_factory=lambda: Measurement.unavailable("Not collected"),
    )


class MemoryComponent(BaseModel):
    """
    One independently measurable slice of GPU memory.

    The component-based design is the key extensibility point: adding
    LoRA adapters, prefix cache, speculative decoding buffers, or any
    future category requires only a new MemoryComponent — no schema
    changes to MemoryBreakdownResult or any other class.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    measurement: Measurement[int]
    description: str | None = None
    order: int = 99


# Well-known component names (constants, not enum — plugins may add freely)
class ComponentName:
    WEIGHTS = "Weights"
    KV_CACHE = "KV Cache"
    ACTIVATIONS = "Activations"
    WORKSPACE = "Workspace"
    RUNTIME_BUFFERS = "Runtime Buffers"
    LORA_ADAPTERS = "LoRA Adapters"
    PREFIX_CACHE = "Prefix Cache"
    OTHER = "Other"


# Default display order for well-known names
COMPONENT_ORDER: dict[str, int] = {
    ComponentName.WEIGHTS: 0,
    ComponentName.KV_CACHE: 1,
    ComponentName.ACTIVATIONS: 2,
    ComponentName.WORKSPACE: 3,
    ComponentName.RUNTIME_BUFFERS: 4,
    ComponentName.LORA_ADAPTERS: 5,
    ComponentName.PREFIX_CACHE: 6,
    ComponentName.OTHER: 99,
}


class MemoryBreakdownResult(BaseModel):
    """
    Explains where GPU memory is consumed.

    This is the flagship result of LLM Inspector v1.

    components is a list of MemoryComponent objects, each with its own
    Measurement[int] (bytes + source + status).  The list is sorted by
    component.order for consistent display.

    Adding a new memory category never requires a schema change — the
    plugin simply appends a new MemoryComponent.
    """

    model_config = ConfigDict(frozen=True)

    components: list[MemoryComponent] = Field(default_factory=list)
    total: Measurement[int] = Field(
        default_factory=lambda: Measurement.unavailable("No components measured"),
    )

    def get(self, name: str) -> MemoryComponent | None:
        """Look up a component by name."""
        return next((c for c in self.components if c.name == name), None)

    def sorted_components(self) -> list[MemoryComponent]:
        """Return components sorted by display order."""
        return sorted(self.components, key=lambda c: c.order)


class RuntimeResult(BaseModel):
    """
    Runtime-specific metadata that does not fit other collectors.

    details is a dict of Measurement[str] so the CLI can show the source
    of each runtime detail.  Keys are plugin-defined — they are not part
    of the schema.
    """

    model_config = ConfigDict(frozen=True)

    runtime_kind: RuntimeKind = RuntimeKind.UNKNOWN
    version: Measurement[str] = Field(
        default_factory=lambda: Measurement.unavailable("Not collected"),
    )
    details: dict[str, Measurement[str]] = Field(default_factory=dict)
