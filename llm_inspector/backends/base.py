"""HardwareBackend ABC — the contract all backend implementations satisfy."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from llm_inspector.models.enums import BackendKind


@dataclass(frozen=True)
class DeviceInfo:
    """Snapshot of one hardware accelerator."""

    index: int
    name: str
    vram_total_bytes: int
    vram_used_bytes: int
    gpu_utilization_pct: int
    driver_version: str | None = None
    cuda_version: str | None = None


@dataclass(frozen=True)
class DeviceProcess:
    """A process using a hardware device."""

    pid: int
    device_index: int
    vram_used_bytes: int


class HardwareBackend(ABC):
    """
    Abstract hardware backend.

    Implementations provide device enumeration and per-process VRAM data.
    The Inspector selects the appropriate backend via BackendRegistry.

    All methods return empty collections rather than raising when the
    hardware is unavailable — callers must handle empty results gracefully.
    """

    kind: BackendKind

    @abstractmethod
    def is_available(self) -> bool:
        """Return True if this backend can be used on the current machine."""

    @abstractmethod
    def list_devices(self) -> list[DeviceInfo]:
        """Return a snapshot of all accessible accelerator devices."""

    @abstractmethod
    def device_processes(self, device_index: int) -> list[DeviceProcess]:
        """Return all processes currently using *device_index*."""

    def pid_to_devices(self) -> dict[int, list[DeviceProcess]]:
        """
        Build a mapping of PID → list[DeviceProcess] across all devices.

        A single PID may appear on multiple devices (tensor-parallel).
        """
        result: dict[int, list[DeviceProcess]] = {}
        for device in self.list_devices():
            for proc in self.device_processes(device.index):
                result.setdefault(proc.pid, []).append(proc)
        return result

    @abstractmethod
    def driver_version(self) -> str | None:
        """Return the driver version string, or None if unavailable."""

    @abstractmethod
    def runtime_version(self) -> str | None:
        """Return the compute runtime version (CUDA, ROCm…), or None."""
