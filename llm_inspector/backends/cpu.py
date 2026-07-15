"""
CPUBackend — always-available fallback when no GPU backend is detected.

Returns process RAM data via psutil.  GPU-specific fields return empty
results rather than raising.  This backend enables the tool to run on
any machine for development and testing.
"""

from __future__ import annotations

import contextlib

from llm_inspector.backends.base import DeviceInfo, DeviceProcess, HardwareBackend
from llm_inspector.models.enums import BackendKind


class CPUBackend(HardwareBackend):
    """CPU-only fallback backend. Always available."""

    kind = BackendKind.CPU

    def is_available(self) -> bool:
        return True

    def list_devices(self) -> list[DeviceInfo]:
        return []

    def device_processes(self, device_index: int) -> list[DeviceProcess]:
        return []

    def driver_version(self) -> str | None:
        return None

    def runtime_version(self) -> str | None:
        return None
