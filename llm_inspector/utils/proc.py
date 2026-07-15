"""
Process introspection helpers.

On Linux: reads /proc/<pid>/cmdline, /proc/<pid>/exe, /proc/<pid>/cwd.
On macOS/Windows: falls back to psutil so the tool can be developed locally.

All functions return None / [] on any error — callers must handle
missing data explicitly.
"""

from __future__ import annotations

import os
import platform
from datetime import datetime, timezone
from pathlib import Path

_IS_LINUX = platform.system() == "Linux"


def read_cmdline(pid: int) -> list[str]:
    """
    Read argv for *pid* as a list of strings.

    Returns an empty list if the process has exited or access is denied.
    """
    if _IS_LINUX:
        return _cmdline_proc(pid)
    return _cmdline_psutil(pid)


def read_exe(pid: int) -> str | None:
    """Return the resolved executable path for *pid*, or None."""
    if _IS_LINUX:
        try:
            return os.readlink(f"/proc/{pid}/exe")
        except OSError:
            pass
    return _psutil_attr(pid, "exe", call=True)


def read_name(pid: int) -> str | None:
    """Return the process name (comm) for *pid*, or None."""
    if _IS_LINUX:
        try:
            return Path(f"/proc/{pid}/comm").read_text().strip()
        except OSError:
            pass
    return _psutil_attr(pid, "name", call=True)


def read_ppid(pid: int) -> int | None:
    """Return the parent PID for *pid*, or None."""
    return _psutil_attr(pid, "ppid", call=True)  # psutil.Process.ppid() is a method


def read_start_time(pid: int) -> datetime | None:
    """Return the process start time as a UTC datetime, or None."""
    ts = _psutil_attr(pid, "create_time", call=True)  # psutil.Process.create_time() is a method
    if ts is None:
        return None
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


def read_uptime(pid: int) -> float | None:
    """Return seconds since the process started, or None."""
    start = read_start_time(pid)
    if start is None:
        return None
    return (datetime.utcnow() - start).total_seconds()


def list_all_pids() -> list[int]:
    """Return all PIDs currently visible on the machine."""
    try:
        import psutil  # noqa: PLC0415

        return [p.pid for p in psutil.process_iter(["pid"])]
    except Exception:  # noqa: BLE001
        return []


# ── Private helpers ───────────────────────────────────────────────────────────


def _cmdline_proc(pid: int) -> list[str]:
    try:
        raw = Path(f"/proc/{pid}/cmdline").read_bytes()
    except OSError:
        return []
    if not raw:
        return []
    parts = raw.rstrip(b"\x00").split(b"\x00")
    return [p.decode("utf-8", errors="replace") for p in parts]


def _cmdline_psutil(pid: int) -> list[str]:
    try:
        import psutil  # noqa: PLC0415

        return psutil.Process(pid).cmdline()
    except Exception:  # noqa: BLE001
        return []


def _psutil_attr(pid: int, attr: str, *, call: bool = False) -> object:
    try:
        import psutil  # noqa: PLC0415

        proc = psutil.Process(pid)
        val = getattr(proc, attr)
        return val() if call else val
    except Exception:  # noqa: BLE001
        return None
