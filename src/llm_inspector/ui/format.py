"""
Shared formatting helpers for the terminal UI.

All functions are pure — they take data and return strings.
No Rich imports here; these are used by both tables and panels.
"""

from __future__ import annotations

from llm_inspector.models.measurement import Measurement


def fmt_bytes(b: int | None) -> str:
    """Format bytes as a human-readable string (GB / MB / KB)."""
    if b is None:
        return "Unavailable"
    if b >= 1024**3:
        return f"{b / 1024**3:.1f} GB"
    if b >= 1024**2:
        return f"{b / 1024**2:.1f} MB"
    if b >= 1024:
        return f"{b / 1024:.1f} KB"
    return f"{b} B"


def fmt_measurement_bytes(m: Measurement[int]) -> str:
    """Format a measured Measurement[int] (bytes) or 'Unavailable'."""
    if not m.is_available or m.value is None:
        return "Unavailable"
    return fmt_bytes(m.value)


def fmt_projected_bytes(m: Measurement[int]) -> str:
    """Format measured or simulated bytes for Optimization Analysis."""
    if not m.has_value or m.value is None:
        return "Unavailable"
    return fmt_bytes(m.value)


def fmt_measurement_str(m: Measurement[str]) -> str:
    """Format a Measurement[str] as its value or 'Unavailable'."""
    if not m.is_available or m.value is None:
        return "Unavailable"
    return m.value


def fmt_measurement_int(m: Measurement[int], suffix: str = "") -> str:
    """Format a Measurement[int] as an integer string or 'Unavailable'."""
    if not m.is_available or m.value is None:
        return "Unavailable"
    val = f"{m.value:,}"
    return f"{val}{suffix}" if suffix else val


def fmt_params(count: int | None) -> str:
    """Format a parameter count as 'XB', 'XM', etc."""
    if count is None:
        return "Unavailable"
    if count >= 1e12:
        return f"{count / 1e12:.0f} Trillion"
    if count >= 1e9:
        return f"{count / 1e9:.0f} Billion"
    if count >= 1e6:
        return f"{count / 1e6:.0f} Million"
    return str(count)


def fmt_uptime(seconds: float | None) -> str:
    """Format uptime seconds as 'Xh Ym' or 'Xs'."""
    if seconds is None:
        return "unknown"
    s = int(seconds)
    h, rem = divmod(s, 3600)
    m = rem // 60
    if h:
        return f"{h}h {m}m"
    if m:
        return f"{m}m"
    return f"{s}s"


def truncate(s: str, max_len: int = 60) -> str:
    """Truncate a string with an ellipsis if longer than max_len."""
    if len(s) <= max_len:
        return s
    return s[: max_len - 1] + "…"
