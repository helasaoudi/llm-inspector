"""
Command-line argument parsing helpers.

These functions parse argv lists in the style common to Python inference
servers (``--model meta-llama/Llama-3-8B``, ``--port=8000``, etc.).

All functions return None on any parse failure — callers must handle
missing arguments explicitly.
"""

from __future__ import annotations


def parse_arg(cmdline: list[str], *flags: str) -> str | None:
    """
    Extract the value for any of the given flags from an argv list.

    Handles both forms:
      ``--flag value``
      ``--flag=value``

    Args:
        cmdline: The process argv list.
        *flags:  One or more flag names to try, e.g. ``"--model"``, ``"-m"``.

    Returns:
        The value string, or None if none of the flags were found.
    """
    for i, token in enumerate(cmdline):
        for flag in flags:
            if token == flag:
                if i + 1 < len(cmdline):
                    return cmdline[i + 1]
            elif token.startswith(f"{flag}="):
                return token.split("=", 1)[1]
    return None


def parse_int_arg(cmdline: list[str], *flags: str, default: int | None = None) -> int | None:
    """Parse an integer argument from cmdline, returning *default* on failure."""
    raw = parse_arg(cmdline, *flags)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def flag_present(cmdline: list[str], *flags: str) -> bool:
    """Return True if any of the flags appear as a bare token in cmdline."""
    return any(token in flags for token in cmdline)


def detect_port(cmdline: list[str], default: int = 8000) -> int:
    """
    Extract the server port from cmdline, falling back to *default*.

    Checks ``--port`` and ``-p``.
    """
    return parse_int_arg(cmdline, "--port", "-p", default=default) or default


def detect_host(cmdline: list[str], default: str = "127.0.0.1") -> str:
    """
    Extract the server host from cmdline, falling back to *default*.

    Checks ``--host``.
    """
    return parse_arg(cmdline, "--host") or default
