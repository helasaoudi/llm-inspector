"""Socket path discovery for embedded inspectors."""

from __future__ import annotations

import os
from pathlib import Path

SOCKET_ENV_VAR = "LLM_INSPECTOR_SOCK"
_RUN_DIR = Path("/run/llminspect")
_TMP_DIR = Path("/tmp/llminspect")


def socket_env_var() -> str:
    return SOCKET_ENV_VAR


def default_socket_path(pid: int) -> Path:
    """
    Preferred socket path for *pid*.

    Tries /run/llminspect first (requires writable dir), falls back to /tmp.
    """
    for base in (_RUN_DIR, _TMP_DIR):
        try:
            base.mkdir(parents=True, exist_ok=True)
            if os.access(base, os.W_OK):
                return base / f"{pid}.sock"
        except OSError:
            continue
    return _TMP_DIR / f"{pid}.sock"


def discover_socket_path(pid: int, environ: dict[str, str] | None = None) -> Path | None:
    """
    Find the Unix socket for an embedded inspector on *pid*.

    Checks, in order:
      1. LLM_INSPECTOR_SOCK in process environ
      2. Default paths (/run/llminspect/<pid>.sock, /tmp/llminspect/<pid>.sock)
    """
    if environ:
        sock = environ.get(SOCKET_ENV_VAR)
        if sock and Path(sock).exists():
            return Path(sock)

    for path in (default_socket_path(pid), _TMP_DIR / f"{pid}.sock", _RUN_DIR / f"{pid}.sock"):
        if path.exists():
            return path

    return None
