"""Resolve vLLM HTTP API base URL for API server and EngineCore worker PIDs."""

from __future__ import annotations

import re

from llm_inspector.utils.cmdline import detect_host, detect_port
from llm_inspector.utils.http import get_json

_ENGINE_CORE_RE = re.compile(r"vllm::enginecore", re.IGNORECASE)

# Ports to probe when worker cmdline has no --port (API server is sibling process).
# Include common non-default ports (e.g. 8011) so a live :8000 for another job
# is not the only candidate — sibling --port lookup still runs first.
_DEFAULT_PROBE_PORTS = (8000, 8001, 8010, 8011, 8080, 8888, 9000)


def is_engine_core_process(cmdline: list[str]) -> bool:
    """Return True if this PID is a vLLM GPU worker (EngineCore)."""
    joined = " ".join(cmdline)
    return bool(_ENGINE_CORE_RE.search(joined))


def _port_from_related_processes(pid: int) -> tuple[int, str] | None:
    """
    Find --port on parent / sibling / child processes of *pid*.

    EngineCore and Worker_TP* often omit --port; the OpenAI API server that
    launched them usually has it. Prefer that over probing localhost for any
    live /v1/models (which may belong to a different job).
    """
    from llm_inspector.utils import proc as proc_utils  # noqa: PLC0415

    ppid = proc_utils.read_ppid(pid)
    related: list[tuple[int, str]] = []

    if ppid is not None:
        related.append((ppid, "parent"))

    for other in proc_utils.list_all_pids():
        if other == pid:
            continue
        other_ppid = proc_utils.read_ppid(other)
        if ppid is not None and other_ppid == ppid:
            related.append((other, "sibling"))
        elif other_ppid == pid:
            related.append((other, "child"))

    for other, relation in related:
        cmdline = proc_utils.read_cmdline(other)
        if not cmdline:
            continue
        port = detect_port(cmdline, default=0)
        if port:
            return port, f"{relation} pid {other} cmdline --port → {port}"
    return None


def resolve_vllm_base_url(
    cmdline: list[str],
    environ: dict[str, str] | None = None,
    *,
    pid: int | None = None,
) -> tuple[str, str]:
    """
    Resolve the vLLM OpenAI API base URL for *cmdline*.

    Returns (base_url, source_description) for provenance.

    Resolution order:
      1. --port / --host in cmdline (API server process)
      2. LLM_INSPECTOR_VLLM_PORT or VLLM_PORT in environ
      3. Parent / sibling / child process cmdline --port (when *pid* given)
      4. EngineCore / worker: probe common localhost ports until /v1/models responds
      5. Fallback: http://127.0.0.1:8000 (may be unreachable)
    """
    env = environ or {}
    host = detect_host(cmdline, "127.0.0.1")
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"

    port = detect_port(cmdline, default=0)
    if port:
        base = f"http://{host}:{port}"
        return base, f"cmdline --host / --port → {base}"

    for key in ("LLM_INSPECTOR_VLLM_PORT", "VLLM_PORT", "VLLM_API_PORT"):
        raw = env.get(key)
        if raw:
            try:
                p = int(raw)
                base = f"http://{host}:{p}"
                return base, f"environ ${key} → {base}"
            except ValueError:
                pass

    if pid is not None:
        found = _port_from_related_processes(pid)
        if found is not None:
            p, how = found
            base = f"http://{host}:{p}"
            return base, f"{how} → {base}"

    if is_engine_core_process(cmdline) or pid is not None:
        for p in _DEFAULT_PROBE_PORTS:
            base = f"http://{host}:{p}"
            if _api_reachable(base):
                return base, f"port probe → GET {base}/v1/models"

    base = f"http://{host}:8000"
    return base, f"default API URL → {base}"


def _api_reachable(base_url: str) -> bool:
    data = get_json(f"{base_url}/v1/models", timeout=0.5)
    return isinstance(data, dict) and "data" in data
