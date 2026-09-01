"""Discover vLLM multiprocess TP worker groups (EngineCore + Worker_TP*).

vLLM tensor-parallel jobs often use one process per GPU rank
(``VLLM::Worker_TP0``, ``VLLM::Worker_TP1``, …), not one PID on many GPUs.
``llminspect inspect <any-member-pid>`` should still report the whole job.
"""

from __future__ import annotations

import re

from llm_inspector.utils import proc as proc_utils

_WORKER_TP_RE = re.compile(r"VLLM::Worker_TP(\d+)", re.IGNORECASE)
_ENGINE_CORE_RE = re.compile(r"VLLM::EngineCore", re.IGNORECASE)


def _identity_text(pid: int) -> str:
    parts = proc_utils.read_cmdline(pid)
    name = proc_utils.read_name(pid)
    if name:
        parts = [*parts, name]
    return " ".join(parts)


def is_vllm_tp_member(pid: int) -> bool:
    """True if *pid* looks like a vLLM EngineCore or Worker_TP* process."""
    text = _identity_text(pid)
    if not text.strip():
        return False
    return bool(_WORKER_TP_RE.search(text) or _ENGINE_CORE_RE.search(text))


def tp_rank(pid: int) -> int:
    """
    Sort key: Worker_TP{N} → N; EngineCore → -1 (listed first); other → 10_000.
    """
    text = _identity_text(pid)
    m = _WORKER_TP_RE.search(text)
    if m:
        return int(m.group(1))
    if _ENGINE_CORE_RE.search(text):
        return -1
    return 10_000


def worker_label(pid: int) -> str:
    """Human label for a group member."""
    text = _identity_text(pid)
    m = _WORKER_TP_RE.search(text)
    if m:
        return f"Worker_TP{m.group(1)}"
    if _ENGINE_CORE_RE.search(text):
        return "EngineCore"
    name = proc_utils.read_name(pid)
    return name or f"pid:{pid}"


def discover_vllm_tp_group(pid: int) -> list[int]:
    """
    Return PIDs that belong to the same vLLM TP job as *pid*.

    Grouping rule (measured via /proc + process names only):
      - Members are EngineCore / Worker_TP* processes
      - Share the same parent PID, or are parent/child of each other

    If *pid* is not a TP member, returns ``[pid]`` only.
    """
    if not is_vllm_tp_member(pid):
        return [pid]

    ppid = proc_utils.read_ppid(pid)
    members: set[int] = {pid}

    for other in proc_utils.list_all_pids():
        if other == pid or not is_vllm_tp_member(other):
            continue
        other_ppid = proc_utils.read_ppid(other)
        # Same parent (typical: API server → Worker_TP0, Worker_TP1, EngineCore)
        if ppid is not None and other_ppid == ppid:
            members.add(other)
            continue
        # Parent/child link (EngineCore ↔ workers)
        if other_ppid == pid or (ppid is not None and other == ppid):
            members.add(other)

    return sorted(members, key=tp_rank)
