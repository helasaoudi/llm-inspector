"""
ProcFS inspector — Linux /proc/<pid> data extraction.

Reads OS-level information from the /proc filesystem to inspect
any running process without requiring its cooperation.

This is the "don't ask, observe" layer:

  /proc/<pid>/maps     → memory-mapped files (model weights, libraries)
  /proc/<pid>/environ  → environment variables (HF_HOME, CUDA_VISIBLE_DEVICES)
  /proc/<pid>/fd       → open file descriptors

All functions return empty collections on permission errors or
when /proc is unavailable (macOS). Callers must handle that gracefully.

Linux only — macOS has no /proc. Functions are safe to call on macOS
but return empty results.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

# ── Known model weight extensions ────────────────────────────────────────────

_MODEL_WEIGHT_SUFFIXES: frozenset[str] = frozenset({
    ".safetensors",
    ".bin",         # PyTorch legacy
    ".pt",
    ".pth",
    ".gguf",        # llama.cpp
    ".ggml",        # llama.cpp legacy
    ".pkl",
    ".ot",          # OpenVINO
})

# Regex to extract model name from a HuggingFace cache path.
# HF stores models as:  .../hub/models--{org}--{name}/snapshots/{hash}/{file}
#                   or:  .../hub/models--{org}--{name}/blobs/{hash}
_HF_MODEL_RE = re.compile(
    r"models--([A-Za-z0-9_\-\.]+)--([A-Za-z0-9_\-\.]+)"
)

# Known inference frameworks detectable from loaded .so paths
_FRAMEWORK_SIGNATURES: list[tuple[str, str]] = [
    ("vllm",         "vLLM"),
    ("torch",        "PyTorch"),
    ("ggml",         "llama.cpp"),
    ("gguf",         "llama.cpp"),
    ("tensorrt",     "TensorRT"),
    ("tritonserver", "Triton"),
]

# Environment variable names that carry model identity, in priority order.
# Many inference servers set one of these when they load a model.
_MODEL_ENV_VARS: tuple[str, ...] = (
    "SERVED_MODEL_NAME",   # vLLM
    "MODEL_ID",
    "MODEL_NAME",
    "HF_MODEL_ID",
    "HF_MODEL",
    "MODEL",
    "WHISPER_MODEL",       # Whisper STT servers
    "LLM_MODEL",
    "EMBEDDING_MODEL",
)


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MappedFile:
    """A single memory-mapped file entry from /proc/<pid>/maps."""

    path: str
    size_bytes: int      # mapped region size (may span multiple lines for same file)
    is_model_weight: bool
    is_library: bool


@dataclass
class ProcFSSnapshot:
    """All /proc-derived data for a single process."""

    pid: int
    mapped_files: list[MappedFile] = field(default_factory=list)
    environ: dict[str, str] = field(default_factory=dict)
    open_files: list[str] = field(default_factory=list)
    status: dict[str, str] = field(default_factory=dict)

    # Whether /proc/<pid>/maps was readable (False = permission denied or absent)
    maps_readable: bool = False
    # Whether /proc/<pid>/fd was readable
    fd_readable: bool = False
    # Whether /proc/<pid>/environ was readable
    environ_readable: bool = False

    # ── Derived properties ─────────────────────────────────────────────────

    @property
    def model_weight_files(self) -> list[MappedFile]:
        """All mapped model weight files."""
        return [f for f in self.mapped_files if f.is_model_weight]

    @property
    def total_weights_bytes(self) -> int:
        """Sum of all mapped model weight file sizes."""
        # Count each unique path only once (may appear in multiple map entries)
        seen: set[str] = set()
        total = 0
        for f in self.model_weight_files:
            if f.path not in seen:
                seen.add(f.path)
                total += f.size_bytes
        return total

    @property
    def hf_model_name(self) -> str | None:
        """
        Extract 'org/model' from HuggingFace cache paths.

        Returns None if no HF cache path is found.
        """
        for f in self.model_weight_files:
            m = _HF_MODEL_RE.search(f.path)
            if m:
                org, name = m.group(1), m.group(2)
                return f"{org}/{name}"
        # Also check open file descriptors
        for path in self.open_files:
            m = _HF_MODEL_RE.search(path)
            if m:
                org, name = m.group(1), m.group(2)
                return f"{org}/{name}"
        # Check HF_HOME env var for cache path pattern
        hf_home = self.environ.get("HF_HOME") or self.environ.get("TRANSFORMERS_CACHE")
        if hf_home:
            m = _HF_MODEL_RE.search(hf_home)
            if m:
                org, name = m.group(1), m.group(2)
                return f"{org}/{name}"
        return None

    @property
    def detected_frameworks(self) -> list[str]:
        """Frameworks detected from loaded shared libraries."""
        found: list[str] = []
        seen: set[str] = set()
        all_paths = [f.path for f in self.mapped_files] + self.open_files
        combined = " ".join(all_paths).lower()
        for sig, name in _FRAMEWORK_SIGNATURES:
            if sig in combined and name not in seen:
                found.append(name)
                seen.add(name)
        return found

    @property
    def cuda_visible_devices(self) -> str | None:
        """CUDA_VISIBLE_DEVICES environment variable, if set."""
        return self.environ.get("CUDA_VISIBLE_DEVICES")

    @property
    def model_from_env(self) -> tuple[str, str] | None:
        """
        Extract a model name from environment variables.

        Returns (model_name, env_var_name) for provenance, or None.

        Checks a curated priority list first, then falls back to any
        variable ending in ``_MODEL`` whose value is not a filesystem path
        (paths are handled by /proc/maps weight-file detection instead).
        """
        for var in _MODEL_ENV_VARS:
            val = self.environ.get(var)
            if val and not val.startswith("/"):
                return val, var
        # Heuristic: any *_MODEL variable holding a non-path value
        for key, val in self.environ.items():
            if key.endswith("_MODEL") and val and not val.startswith("/"):
                return val, key
        return None

    @property
    def pytorch_version(self) -> str | None:
        """PyTorch version from the PYTORCH_VERSION env var, if present."""
        return self.environ.get("PYTORCH_VERSION")

    @property
    def hf_model_path(self) -> str | None:
        """Full path of the first model weight file found in /proc maps."""
        for f in self.model_weight_files:
            return f.path
        return None


# ── Public API ────────────────────────────────────────────────────────────────

def snapshot(pid: int) -> ProcFSSnapshot:
    """
    Build a full ProcFSSnapshot for *pid*.

    Safe to call on any OS — returns an empty snapshot on macOS or
    when the process is inaccessible.

    Tracks readability flags (maps_readable, fd_readable, environ_readable)
    so callers can distinguish "empty because permission denied" from
    "empty because no model files found".
    """
    result = ProcFSSnapshot(pid=pid)
    base = Path(f"/proc/{pid}")

    if not base.exists():
        return result  # macOS or inaccessible

    maps, maps_ok = _read_maps(base / "maps")
    result.mapped_files = maps
    result.maps_readable = maps_ok

    result.status = _read_status(base / "status")

    environ, environ_ok = _read_environ(base / "environ")
    result.environ = environ
    result.environ_readable = environ_ok

    open_files, fd_ok = _read_fd_paths(base / "fd")
    result.open_files = open_files
    result.fd_readable = fd_ok

    return result


# ── Private helpers ───────────────────────────────────────────────────────────

def _read_maps(maps_path: Path) -> tuple[list[MappedFile], bool]:
    """
    Parse /proc/<pid>/maps.

    Returns (list_of_mapped_files, was_readable).
    Format:  address perms offset dev inode [pathname]
    """
    if not maps_path.exists():
        return [], False

    size_by_path: dict[str, int] = {}

    try:
        text = maps_path.read_text(errors="replace")
    except PermissionError:
        return [], False
    except OSError:
        return [], False

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        path = parts[5]
        if not path or path.startswith("["):
            continue

        addr_range = parts[0]
        try:
            start_str, end_str = addr_range.split("-")
            size = int(end_str, 16) - int(start_str, 16)
        except ValueError:
            size = 0

        size_by_path[path] = size_by_path.get(path, 0) + size

    result: list[MappedFile] = []
    for path, size in size_by_path.items():
        ext = os.path.splitext(path)[1].lower()
        is_weight = ext in _MODEL_WEIGHT_SUFFIXES
        is_lib = path.endswith(".so") or ".so." in path
        result.append(MappedFile(
            path=path,
            size_bytes=size,
            is_model_weight=is_weight,
            is_library=is_lib,
        ))

    return result, True


def _read_status(status_path: Path) -> dict[str, str]:
    """
    Parse /proc/<pid>/status into a key→value dict.

    Always readable (kernel allows it for any PID).
    Useful fields: Name, Pid, VmRSS, VmPeak, VmSize, Threads.
    """
    if not status_path.exists():
        return {}
    try:
        text = status_path.read_text(errors="replace")
    except OSError:
        return {}
    result: dict[str, str] = {}
    for line in text.splitlines():
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


def _read_environ(environ_path: Path) -> tuple[dict[str, str], bool]:
    """Parse /proc/<pid>/environ. Returns (dict, was_readable)."""
    if not environ_path.exists():
        return {}, False
    try:
        raw = environ_path.read_bytes()
    except PermissionError:
        return {}, False
    except OSError:
        return {}, False

    env: dict[str, str] = {}
    for entry in raw.split(b"\x00"):
        if b"=" in entry:
            key, _, val = entry.partition(b"=")
            try:
                env[key.decode(errors="replace")] = val.decode(errors="replace")
            except Exception:  # noqa: BLE001
                pass
    return env, True


def _read_fd_paths(fd_dir: Path) -> tuple[list[str], bool]:
    """
    Read symlinks in /proc/<pid>/fd. Returns (paths, was_readable).

    Ignores sockets, pipes, and other special fds.
    """
    if not fd_dir.exists():
        return [], False
    paths: list[str] = []
    try:
        for entry in fd_dir.iterdir():
            try:
                target = os.readlink(entry)
                if target.startswith("/") and not target.startswith("/dev/"):
                    paths.append(target)
            except OSError:
                pass
    except PermissionError:
        return [], False
    except OSError:
        return [], False
    return paths, True
