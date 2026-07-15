"""
Streaming metrics — continuous counters updated during inference.

These are READ on inspect, not recomputed.  Hooks / monitors only do
atomic max()/+= with near-zero overhead.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class StreamingMetrics:
    """
    Prometheus-style counters for metrics that only exist during inference.

    Updated by lightweight hooks / the memory monitor; read by embedded
    collectors on RPC request.
    """

    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    longest_decode_ms: float = 0.0
    largest_kv_cache_bytes: int = 0

    # Phase 4 — activation tracking
    # baseline_allocated_bytes: GPU allocated at attach (weights + KV steady state)
    # peak_activation_bytes: max(allocated - baseline) observed since attach
    baseline_allocated_bytes: int | None = None
    peak_activation_bytes: int = 0
    activation_hooks_enabled: bool = False

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def set_baseline_allocated(self, nbytes: int) -> None:
        with self._lock:
            if self.baseline_allocated_bytes is None:
                self.baseline_allocated_bytes = max(0, int(nbytes))

    def record_allocated(self, nbytes: int) -> None:
        nbytes = int(nbytes)
        with self._lock:
            if nbytes > self.peak_allocated_bytes:
                self.peak_allocated_bytes = nbytes
            if self.baseline_allocated_bytes is not None:
                delta = nbytes - self.baseline_allocated_bytes
                if delta > self.peak_activation_bytes:
                    self.peak_activation_bytes = delta

    def record_reserved(self, nbytes: int) -> None:
        with self._lock:
            if nbytes > self.peak_reserved_bytes:
                self.peak_reserved_bytes = nbytes

    def record_decode_ms(self, ms: float) -> None:
        with self._lock:
            if ms > self.longest_decode_ms:
                self.longest_decode_ms = ms

    def record_kv_cache_bytes(self, nbytes: int) -> None:
        with self._lock:
            if nbytes > self.largest_kv_cache_bytes:
                self.largest_kv_cache_bytes = nbytes

    def record_activation_bytes(self, nbytes: int) -> None:
        """Record a direct activation-size observation (forward hooks)."""
        with self._lock:
            if nbytes > self.peak_activation_bytes:
                self.peak_activation_bytes = int(nbytes)

    def current_activation_bytes(self, allocated: int) -> int:
        """Transient activations right now vs attach baseline."""
        with self._lock:
            if self.baseline_allocated_bytes is None:
                return 0
            return max(0, int(allocated) - self.baseline_allocated_bytes)

    def snapshot(self) -> dict[str, int | float | bool | None]:
        with self._lock:
            return {
                "peak_allocated_bytes": self.peak_allocated_bytes,
                "peak_reserved_bytes": self.peak_reserved_bytes,
                "longest_decode_ms": self.longest_decode_ms,
                "largest_kv_cache_bytes": self.largest_kv_cache_bytes,
                "baseline_allocated_bytes": self.baseline_allocated_bytes,
                "peak_activation_bytes": self.peak_activation_bytes,
                "activation_hooks_enabled": self.activation_hooks_enabled,
            }
