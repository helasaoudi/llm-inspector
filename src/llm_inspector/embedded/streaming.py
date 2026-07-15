"""
Streaming metrics — continuous counters updated during inference.

These are READ on inspect, not recomputed.  Hooks only do atomic max()/+=
with near-zero overhead.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class StreamingMetrics:
    """
    Prometheus-style counters for metrics that only exist during inference.

    Updated by lightweight hooks; read by embedded collectors on RPC request.
    """

    peak_allocated_bytes: int = 0
    peak_reserved_bytes: int = 0
    longest_decode_ms: float = 0.0
    largest_kv_cache_bytes: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record_allocated(self, nbytes: int) -> None:
        with self._lock:
            if nbytes > self.peak_allocated_bytes:
                self.peak_allocated_bytes = nbytes

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

    def snapshot(self) -> dict[str, int | float]:
        with self._lock:
            return {
                "peak_allocated_bytes": self.peak_allocated_bytes,
                "peak_reserved_bytes": self.peak_reserved_bytes,
                "longest_decode_ms": self.longest_decode_ms,
                "largest_kv_cache_bytes": self.largest_kv_cache_bytes,
            }
