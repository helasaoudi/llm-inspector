"""
BackendRegistry — auto-detects and selects the best available HardwareBackend.

Priority order: CUDA → ROCm → Metal → CPU.

The Inspector never instantiates a backend directly — it always goes
through the registry.  This keeps hardware detection in one place and
makes it trivially mockable in tests.
"""

from __future__ import annotations

from llm_inspector.backends.base import HardwareBackend
from llm_inspector.backends.cpu import CPUBackend
from llm_inspector.backends.cuda import CUDABackend


class BackendRegistry:
    """
    Discovers and ranks available hardware backends.

    Usage::

        registry = BackendRegistry()
        backend = registry.best_available()
    """

    def __init__(self, candidates: list[HardwareBackend] | None = None) -> None:
        # Default priority order.  Inject a custom list for testing.
        self._candidates: list[HardwareBackend] = candidates or [
            CUDABackend(),
            # ROCmBackend(),   # Phase 2
            # MetalBackend(),  # Phase 2
            CPUBackend(),
        ]

    def best_available(self) -> HardwareBackend:
        """
        Return the highest-priority available backend.

        Always returns something — CPUBackend is the guaranteed fallback.
        """
        for backend in self._candidates:
            if backend.is_available():
                return backend
        return CPUBackend()

    def all_available(self) -> list[HardwareBackend]:
        """Return all backends that are available on this machine."""
        return [b for b in self._candidates if b.is_available()]
