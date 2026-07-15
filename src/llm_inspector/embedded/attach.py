"""
attach() — one-line embedded inspector integration.

Usage (HuggingFace local inference):
    from llm_inspector import attach
    model = AutoModelForCausalLM.from_pretrained(...)
    attach(model=model)

Usage (vLLM — when engine object is accessible):
    from llm_inspector import attach
    attach(engine=llm_engine)

The process exposes a Unix socket; ``llminspect inspect <pid>`` fetches
deep metrics automatically via EmbeddedSource.
"""

from __future__ import annotations

import atexit
import logging
import threading
from typing import Any

from llm_inspector.embedded.adapters.registry import bind_context
from llm_inspector.embedded.collectors import (
    collect_memory_breakdown_embedded,
    collect_memory_embedded,
    collect_model_embedded,
)
from llm_inspector.embedded.hooks import (
    clear_activation_hooks,
    install_activation_hooks,
    resolve_hook_target,
)
from llm_inspector.embedded.streaming import StreamingMetrics
from llm_inspector.transport.unix import UnixSocketTransport

_log = logging.getLogger(__name__)

# Module-level singleton — one inspector per process
_state: _AttachState | None = None


class _AttachState:
    def __init__(
        self,
        transport: UnixSocketTransport,
        streaming: StreamingMetrics,
        adapter_ctx: tuple[Any, Any] | None,
        monitor_thread: threading.Thread | None,
    ) -> None:
        self.transport = transport
        self.streaming = streaming
        self.adapter_ctx = adapter_ctx
        self.monitor_thread = monitor_thread


def attach(
    model: Any | None = None,
    engine: Any | None = None,
    *,
    trace_layers: bool = False,
    trace_timing: bool = False,
    socket_path: str | None = None,
) -> None:
    """
    Attach embedded inspector to the current inference process.

    Args:
        model: PyTorch module or HuggingFace PreTrainedModel.
        engine: vLLM LLMEngine / AsyncLLMEngine / EngineCore.
        trace_layers: Opt-in forward hooks for activation peaks (off by default).
        trace_timing: Opt-in prefill/decode timing hooks (off by default).
        socket_path: Override Unix socket path (default: auto).

    Safe to call multiple times — subsequent calls are no-ops if already attached.
    Never raises into the host process.
    """
    global _state  # noqa: PLW0603

    if _state is not None:
        _log.debug("llm-inspector already attached")
        return

    if model is None and engine is None:
        _log.warning("attach() called with no model or engine — limited metrics only.")

    try:
        import os
        from pathlib import Path

        # Env opt-in for Docker / vLLM plugin without code changes
        if not trace_layers and os.environ.get(
            "LLM_INSPECTOR_TRACE_LAYERS", ""
        ).lower() in ("1", "true", "yes"):
            trace_layers = True
        if not trace_timing and os.environ.get(
            "LLM_INSPECTOR_TRACE_TIMING", ""
        ).lower() in ("1", "true", "yes"):
            trace_timing = True

        options: dict[str, Any] = {
            "trace_layers": trace_layers,
            "trace_timing": trace_timing,
        }
        adapter, ctx = bind_context(model=model, engine=engine, options=options)
        adapter_ctx = (adapter, ctx)

        streaming = StreamingMetrics()
        _seed_baseline(streaming)

        transport = UnixSocketTransport(
            path=Path(socket_path) if socket_path else None,
        )

        handlers = {
            "memory": lambda req: collect_memory_embedded(req, streaming),
            "model": lambda req: collect_model_embedded(req, adapter_ctx),
            "memory-breakdown": lambda req: collect_memory_breakdown_embedded(
                req, adapter_ctx, streaming
            ),
        }
        transport.serve(handlers)

        monitor = _start_memory_monitor(streaming) if _cuda_available() else None

        if trace_layers:
            target = resolve_hook_target(model, engine, adapter)
            n = install_activation_hooks(target, streaming)
            if n == 0:
                _log.info(
                    "trace_layers=True but no hookable nn.Module found "
                    "(baseline activation tracking still active)"
                )
        if trace_timing:
            _log.info("trace_timing=True — timing hooks not yet implemented")

        _state = _AttachState(
            transport=transport,
            streaming=streaming,
            adapter_ctx=adapter_ctx,
            monitor_thread=monitor,
        )
        atexit.register(detach)
        _log.info("llm-inspector attached (socket: %s)", transport.path)

    except Exception as exc:  # noqa: BLE001 — never crash host
        _log.error("llm-inspector attach failed: %s", exc)


def detach() -> None:
    """Stop embedded inspector and clean up socket."""
    global _state  # noqa: PLW0603
    if _state is None:
        return
    try:
        clear_activation_hooks()
        if _state.monitor_thread:
            stop = getattr(_state.monitor_thread, "_stop", None)
            if stop is not None:
                stop.set()
            _state.monitor_thread.join(timeout=1.0)
        _state.transport.stop()
    except Exception as exc:  # noqa: BLE001
        _log.debug("detach error: %s", exc)
    finally:
        _state = None


def auto_attach() -> None:
    """
    Attach via environment variable — zero code changes.

    Set ``LLM_INSPECTOR_ATTACH=1`` before starting the inference process.
    Does not bind a model; memory metrics only.
    """
    import os

    if os.environ.get("LLM_INSPECTOR_ATTACH", "").lower() in ("1", "true", "yes"):
        attach()


def is_attached() -> bool:
    return _state is not None


def _cuda_available() -> bool:
    try:
        import torch  # noqa: PLC0415

        return torch.cuda.is_available()
    except ImportError:
        return False


def _seed_baseline(streaming: StreamingMetrics) -> None:
    """Record attach-time allocated bytes as the activation baseline."""
    if not _cuda_available():
        return
    try:
        import torch  # noqa: PLC0415

        allocated = int(torch.cuda.memory_allocated())
        reserved = int(torch.cuda.memory_reserved())
        streaming.set_baseline_allocated(allocated)
        streaming.record_allocated(allocated)
        streaming.record_reserved(reserved)
    except Exception:  # noqa: BLE001
        pass


def _start_memory_monitor(streaming: StreamingMetrics) -> threading.Thread:
    """
    Background thread that samples torch CUDA memory for streaming peaks.

    Polls every 500ms — cheap, no inference hooks required.
    """
    stop = threading.Event()

    def _loop() -> None:
        try:
            import torch  # noqa: PLC0415
        except ImportError:
            return
        while not stop.is_set():
            try:
                if torch.cuda.is_available():
                    streaming.record_allocated(torch.cuda.memory_allocated())
                    streaming.record_reserved(torch.cuda.memory_reserved())
            except Exception:  # noqa: BLE001
                pass
            stop.wait(0.5)

    t = threading.Thread(target=_loop, name="llminspect-memory-monitor", daemon=True)
    t.start()
    t._stop = stop  # type: ignore[attr-defined]
    return t
