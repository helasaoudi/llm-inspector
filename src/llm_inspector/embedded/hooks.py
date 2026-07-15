"""
Optional forward hooks for activation peak tracking (Phase 4).

Enabled via ``attach(..., trace_layers=True)``.  Hooks never raise into
the host forward path.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

from llm_inspector.embedded.streaming import StreamingMetrics

_log = logging.getLogger(__name__)

# Keep handles so detach / GC can remove them
_hook_handles: list[Any] = []


def _tensor_nbytes(obj: Any) -> int:
    """Best-effort byte size of a tensor / nested tensor structure."""
    if obj is None:
        return 0
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_nbytes(x) for x in obj)
    if isinstance(obj, dict):
        return sum(_tensor_nbytes(v) for v in obj.values())

    try:
        import torch  # noqa: PLC0415

        tensor_type = getattr(torch, "Tensor", None)
        if isinstance(tensor_type, type) and isinstance(obj, tensor_type):
            return int(obj.numel() * obj.element_size())
    except Exception:  # noqa: BLE001
        pass

    # Duck-typed tensors / HF ModelOutput
    if hasattr(obj, "numel") and hasattr(obj, "element_size"):
        try:
            return int(obj.numel() * obj.element_size())
        except Exception:  # noqa: BLE001
            return 0
    if hasattr(obj, "values") and callable(obj.values):
        try:
            return sum(_tensor_nbytes(v) for v in obj.values())
        except Exception:  # noqa: BLE001
            pass
    if hasattr(obj, "__dict__"):
        try:
            return sum(_tensor_nbytes(v) for v in vars(obj).values())
        except Exception:  # noqa: BLE001
            pass
    return 0


def install_activation_hooks(
    model: Any,
    streaming: StreamingMetrics,
) -> int:
    """
    Register forward hooks on *model* leaf modules.

    Returns the number of hooks installed.  Safe if torch/model missing.
    """
    clear_activation_hooks()

    try:
        import torch.nn as nn  # noqa: PLC0415
    except ImportError:
        _log.debug("torch not available — activation hooks skipped")
        return 0

    if model is None or not isinstance(model, nn.Module):
        _log.debug("No nn.Module for activation hooks")
        return 0

    def _make_hook() -> Callable[..., None]:
        def _hook(_module: Any, _inputs: Any, output: Any) -> None:
            try:
                nbytes = _tensor_nbytes(output)
                if nbytes > 0:
                    streaming.record_activation_bytes(nbytes)
                # Also capture allocator spike relative to baseline
                try:
                    import torch  # noqa: PLC0415

                    if torch.cuda.is_available():
                        streaming.record_allocated(torch.cuda.memory_allocated())
                except Exception:  # noqa: BLE001
                    pass
            except Exception:  # noqa: BLE001 — never break forward
                pass

        return _hook

    count = 0
    try:
        for module in model.modules():
            # Hook leaves only — cheaper than every container module
            if any(module.children()):
                continue
            handle = module.register_forward_hook(_make_hook())
            _hook_handles.append(handle)
            count += 1
    except Exception as exc:  # noqa: BLE001
        _log.warning("Failed to install activation hooks: %s", exc)
        clear_activation_hooks()
        return 0

    streaming.activation_hooks_enabled = count > 0
    _log.info("llm-inspector installed %d activation forward hooks", count)
    return count


def clear_activation_hooks() -> None:
    """Remove all hooks installed by this module."""
    global _hook_handles  # noqa: PLW0603
    for handle in _hook_handles:
        try:
            handle.remove()
        except Exception:  # noqa: BLE001
            pass
    _hook_handles = []


def resolve_hook_target(
    model: Any | None,
    engine: Any | None,
    adapter: Any | None = None,
) -> Any | None:
    """Pick an nn.Module to hook — model directly, or vLLM runner model."""
    if model is not None:
        return model
    if adapter is not None and engine is not None:
        resolve = getattr(adapter, "_resolve_model", None)
        if callable(resolve):
            try:
                return resolve(engine)
            except Exception:  # noqa: BLE001
                return None
    return None
