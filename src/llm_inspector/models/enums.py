"""Shared enumeration types used across models, collectors, and plugins."""

from __future__ import annotations

from enum import StrEnum


class RuntimeKind(StrEnum):
    """Known LLM inference runtimes, detected by heuristic cmdline analysis."""

    VLLM = "vLLM"
    HUGGING_FACE = "HuggingFace"
    OLLAMA = "Ollama"
    TENSORRT = "TensorRT-LLM"
    SGLANG = "SGLang"
    LLAMA_CPP = "llama.cpp"
    FASTAPI = "FastAPI"
    UNKNOWN = "Unknown"


class BackendKind(StrEnum):
    """Hardware acceleration backend."""

    CUDA = "CUDA"
    ROCM = "ROCm"
    METAL = "Metal"
    CPU = "CPU"


class CollectorPhase(StrEnum):
    """
    Pipeline phase for a collector.

    Phase A collectors run sequentially before InspectionContext is built.
    Phase B collectors run (potentially in parallel) with the full context.
    """

    A = "A"
    B = "B"


class CollectorStatus(StrEnum):
    """Outcome of a collector.collect() call."""

    SUCCESS = "success"
    PARTIAL = "partial"   # data returned but some fields missing
    FAILED = "failed"     # no data returned; error message set
