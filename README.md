<div align="center">

# LLM Inspector

**The `htop` for LLM inference.**

*Measured. Not guessed.*

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20GPU%20servers-lightgrey)](#)

**Inspect → Understand → Optimize**

</div>

---

LLM Inspector inspects live inference processes and shows exactly how GPU memory is being used, what model is running, how the runtime is configured, and where every reported value comes from.

Unlike traditional monitoring tools, it doesn't stop at inspection. It also analyzes the running workload and projects optimization opportunities—starting with quantization—to help you understand how different strategies would impact GPU memory **before** making any changes.

---

## Example

```text
═════════════════════════════ LLM INSPECTOR v0.6.0 ═════════════════════════════

Process ────────────────────────────────────────────────────────────────────────
PID                   3634048
Runtime               vLLM
Command               VLLM::EngineCore

Hardware ───────────────────────────────────────────────────────────────────────
GPU                   NVIDIA GeForce RTX 3080 Ti
VRAM Total            12.0 GB
VRAM Used             6.3 GB

Model ──────────────────────────────────────────────────────────────────────────
Name                  Benchmaxx-Llama-3.2-1B
Architecture          LlamaForCausalLM
Precision             torch.bfloat16
Parameters            1.2B
Context Length        2,048 tokens

Model Details ──────────────────────────────────────────────────────────────────
Vocab Size            128,256
Layers                16
Hidden Size           2,048
Attention Heads       32
KV Heads              8

Memory ─────────────────────────────────────────────────────────────────────────
GPU Used              6.3 GB
GPU Allocated         5.5 GB
GPU Reserved          5.8 GB
Peak GPU              5.7 GB

Memory Breakdown ───────────────────────────────────────────────────────────────
Weights               2.3 GB
KV Cache              2.9 GB
Activations           0 B
Workspace             353 MB
Other                 223 MB
────────────────────────────────────────────────────────────────────────────────
Total                 5.8 GB

Runtime Details ────────────────────────────────────────────────────────────────
Runtime               vLLM
PagedAttention        Enabled

Optimization Analysis (Projected) ──────────────────────────────────────────────
  Quantization
  Method              New Total       Saved
  FP8                 4.7 GB          1.1 GB
  INT8                4.7 GB          1.1 GB
  AWQ 4-bit           4.2 GB          1.6 GB
  GPTQ 4-bit          4.1 GB          1.7 GB

  Recommendation
  ✓ GPTQ 4-bit is the best trade-off (saves 1.7 GB, typical quality: Good).
  ⚠ Your largest memory consumer is KV Cache (50% of measured breakdown).
    Weight quantization alone may not remove the bottleneck.

────────────────────────────────────────────────────────────────────────────────
Measured. Not guessed.
```

---

## Why?

Existing tools tell you that your GPU is using 18 GB.

LLM Inspector tells you **why**:

```text
Weights       7 GB
KV Cache      8 GB
Workspace     1 GB
Other         2 GB
```

Then it tells you what would happen if you optimized:

```text
FP8 would save ~3 GB of weights.
AWQ would save ~5 GB of weights.
Weight quantization won't fix an 8 GB KV Cache bottleneck.
```

That is the difference between a GPU monitor and an inference advisor.

---

## Embedded goes deeper

Most tools stop here:

```text
GPU
└── Process A
    └── 17.3 GB
```

LLM Inspector goes one level deeper:

```text
GPU
└── Process A
    ├── Weights
    ├── KV Cache
    ├── Workspace
    ├── Activations
    └── Optimization Analysis (Projected)
```

That bridge—**system observability + model internals**, with an `htop`-style CLI—is the innovation. Deep metrics come from optional embedded `attach()` inside the inference process (see the install guide).

---

## Measured vs Projected

| Section | Kind |
|---------|------|
| Process, Hardware, Model, Memory, Runtime | **Measured** from live sources (or `Unavailable` with a reason) |
| Optimization Analysis | **Projected** from measured inputs — never mutates the model |

```bash
llminspect inspect <pid> --verbose   # show provenance for every field
```

---

## Install

Works on any NVIDIA GPU machine — laptop, workstation, cloud VM, bare-metal server, or DGX. Docker is optional.

```bash
git clone https://github.com/helasaoudi/llm-inspector
cd llm-inspector
python -m venv .venv && source .venv/bin/activate
pip install -e ".[torch]"
```

---

## Try it

```bash
llminspect ps
llminspect inspect <pid>
llminspect inspect <pid> --verbose
```

With Ollama on the host (no Docker):

```bash
ollama run llama3
llminspect inspect $(pgrep -f "ollama serve") --verbose
```

---

## Supports

Ollama · vLLM · HuggingFace Transformers · FastAPI · custom PyTorch  
macOS · Linux · any NVIDIA GPU server (including DGX)

---

## Documentation

| Guide | When to read it |
|-------|-----------------|
| **[Install & Integration](docs/INSTALL_GUIDE.md)** | Host + Docker install, `attach()`, vLLM plugin, troubleshooting |
| [DGX notes](docs/DGX_GUIDE.md) | Extra detail from a DGX Spark inference-service setup |

---

## License

MIT
