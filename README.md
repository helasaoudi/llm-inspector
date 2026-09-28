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

Unlike traditional monitoring tools, it doesn't stop at inspection. It also analyzes the running workload and projects optimization opportunities starting with quantization to help you understand how different strategies would impact GPU memory **before** making any changes.

---

## Example

<div align="center">

![LLM Inspector demo](https://raw.githubusercontent.com/helasaoudi/llm-inspector/main/docs/assets/llm-inspector-demo.gif)

</div>

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
pip install llm-inspector
```

For embedded deep metrics (Weights / KV / Activations) in the same Python env as the model:

```bash
pip install "llm-inspector[torch]"
```

From source (contributors):

```bash
git clone https://github.com/helasaoudi/llm-inspector
cd llm-inspector
python -m venv .venv && source .venv/bin/activate
pip install -e ".[torch]"
```

---

## Try it

```bash
llminspect gpu
llminspect ps
llminspect inspect <pid> --verbose
```

### Multi-GPU (vLLM tensor parallel)

```bash
# after: vllm serve <model> --tensor-parallel-size 2 --port 8000
llminspect gpu
llminspect ps
llminspect inspect $(pgrep -f 'VLLM::Worker_TP0' | head -1) --verbose
# inspects Worker_TP0 + Worker_TP1 (+ EngineCore) in one report
```

### Multi-core / CPU (e.g. Ollama on macOS)

```bash
ollama run llama3
llminspect inspect $(pgrep -f "ollama serve" | head -1) --verbose
# Hardware shows CPU Cores / System RAM — not GPU ids
```

Multi-GPU ≠ multi-core. Details: [Install guide](docs/INSTALL_GUIDE.md#multi-gpu-vs-multi-core-not-the-same).

---

## Supports

Ollama · vLLM · HuggingFace Transformers · FastAPI · custom PyTorch  
macOS · Linux · any NVIDIA GPU server (including DGX)

---

## Documentation

| Guide | When to read it |
|-------|-----------------|
| **[Install & Integration](docs/INSTALL_GUIDE.md)** | PyPI install, multi-GPU / multi-core commands, Docker, `attach()`, troubleshooting |
| [DGX notes](docs/DGX_GUIDE.md) | DGX + Docker TP inspect examples |

---

## License

MIT
