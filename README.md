<div align="center">

# LLM Inspector

**The `htop` for LLM inference.**

*Measured. Not guessed.*

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20DGX-lightgrey)](#)

</div>

---

Most LLM memory tools ask you to fill in a form.
LLM Inspector asks for nothing.

It reads what is **actually running on your machine** — live process state, runtime APIs, GPU memory — and tells you exactly what is happening and where every number came from.

```
Weights       5.3 GB    [Ollama /api/ps → size_vram]
Process RAM   84.3 MB   [psutil.Process.memory_info().rss]
Version       0.32.0    [Ollama /api/version]
KV Cache      Unavailable  (Ollama does not expose KV cache allocation.)
```

Every value is either measured from a live source or explicitly marked `Unavailable` with a reason. **There is no third option.**

---

## Install

```bash
git clone https://github.com/helasaoudi/llm-inspector
cd llm-inspector
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Try it

```bash
# List all detected LLM processes
llminspect ps

# Inspect a process
llminspect inspect <pid>

# See exactly where every number came from
llminspect inspect <pid> --verbose
```

With Ollama running:

```bash
ollama run llama3

# in another terminal
llminspect inspect $(pgrep -f "ollama serve") --verbose
```

---

## Supports

Ollama · vLLM · HuggingFace Transformers · macOS · Linux · NVIDIA DGX

---

## DGX / integration guide

How to wire LLM Inspector into Docker inference services on NVIDIA DGX (external `llminspect` + embedded `attach()`, including the vLLM `general_plugins` pattern):

**[docs/DGX_GUIDE.md](docs/DGX_GUIDE.md)** — LLM Inspector Integration Guide

---

## License

MIT
