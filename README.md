<div align="center">

# LLM Inspector

**The `htop` for LLM inference.**

*Measured. Not guessed.*

[![Python](https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white)](https://python.org)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platforms](https://img.shields.io/badge/platform-macOS%20%7C%20Linux%20%7C%20GPU%20servers-lightgrey)](#)

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

Works on **any NVIDIA GPU machine** — laptop, workstation, cloud VM, bare-metal server, or DGX. Docker is optional.

```bash
git clone https://github.com/helasaoudi/llm-inspector
cd llm-inspector
python -m venv .venv && source .venv/bin/activate
pip install -e ".[torch]"
```

Full guide (host + Docker + `attach()` + vLLM plugin):

**[docs/INSTALL_GUIDE.md](docs/INSTALL_GUIDE.md)**

## Try it

```bash
# List all detected LLM processes
llminspect ps

# Inspect a process
llminspect inspect <pid>

# See exactly where every number came from
llminspect inspect <pid> --verbose
```

With Ollama on the host (no Docker):

```bash
ollama run llama3

# in another terminal
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
| **[Install & Integration](docs/INSTALL_GUIDE.md)** | Install on a simple server or in Docker; external vs embedded |
| [DGX notes](docs/DGX_GUIDE.md) | Extra detail from a DGX Spark inference-service setup |

---

## License

MIT
