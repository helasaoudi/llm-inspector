# LLM Inspector — Install & Integration Guide

**For any machine with an NVIDIA GPU** — laptop, workstation, cloud VM, bare-metal server, or DGX.

Docker is **optional**. On a simple server running Ollama, vLLM, or a Python script directly on the host, install the CLI once and inspect — no containers required.

*Measured. Not guessed.* Every value is measured from a live source or marked `Unavailable` with a reason.

---

## Choose your path

| Your setup | What to do |
|------------|------------|
| **Quick test on a running inference service** | [Test on your inference service](#test-on-your-inference-service-from-pypi) |
| **Simple server / bare metal** (Ollama, vLLM, HF script on the host) | [A. Install on the host](#a-install-on-the-host-no-docker) → [External inspect](#3-external-inspection-zero-code-changes) |
| **Docker GPU services** | [B. Install inside the image](#b-install-in-docker) → inspect with `docker exec` |
| **Deep metrics** (Allocated / Reserved / Peak / Weights / KV) | Also [Embedded `attach()`](#4-embedded-integration-deep-metrics) |

---

## Test on your inference service (from PyPI)

Package is on PyPI: [`llm-inspector`](https://pypi.org/project/llm-inspector/). Pick the path that matches how your service runs.

### Path 1 — Inference on the host (Ollama, vLLM, Python script)

```bash
# 1. Install the CLI (any venv / user pip on the GPU machine)
pip install llm-inspector

# 2. Confirm it works
llminspect --help
llminspect gpu

# 3. List GPU inference processes
llminspect ps

# 4. Inspect one PID from the table (or from nvidia-smi)
llminspect inspect <pid>
llminspect inspect <pid> --verbose
```

That is enough for **external** metrics (GPU Used, process, runtime APIs).

For **deep** metrics (Allocated / Reserved / Peak / Weights / KV / Model Details / Optimization Analysis), install into the **same Python env** as the model and call `attach()` after load:

```bash
pip install "llm-inspector[torch]"
```

```python
from llm_inspector import attach
attach(model=model)          # HuggingFace / PyTorch
# or: attach(engine=llm.llm_engine)   # vLLM LLM()
```

Then re-run `llminspect inspect <pid> --verbose`.

### Path 2 — Inference in Docker (typical production service)

Install **inside the inference image** (not only on the host), rebuild/restart, then inspect with `docker exec`.

**1. Add to your Dockerfile** (preferred — PyPI):

```dockerfile
RUN pip install "llm-inspector[torch]"
```

**2. Rebuild and start** the service; wait until the model is loaded (VRAM stable).

**3. Smoke test from the host:**

```bash
# replace with your container name
CTR=my-inference   # e.g. vllm-server, stt-tts-service

docker exec "$CTR" llminspect --help
docker exec "$CTR" llminspect ps
```

**4. Inspect using a PID from that `ps` output** (container PID namespace):

```bash
docker exec "$CTR" llminspect inspect <pid-from-ps>
docker exec "$CTR" llminspect inspect <pid-from-ps> --verbose
```

**Never** pass a host `nvidia-smi` PID into `docker exec … inspect`.

**5. Optional — deep metrics in Docker**

- App / HF / TTS: `attach(model=…)` after load (see [§4](#4-embedded-integration-deep-metrics))
- `vllm serve`: vLLM plugin + `LLM_INSPECTOR_ATTACH=1` (see [§5](#5-vllm-serve-plugin-pattern))
- Confirm attach socket: `docker exec "$CTR" ls -la /tmp/llminspect/`

### What “success” looks like

| You did | Expect |
|---------|--------|
| External only | Process + Hardware GPU Used; Model may fill from APIs; Allocated / Weights often `Unavailable` |
| + `attach()` / plugin | Allocated / Reserved / Peak, Weights / KV / Workspace, Model Details, Optimization Analysis |

### Multi-GPU (v0.7+)

```bash
llminspect gpu          # all real NVML devices (Total / Used / Free / Util)
llminspect ps           # GPUs column: e.g. 0 or 0,1,2,3
llminspect inspect <pid> --verbose
```

- Process / Model / Runtime appear **once** (process-level).
- Each GPU the PID uses gets its own block: device VRAM, util, process VRAM (NVML).
- Per-GPU Allocated / Reserved / Peak appear only when `attach()` can measure them via `torch.cuda.memory_*(device)` inside the inference process — otherwise `Unavailable`.
- No projected per-GPU Weights/KV splits; memory breakdown stays process-level and measured.

**vLLM tensor-parallel jobs** often use one process per GPU (`VLLM::Worker_TP0`, `Worker_TP1`, …). Inspecting **any** member PID auto-groups siblings:

```bash
llminspect inspect 1538888 --verbose   # also pulls Worker_TP1 if same parent
llminspect inspect 1538888 --no-group  # single PID only
```

---

## Requirements

| Requirement | Notes |
|-------------|--------|
| Python **3.11+** | For `llminspect` CLI and/or `attach()` |
| OS | Linux or macOS |
| GPU | NVIDIA driver (`nvidia-smi` works). Consumer, datacenter, and DGX all fine. |
| Optional | PyTorch in the **same process** as inference (only for embedded `attach()`) |

macOS: external inspect works for CPU/Metal runtimes like Ollama; NVML GPU fields need NVIDIA.

---

## 1. Two integration levels

| Level | Code changes | What you get |
|-------|--------------|--------------|
| **External** | None | GPU VRAM (NVML), process info, runtime plugin data (Ollama/vLLM APIs) |
| **Embedded** | One `attach()` after model load | GPU Allocated / Reserved / Peak, Weights / KV / Activations / Workspace / Other, tokenizer + architecture details |

Start with external. Add embedded when you need “where did my VRAM go?” beyond NVML totals.

---

## A. Install on the host (no Docker)

Use this on a normal GPU server or workstation.

### From PyPI (recommended)

```bash
pip install llm-inspector
# or, for embedded deep metrics in the model env:
pip install "llm-inspector[torch]"
```

### From source (contributors / unreleased changes)

```bash
git clone https://github.com/helasaoudi/llm-inspector.git
cd llm-inspector
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -U pip
pip install -e ".[torch]"          # CLI + embedded support
# or: pip install -e ".[dev]"      # + pytest / lint tools
```

### Verify

```bash
llminspect --help
llminspect gpu          # GPU summary
llminspect ps           # inference processes on this machine
```

### System-wide (optional)

```bash
pip install "llm-inspector[torch]"
# ensure `llminspect` is on PATH (venv bin or ~/.local/bin)
```

### Cross-user processes (Linux)

If inference runs as another user (e.g. `root`) and `/proc` is unreadable:

```bash
sudo $(which llminspect) inspect <pid>
# or run llminspect as the same user as the inference process
```

---

## B. Install in Docker

Use this when inference runs **inside** a container. Install `llm-inspector` **in that image** so you can:

```bash
docker exec <container> llminspect ps
docker exec <container> llminspect inspect <container-pid>
```

### Dockerfile snippet (PyPI — recommended)

```dockerfile
RUN pip install "llm-inspector[torch]"
```

From source (only for unreleased local changes):

```dockerfile
COPY llm-inspector /tmp/llm-inspector
RUN pip install "/tmp/llm-inspector[torch]"
```

### Compose / env (embedded)

```yaml
environment:
  - LLM_INSPECTOR_ATTACH=1          # for auto_attach / vLLM plugin
  # - LLM_INSPECTOR_TRACE_LAYERS=1  # optional activation hooks
```

### Critical: PID namespaces

| You run | Use PID from |
|---------|----------------|
| Host: `llminspect inspect …` | Host `llminspect ps` or host `nvidia-smi` |
| `docker exec <ctr> llminspect inspect …` | `docker exec <ctr> llminspect ps` |

**Never** pass a host PID into `docker exec … inspect`.

---

## 2. Quick start by workload

### Ollama (host — simplest)

```bash
# terminal 1
ollama serve
ollama run llama3

# terminal 2
llminspect ps
llminspect inspect $(pgrep -f "ollama serve")
llminspect inspect $(pgrep -f "ollama serve") --verbose
```

No `attach()` needed for basic Ollama metrics (`/api/ps`).

### HuggingFace / PyTorch script (host)

```bash
pip install "llm-inspector[torch]"
```

```python
from transformers import AutoModelForCausalLM
from llm_inspector import attach

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B", device_map="cuda"
)
attach(model=model)  # after load; never raises into your app

# run inference as usual…
```

In another shell:

```bash
llminspect ps
llminspect inspect <pid> --verbose
```

### vLLM library API (host)

```python
from vllm import LLM
from llm_inspector import attach

llm = LLM(model="meta-llama/Llama-3-8B")
attach(engine=llm.llm_engine)  # or EngineCore on vLLM v1
```

### vLLM server (`vllm serve`) — needs a plugin

You cannot paste `attach(engine=…)` into the shell; the engine lives inside `EngineCore`.

Use a vLLM `general_plugins` entry point (see [§5](#5-vllm-serve-plugin-pattern)) plus:

```bash
export LLM_INSPECTOR_ATTACH=1
vllm serve openai/whisper-large-v3 --port 8000
```

Then:

```bash
llminspect ps
llminspect inspect $(pgrep -f 'VLLM::EngineCore' | head -1) --verbose
```

### Dockerized service

```bash
# Dockerfile: RUN pip install "llm-inspector[torch]"
docker compose up -d --build

# wait until the model is loaded (VRAM stable)
docker exec my-inference llminspect ps
docker exec my-inference llminspect inspect <pid-from-ps>
docker exec my-inference llminspect inspect <pid-from-ps> --verbose
```

Full checklist: [Test on your inference service](#test-on-your-inference-service-from-pypi).

---

## 3. External inspection (zero code changes)

Works for any GPU process the CLI can see: Ollama, vLLM, HuggingFace, FastAPI, custom PyTorch.

```bash
llminspect ps
llminspect gpu
llminspect inspect <pid>
llminspect inspect <pid> --verbose    # show provenance [source…]
llminspect runtimes                   # registered plugins
```

### What external can / cannot do

| Usually available | Needs embedded `attach()` |
|-------------------|---------------------------|
| GPU Used (NVML) | GPU Allocated / Reserved / Peak |
| Process RAM | Weights / KV / Activations / Workspace |
| Runtime name, some API fields | Full model config from engine internals |

---

## 4. Embedded integration (deep metrics)

Call **after** the model or engine is on the GPU.

### Generic template

```python
def load_model():
    model = ...  # your load logic
    try:
        from llm_inspector import attach
        attach(model=model)
        # optional: attach(model=model, trace_layers=True)
    except ImportError:
        pass
    return model
```

### Environment-only (memory peaks, no model binding)

```bash
export LLM_INSPECTOR_ATTACH=1
```

```python
from llm_inspector import auto_attach
auto_attach()  # once at startup
```

Gives Allocated / Reserved / Peak via `torch.cuda.*`, not full Weights/KV from the engine.

### Phase 4 breakdown (with `attach` after load)

| Component | Meaning |
|-----------|---------|
| Weights | Parameter tensors |
| KV Cache | PagedAttention blocks (vLLM) |
| Activations | Transient; **0 when idle** (expected) |
| Workspace | `reserved − allocated` |
| Other | Residual allocated not explained above |

### Phase 5 model details (with `attach` after load)

Extra **Model Details** section when the engine/model exposes them:

| Field | Source |
|-------|--------|
| Tokenizer / vocab / BOS / EOS / chat template | Bound tokenizer (or `hf_config.vocab_size`) |
| Layers, hidden size, attention heads, KV heads | `hf_config` |
| Experts (MoE) | `hf_config.num_experts` when present |
| Embed / Transformer / Head params | `named_parameters()` name buckets |

### Phase 6 Optimization Analysis (Projected)

After the measured sections, `llminspect inspect <pid>` continues with
**Optimization Analysis (Projected)** — currently Quantization scenarios
(FP8, INT8, AWQ/GPTQ 4-bit, …).

- Current memory stays **Measured**
- New totals are **Projected** (weight memory simulated; KV/Workspace/Other pass through)
- Recommendations call out bottlenecks (e.g. KV Cache dominance)
- Never modifies the model

---

## 5. vLLM serve plugin pattern

Minimal package:

**`pyproject.toml`**

```toml
[project]
name = "llminspect-vllm-plugin"
version = "0.1.0"
dependencies = []

[project.entry-points."vllm.general_plugins"]
llminspect = "llminspect_vllm_plugin:register"
```

**`llminspect_vllm_plugin.py`**

```python
import logging
import os

_log = logging.getLogger(__name__)


def register():
    if os.environ.get("LLM_INSPECTOR_ATTACH", "1").lower() not in ("1", "true", "yes"):
        return
    from vllm.v1.engine.core import EngineCore

    original = EngineCore._initialize_kv_caches

    def patched(self, *args, **kwargs):
        result = original(self, *args, **kwargs)
        try:
            from llm_inspector import attach, is_attached
            if not is_attached():
                attach(engine=self)
        except Exception as exc:
            _log.warning("llm-inspector attach failed: %s", exc)
        return result

    EngineCore._initialize_kv_caches = patched
```

Install next to vLLM (host venv or Docker image):

```bash
pip install "llm-inspector[torch]"
pip install -e "/path/to/llminspect-vllm-plugin"   # your general_plugins package
export LLM_INSPECTOR_ATTACH=1
```

---

## 6. Integration matrix

| Runtime | External | Embedded |
|---------|----------|----------|
| Ollama | ✅ `/api/ps` | N/A (separate binary) |
| vLLM serve | ✅ `/v1/models`, `/metrics` | ✅ plugin → `attach(engine=EngineCore)` |
| vLLM `LLM()` | ✅ if API up | ✅ `attach(engine=…)` |
| HuggingFace script | ✅ GPU process | ✅ `attach(model=…)` |
| FastAPI + local model | ✅ | ✅ `attach` after load |
| Custom PyTorch | ✅ NVML | ✅ `attach(model=…)` |

---

## 7. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `llminspect: command not found` | Not installed / venv not active | Activate venv or install into the environment you use |
| No processes in `llminspect ps` | Nothing on GPU / wrong machine | Check `nvidia-smi`; for Docker, exec into the container |
| GPU Allocated: Unavailable | No `attach()` | Add `attach()` or vLLM plugin |
| Activations: 0 B | Idle between requests | Expected; run inference and re-inspect (or check peak in `--verbose`) |
| Wrong PID / permission errors in Docker | Host PID used in container | Use `docker exec … llminspect ps` PIDs |
| `/proc` permission denied | Other user’s process | `sudo $(which llminspect) inspect <pid>` |
| VRAM Total: Unavailable | Some unified-memory GPUs (e.g. GB10) | NVML limitation; per-process Used still works |
| Model name Unavailable mid-startup | Inspected before load finished | Wait until VRAM stabilizes, then inspect |

**Attach health check (embedded):**

```bash
# host
ls -la /tmp/llminspect/

# docker
docker exec <container> ls -la /tmp/llminspect/
```

---

## 8. Docs map

| Doc | Audience |
|-----|----------|
| **This file** | Everyone — install + bare metal + Docker |
| [DGX_GUIDE.md](DGX_GUIDE.md) | Extra notes from a DGX Spark inference-service deployment |

---

## Philosophy

> **Measured. Not guessed.**

- `--verbose` shows the source of every number  
- Missing data → `Unavailable` + reason, never invented  
- `attach()` must never crash inference (`try` / `except` in your code; the library also swallows attach failures)
