# LLM Inspector — DGX notes

> **Looking for install on any GPU machine?** Start here: **[INSTALL_GUIDE.md](INSTALL_GUIDE.md)** (bare metal + Docker).

This document keeps **DGX Spark–specific** notes from integrating LLM Inspector into an inference stack (`stt-tts-service` + `vllm-server`). Same APIs as everywhere else; the differences are mostly Docker PID namespaces, GB10 NVML limits, and the vLLM EngineCore plugin.

**Goal:** Observe live GPU inference from outside (`llminspect`) or inside (`attach()`), with every metric measured or explicitly `Unavailable`.

---

## 1. Two integration levels

| Level | Code changes | What you get |
|-------|--------------|--------------|
| **External** (zero code) | None | GPU VRAM, runtime, model name (if API exposes it), process info |
| **Embedded** (recommended for prod) | `attach()` after model load | GPU Allocated / Reserved / Peak, weights / KV breakdown, model config from engine |

```bash
pip install "llm-inspector[torch]"
# or from source:
pip install -e "/path/to/llm-inspector[torch]"
```

**Requirements**

- Python 3.11+
- Linux or macOS
- GPU: NVIDIA driver + `nvidia-ml-py` (included)
- Embedded: PyTorch in the **same process** as inference

---

## 2. External inspection (any repo, no code changes)

Works for Ollama, vLLM, HuggingFace, FastAPI, custom PyTorch — anything on GPU.

```bash
# List GPU-attached inference processes
llminspect ps

# Inspect one process (host PID from ps or nvidia-smi)
llminspect inspect <pid>

# See where every number came from
llminspect inspect <pid> --verbose
```

### Docker PID rules

| Run from | Use PID from |
|----------|--------------|
| Host | `llminspect ps` on host / `nvidia-smi` |
| `docker exec <container> …` | `docker exec <container> llminspect ps` |

Never pass a **host** PID into `docker exec … inspect`.

### What external inspect can / cannot do

| Available | Often Unavailable |
|-----------|-------------------|
| GPU Used (NVML) | GPU Allocated / Reserved (needs `attach()`) |
| Process RAM | Weights / KV breakdown (needs `attach()` or runtime `/metrics` bytes) |
| Runtime plugin data (vLLM `/v1/models`, etc.) | Architecture, params (unless API or embedded) |

---

## 3. Embedded integration (deep metrics)

Add one call **after** model / engine load in your inference code.

### HuggingFace / PyTorch

```python
from llm_inspector import attach

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3-8B", device_map="cuda"
)
attach(model=model)  # never raises into inference
```

### Coqui TTS / XTTS (e.g. `stt-tts-service`)

```python
def _attach_inspector(self) -> None:
    try:
        from llm_inspector import attach
    except ImportError:
        return  # optional in dev / Docker without the package

    try:
        underlying = getattr(
            getattr(self._tts, "synthesizer", None), "tts_model", None
        )
        attach(model=underlying) if underlying else attach()
    except Exception:
        pass  # never break inference
```

Call this after `_load_tts_from_disk()` / model init.

### vLLM (library API — you own the engine)

```python
from llm_inspector import attach

engine = LLM(model="meta-llama/Llama-3-8B")
attach(engine=engine.llm_engine)  # or EngineCore in vLLM v1
```

### Environment-only (memory-only, no model binding)

```bash
# Before process start:
export LLM_INSPECTOR_ATTACH=1
```

```python
from llm_inspector import auto_attach

auto_attach()  # once at startup, or via a sitecustomize hook
```

Gives GPU Allocated / Reserved / Peak via `torch.cuda.*`, but **not** model name from the engine.

---

## 4. vLLM server (`vllm serve`) — plugin pattern

You cannot paste `attach(engine=…)` into a shell. The engine is created inside `EngineCore` after startup.

**Pattern:** vLLM `general_plugins` entry point → hook after engine / KV-cache init.

### A. Plugin package (minimal)

`pyproject.toml`:

```toml
[project.entry-points."vllm.general_plugins"]
llminspect = "llminspect_vllm_plugin:register"
```

`llminspect_vllm_plugin.py`:

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

Hook **after** `_initialize_kv_caches` so `kv_cache_config` / weights are available for Memory Breakdown.

### B. Docker

```dockerfile
FROM nvcr.io/nvidia/vllm:26.02-py3

RUN pip install "/path/to/llm-inspector[torch]"
COPY docker/llminspect_vllm_plugin /tmp/plugin
RUN pip install /tmp/plugin
```

```yaml
environment:
  - LLM_INSPECTOR_ATTACH=1
```

### C. Inspect after load

Wait until VRAM stabilizes (often 1–3 minutes on first boot):

```bash
curl http://localhost:8010/health

docker exec vllm-server llminspect ps
# EngineCore VRAM should be ~full model size (e.g. ~17 GB), not a few GB mid-load

docker exec vllm-server llminspect inspect <EngineCore-pid>
docker exec vllm-server llminspect inspect <EngineCore-pid> --verbose
```

### Example measured output (Whisper large-v3 on DGX GB10)

| Field | Example |
|-------|---------|
| Parameters | 1.5B (`model.parameters()`) |
| Weights | ~2.9 GB |
| KV Cache | ~13.1 GB (`kv_cache_tensors`) |
| GPU Allocated / Reserved / Peak | via `torch.cuda.*` |
| VRAM Total | Unavailable on GB10 (NVML “Not Supported”) |

---

## 5. Docker checklist (any service)

| Step | Action |
|------|--------|
| 1 | `pip install llm-inspector[torch]` in the image (or mount / COPY the repo at build) |
| 2 | Optional: `attach()` in app code after model load |
| 3 | For `vllm serve`: add `general_plugins` + `LLM_INSPECTOR_ATTACH=1` |
| 4 | Install `llminspect` CLI in the image if you want `docker exec … llminspect` |
| 5 | Wait for model load before inspecting |

Example layout (`stt-tts-service`):

```text
stt-tts-service/
  Dockerfile                 # llm-inspector via build context
  Dockerfile.vllm            # llm-inspector + vLLM plugin
  docker/llminspect_vllm_plugin/
  app/services/tts_service.py   # lazy attach() after XTTS load
```

---

## 6. Integration by runtime

| Runtime | External (`llminspect inspect`) | Embedded (`attach`) |
|---------|----------------------------------|---------------------|
| Ollama | ✅ `/api/ps`, `/api/version` | N/A (separate process) |
| vLLM serve | ✅ `/v1/models`, `/metrics` | ✅ plugin → `attach(engine=EngineCore)` |
| HuggingFace script | ✅ if GPU process visible | ✅ `attach(model=model)` |
| FastAPI + local model | ✅ detected as FastAPI | ✅ `attach(model=…)` after load |
| Coqui / custom PyTorch | ✅ GPU VRAM via NVML | ✅ `attach(model=underlying_module)` |

---

## 7. What changed in llm-inspector for DGX / vLLM v1

| Area | Change |
|------|--------|
| Process detection | Recognize `VLLM::EngineCore` workers |
| API port probe | Probe common ports when worker cmdline has no `--port` |
| `VLLMAdapter` | `EngineCore` / `EngineCoreProc`, `vllm_config.model_config` |
| Embedded model | Architecture, dtype, `max_model_len`, TP/PP |
| Embedded memory | `model.parameters()` weights + `kv_cache_tensors` KV size |
| Phase 4 breakdown | Workspace = reserved−allocated; Activations = allocated−baseline; Other = residual |
| Collectors | Merge embedded + API so one source fills gaps |

### Phase 4 memory breakdown (embedded)

With `attach()` after model load:

| Component | How it is measured |
|-----------|-------------------|
| Weights | `model.parameters()` byte sum |
| KV Cache | `kv_cache_tensors` sizes (vLLM) |
| Activations | `allocated − attach baseline` (0 when idle; peak noted in `--verbose` source) |
| Workspace | `memory_reserved() − memory_allocated()` |
| Other | `allocated − weights − KV − current activations` |

Optional finer activation peaks:

```python
attach(engine=engine, trace_layers=True)
# or: LLM_INSPECTOR_TRACE_LAYERS=1
```

---

## 8. Minimal integration template

**Any Python inference repo:**

```python
# after model is on GPU
def load_model():
    model = ...  # your load logic
    try:
        from llm_inspector import attach
        attach(model=model)
    except ImportError:
        pass
    return model
```

**Any Docker GPU service:**

```dockerfile
RUN pip install "llm-inspector[torch] @ git+ssh://git@github.com/helasaoudi/llm-inspector.git"
```

**Ops / debugging:**

```bash
llminspect ps
llminspect inspect <pid>
llminspect inspect <pid> --verbose
```

---

## 9. Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| GPU Allocated: Unavailable | No embedded attach | Add `attach()` or vLLM plugin |
| Name: Unavailable on EngineCore | Inspected before load / no attach | Wait until VRAM stable; check `/tmp/llminspect/` socket |
| Cannot read cmdline PID X in `docker exec` | Host PID used in container | Use container PID from `docker exec … llminspect ps` |
| Memory breakdown Unavailable | No attach; `/metrics` has no byte metrics (vLLM 0.15+) | Use embedded attach (plugin) |
| `llminspect: command not found` in container | Not installed in image | Add to Dockerfile |
| Attach worked earlier, then all Unavailable after rebuild | Plugin not installed or `LLM_INSPECTOR_ATTACH=0` | Check logs + `ls /tmp/llminspect/` |

**Quick attach health check:**

```bash
docker exec vllm-server ls -la /tmp/llminspect/
docker logs vllm-server 2>&1 | grep -iE 'llm-inspector|attach' | tail -20
```

---

## 10. Philosophy

> **Measured. Not guessed.**

- Every value has a source (`--verbose` shows it)
- Missing data → `Unavailable` + reason, never fabricated
- `attach()` must never raise into inference (library + your `try`/`except`)
