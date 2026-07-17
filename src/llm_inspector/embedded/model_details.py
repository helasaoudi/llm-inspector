"""
Extract tokenizer + architecture + module breakdown from live objects.

All values are read from bound engine/model/tokenizer — never estimated.
"""

from __future__ import annotations

from typing import Any

from llm_inspector.models.measurement import Measurement


def _cfg_int(cfg: Any, *names: str) -> int | None:
    if cfg is None:
        return None
    for name in names:
        raw = getattr(cfg, name, None)
        if raw is None and hasattr(cfg, "get"):
            try:
                raw = cfg.get(name)
            except Exception:  # noqa: BLE001
                raw = None
        if raw is not None:
            try:
                return int(raw)
            except (TypeError, ValueError):
                continue
    return None


def _meas_int(value: int | None, source: str, missing: str) -> Measurement[int]:
    if value is None:
        return Measurement[int].unavailable(missing)
    return Measurement[int].available(value, source=source)


def _meas_str(value: str | None, source: str, missing: str) -> Measurement[str]:
    if not value:
        return Measurement[str].unavailable(missing)
    return Measurement[str].available(str(value), source=source)


def resolve_hf_config(engine: Any | None = None, model: Any | None = None) -> Any:
    """Best-effort HuggingFace / vLLM model config object."""
    if model is not None:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            return cfg
    if engine is None:
        return None
    for path in (
        ("model_config", "hf_config"),
        ("vllm_config", "model_config", "hf_config"),
        ("model_config",),
    ):
        obj: Any = engine
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        else:
            if obj is not None:
                return obj
    return None


def resolve_tokenizer(
    engine: Any | None = None,
    model: Any | None = None,
) -> Any | None:
    """Locate a tokenizer / processor object on engine or model."""
    candidates: list[Any] = []
    if model is not None:
        candidates.extend(
            [
                getattr(model, "tokenizer", None),
                getattr(model, "processor", None),
            ]
        )
    if engine is not None:
        candidates.extend(
            [
                getattr(engine, "tokenizer", None),
                getattr(engine, "input_tokenizer", None),
                getattr(getattr(engine, "input_preprocessor", None), "tokenizer", None),
                getattr(getattr(engine, "preprocessor", None), "tokenizer", None),
                getattr(getattr(engine, "scheduler", None), "tokenizer", None),
            ]
        )
        # vLLM sometimes keeps tokenizer on the tokenizer group / detokenizer
        tok_group = getattr(engine, "tokenizer_group", None) or getattr(
            engine, "tokenizer", None
        )
        if tok_group is not None:
            candidates.append(getattr(tok_group, "tokenizer", tok_group))

    for tok in candidates:
        if tok is None:
            continue
        # Prefer objects that look like HF tokenizers
        if hasattr(tok, "vocab_size") or hasattr(tok, "get_vocab") or hasattr(
            tok, "name_or_path"
        ):
            return tok
        if hasattr(tok, "tokenizer"):
            inner = getattr(tok, "tokenizer", None)
            if inner is not None:
                return inner
    return None


def extract_tokenizer_fields(tokenizer: Any | None) -> dict[str, Measurement]:
    """Tokenizer class, vocab size, chat template, special tokens."""
    if tokenizer is None:
        reason = "Tokenizer not bound on engine/model (attach after load)."
        return {
            "tokenizer_class": Measurement[str].unavailable(reason),
            "vocab_size": Measurement[int].unavailable(reason),
            "chat_template": Measurement[str].unavailable(reason),
            "bos_token": Measurement[str].unavailable(reason),
            "eos_token": Measurement[str].unavailable(reason),
        }

    cls_name = type(tokenizer).__name__
    tokenizer_class = Measurement[str].available(
        cls_name, source="tokenizer.__class__.__name__"
    )

    vocab: int | None = None
    if hasattr(tokenizer, "vocab_size"):
        try:
            vocab = int(tokenizer.vocab_size)
            vsrc = "tokenizer.vocab_size"
        except Exception:  # noqa: BLE001
            vocab = None
            vsrc = ""
    if vocab is None and hasattr(tokenizer, "get_vocab"):
        try:
            vocab = len(tokenizer.get_vocab())
            vsrc = "len(tokenizer.get_vocab())"
        except Exception:  # noqa: BLE001
            vocab = None
            vsrc = ""
    vocab_size = _meas_int(
        vocab, vsrc or "tokenizer", "Tokenizer has no vocab_size."
    )

    template = getattr(tokenizer, "chat_template", None)
    if template:
        chat_template = Measurement[str].available(
            "yes",
            source="tokenizer.chat_template (present)",
        )
    else:
        chat_template = Measurement[str].available(
            "no",
            source="tokenizer.chat_template (absent)",
        )

    bos = getattr(tokenizer, "bos_token", None) or getattr(
        tokenizer, "bos_token_id", None
    )
    eos = getattr(tokenizer, "eos_token", None) or getattr(
        tokenizer, "eos_token_id", None
    )
    bos_token = _meas_str(
        None if bos is None else str(bos),
        "tokenizer.bos_token",
        "BOS token not set on tokenizer.",
    )
    eos_token = _meas_str(
        None if eos is None else str(eos),
        "tokenizer.eos_token",
        "EOS token not set on tokenizer.",
    )

    return {
        "tokenizer_class": tokenizer_class,
        "vocab_size": vocab_size,
        "chat_template": chat_template,
        "bos_token": bos_token,
        "eos_token": eos_token,
    }


def extract_architecture_fields(hf_config: Any | None) -> dict[str, Measurement]:
    """Layers / hidden size / attention heads / MoE from HF or vLLM config."""
    if hf_config is None:
        reason = "Model config (hf_config) not accessible."
        return {
            "num_layers": Measurement[int].unavailable(reason),
            "hidden_size": Measurement[int].unavailable(reason),
            "num_attention_heads": Measurement[int].unavailable(reason),
            "num_kv_heads": Measurement[int].unavailable(reason),
            "num_experts": Measurement[int].unavailable(reason),
        }

    # Whisper / encoder-decoder: prefer decoder layers when both exist
    layers = _cfg_int(
        hf_config,
        "num_hidden_layers",
        "n_layer",
        "num_layers",
        "decoder_layers",
        "num_decoder_layers",
    )
    # Some multimodal configs nest text_config
    text_cfg = getattr(hf_config, "text_config", None)
    if layers is None and text_cfg is not None:
        layers = _cfg_int(text_cfg, "num_hidden_layers", "n_layer", "num_layers")

    hidden = _cfg_int(hf_config, "hidden_size", "d_model", "n_embd")
    if hidden is None and text_cfg is not None:
        hidden = _cfg_int(text_cfg, "hidden_size", "d_model", "n_embd")

    heads = _cfg_int(
        hf_config, "num_attention_heads", "n_head", "encoder_attention_heads"
    )
    if heads is None and text_cfg is not None:
        heads = _cfg_int(text_cfg, "num_attention_heads", "n_head")

    kv_heads = _cfg_int(
        hf_config,
        "num_key_value_heads",
        "num_kv_heads",
        "n_kv_heads",
    )
    if kv_heads is None and text_cfg is not None:
        kv_heads = _cfg_int(text_cfg, "num_key_value_heads", "num_kv_heads")
    if kv_heads is None and heads is not None:
        # GQA not configured → KV heads == attention heads (measured default)
        kv_heads = heads

    experts = _cfg_int(
        hf_config,
        "num_local_experts",
        "num_experts",
        "n_routed_experts",
    )
    if experts is None and text_cfg is not None:
        experts = _cfg_int(text_cfg, "num_local_experts", "num_experts")

    return {
        "num_layers": _meas_int(
            layers,
            "hf_config.num_hidden_layers",
            "num_hidden_layers not on model config.",
        ),
        "hidden_size": _meas_int(
            hidden,
            "hf_config.hidden_size",
            "hidden_size not on model config.",
        ),
        "num_attention_heads": _meas_int(
            heads,
            "hf_config.num_attention_heads",
            "num_attention_heads not on model config.",
        ),
        "num_kv_heads": _meas_int(
            kv_heads,
            "hf_config.num_key_value_heads",
            "num_key_value_heads not on model config.",
        ),
        "num_experts": (
            Measurement[int].available(experts, source="hf_config.num_experts")
            if experts is not None
            else Measurement[int].unavailable("Not an MoE model (no num_experts).")
        ),
    }


def extract_module_param_breakdown(model: Any | None) -> dict[str, Measurement[int]]:
    """
    Split parameter counts into embed / transformer / head by parameter name.

    Heuristic on ``named_parameters()`` names — still measured counts, only
    the bucket assignment is by name pattern.
    """
    if model is None or not hasattr(model, "named_parameters"):
        reason = "Model module not accessible for named_parameters()."
        return {
            "embed_params": Measurement[int].unavailable(reason),
            "transformer_params": Measurement[int].unavailable(reason),
            "head_params": Measurement[int].unavailable(reason),
        }

    embed = 0
    head = 0
    transformer = 0
    try:
        for name, param in model.named_parameters():
            n = int(param.numel())
            lname = name.lower()
            if any(
                key in lname
                for key in (
                    "embed_tokens",
                    "wte",
                    "word_embeddings",
                    "tok_embeddings",
                    "embedding.word",
                    "model.embed",
                )
            ):
                embed += n
            elif any(
                key in lname
                for key in (
                    "lm_head",
                    "embed_out",
                    "output_projection",
                    "proj_out",
                    "score.weight",  # some classifiers
                )
            ):
                head += n
            else:
                transformer += n
    except Exception as exc:  # noqa: BLE001
        reason = str(exc)
        return {
            "embed_params": Measurement[int].unavailable(reason),
            "transformer_params": Measurement[int].unavailable(reason),
            "head_params": Measurement[int].unavailable(reason),
        }

    src = "model.named_parameters() → name-bucketed sum(numel)"
    return {
        "embed_params": Measurement[int].available(embed, source=src),
        "transformer_params": Measurement[int].available(transformer, source=src),
        "head_params": Measurement[int].available(head, source=src),
    }


def collect_model_detail_fields(
    *,
    engine: Any | None = None,
    model: Any | None = None,
    resolve_model: Any | None = None,
) -> dict[str, Measurement]:
    """Aggregate tokenizer + architecture + module breakdown measurements."""
    hf_cfg = resolve_hf_config(engine=engine, model=model)
    tokenizer = resolve_tokenizer(engine=engine, model=model)

    module = model
    if module is None and callable(resolve_model) and engine is not None:
        try:
            module = resolve_model(engine)
        except Exception:  # noqa: BLE001
            module = None

    fields: dict[str, Measurement] = {}
    fields.update(extract_tokenizer_fields(tokenizer))
    fields.update(extract_architecture_fields(hf_cfg))
    fields.update(extract_module_param_breakdown(module))

    # Vocab size often lives on hf_config even when tokenizer isn't on EngineCore
    if not fields["vocab_size"].is_available and hf_cfg is not None:
        vocab = _cfg_int(hf_cfg, "vocab_size")
        text_cfg = getattr(hf_cfg, "text_config", None)
        if vocab is None and text_cfg is not None:
            vocab = _cfg_int(text_cfg, "vocab_size")
        if vocab is not None:
            fields["vocab_size"] = Measurement[int].available(
                vocab, source="hf_config.vocab_size"
            )

    return fields
