"""Tokenizer + base model loading (single place to change the model)."""
from typing import Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from utils.reproducibility import get_device

TINY_OFFLINE = "tiny-offline"


def _build_offline_tokenizer() -> PreTrainedTokenizerFast:
    """Byte-level tokenizer built locally (256 byte tokens + specials). Handles any Unicode, no download."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    vocab = {ch: i for i, ch in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
    for sp in ("<pad>", "<eos>"):
        vocab[sp] = len(vocab)
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    return PreTrainedTokenizerFast(tokenizer_object=tok, pad_token="<pad>", eos_token="<eos>")


def load_tokenizer(model_cfg: dict) -> PreTrainedTokenizerFast:
    name = model_cfg["name"]
    tok = _build_offline_tokenizer() if name == TINY_OFFLINE else AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


PRECISIONS = ("auto", "fp32", "bf16")


def resolve_dtype(device: str, precision: str = "auto") -> torch.dtype:
    """auto: bf16 on CUDA GPUs that support it, else fp32. fp16 is deliberately unsupported
    (the training loop has no loss scaling)."""
    p = (precision or "auto").lower()
    if p == "fp32":
        return torch.float32
    if p == "bf16":
        return torch.bfloat16
    if p == "auto":
        return torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else torch.float32
    raise ValueError(f"Unsupported precision '{precision}'; use one of {PRECISIONS}")


def load_base_model(model_cfg: dict, tokenizer, device_pref: str = "auto",
                    precision: str = "auto") -> Tuple[torch.nn.Module, str]:
    """Return (model, device). The model is NOT frozen here; models.lora.attach_lora freezes it."""
    device = get_device(device_pref)
    name = model_cfg["name"]
    if name == TINY_OFFLINE:
        cfg = GPT2Config(vocab_size=len(tokenizer), n_positions=model_cfg.get("max_seq_len", 64),
                         n_embd=64, n_layer=2, n_head=2,
                         pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                         bos_token_id=tokenizer.eos_token_id)
        model = GPT2LMHeadModel(cfg)   # random init: pipeline check only
        if precision not in (None, "auto"):
            model = model.to(resolve_dtype(device, precision))
    else:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=resolve_dtype(device, precision))
    model.config.use_cache = False
    return model.to(device), device


def check_tokenizer_model_compat(tokenizer, model) -> dict:
    """Every id the tokenizer can produce must have an embedding row."""
    emb = model.get_input_embeddings().num_embeddings
    n = len(tokenizer)
    if n > emb:
        raise ValueError(f"Tokenizer has {n} tokens but the model only has {emb} embedding rows")
    return {"tokenizer_size": n, "embedding_size": emb}
