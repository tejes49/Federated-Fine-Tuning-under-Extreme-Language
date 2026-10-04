"""LoRA via Hugging Face PEFT: attach, freeze base, count params, save/load adapter."""
from typing import Optional

import torch
from peft import LoraConfig, PeftModel, get_peft_model

# Attention projection names per architecture (used when config gives no explicit list).
AUTO_TARGETS = {
    "gpt2": ["c_attn"],
    "bloom": ["query_key_value"],
    "llama": ["q_proj", "v_proj"], "qwen2": ["q_proj", "v_proj"], "mistral": ["q_proj", "v_proj"],
    "gemma": ["q_proj", "v_proj"], "opt": ["q_proj", "v_proj"], "gpt_neox": ["query_key_value"],
}


def default_target_modules(model) -> list:
    mt = getattr(model.config, "model_type", "")
    if mt not in AUTO_TARGETS:
        raise ValueError(f"No default LoRA targets for model_type '{mt}'. "
                         f"Set model.lora_target_modules in the config.")
    return AUTO_TARGETS[mt]


def attach_lora(model, lora_cfg: dict, target_modules: Optional[list] = None):
    """Freeze every base parameter, then wrap with LoRA so only adapter weights are trainable."""
    for p in model.parameters():
        p.requires_grad = False
    cfg = LoraConfig(r=lora_cfg["r"], lora_alpha=lora_cfg["alpha"], lora_dropout=lora_cfg["dropout"],
                     target_modules=target_modules or default_target_modules(model),
                     bias="none", task_type="CAUSAL_LM",
                     fan_in_fan_out=getattr(model.config, "model_type", "") == "gpt2")  # GPT-2 uses Conv1D
    return get_peft_model(model, cfg)


def count_parameters(model) -> dict:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"trainable_parameters": trainable, "total_parameters": total,
            "trainable_percent": round(100.0 * trainable / total, 4) if total else 0.0}


def save_adapter(peft_model, path: str) -> None:
    peft_model.save_pretrained(path)   # adapter weights + adapter_config.json only (not the base model)


def load_adapter(base_model, path: str, trainable: bool = False):
    return PeftModel.from_pretrained(base_model, path, is_trainable=trainable)


def adapter_state(peft_model) -> dict:
    """Only the LoRA tensors (what a federated client would later send)."""
    return {k: v.detach().cpu().clone() for k, v in peft_model.named_parameters() if "lora_" in k}


def lora_summary(peft_model) -> dict:
    """LoRA hyper-parameters exactly as PEFT holds them (not re-read from the config)."""
    c = next(iter(peft_model.peft_config.values()))
    return {"r": c.r, "alpha": c.lora_alpha, "dropout": c.lora_dropout,
            "target_modules": sorted(c.target_modules), "fan_in_fan_out": c.fan_in_fan_out}


def verify_lora_setup(peft_model) -> dict:
    """Freezing check: all LoRA tensors trainable, nothing else trainable."""
    lora = [(n, p) for n, p in peft_model.named_parameters() if "lora_" in n]
    other = [(n, p) for n, p in peft_model.named_parameters() if "lora_" not in n]
    return {"n_lora_tensors": len(lora),
            "n_lora_frozen": sum(1 for _, p in lora if not p.requires_grad),
            "n_non_lora_trainable": sum(1 for _, p in other if p.requires_grad)}
