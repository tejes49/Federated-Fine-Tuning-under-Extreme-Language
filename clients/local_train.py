"""Single-client LoRA training.

Run:  python -m clients.local_train --language tamil
      python -m clients.local_train --config configs/dev.yaml --language ta --clients-dir /some/dir
Reads data/clients/<id>/{train,val}.jsonl (made by `python -m data.prepare_data`).
Writes results/single_client/<id>/metrics.json and checkpoints/single_client/<id>/adapter/.
"""
import argparse
import json
import math
import time
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.base_model import TINY_OFFLINE, check_tokenizer_model_compat, load_base_model, load_tokenizer
from models.lora import (adapter_state, attach_lora, count_parameters, lora_summary, save_adapter,
                         verify_lora_setup)
from utils.checkpoints import load_config, save_json
from utils.config_overrides import OVERRIDES, apply_overrides
from utils.logging import ExperimentLog, get_logger
from utils.reproducibility import set_seed
from utils.resources import describe_environment, peak_memory_gb

log = get_logger()

LANG_ALIASES = {"english": "en", "hindi": "hi", "tamil": "ta", "telugu": "te", "malayalam": "ml"}


# ---------- data ----------
def resolve_client_id(language: str, data_config: Optional[str] = None) -> str:
    """'tamil' / 'ta' -> client id 'ta' (ids come from configs/data.yaml when available)."""
    key = language.lower().strip()
    code = LANG_ALIASES.get(key, key)
    if data_config and Path(data_config).exists():
        ids = {c["lang"]: c["id"] for c in load_config(data_config).get("clients", [])}
        if code in ids:
            return ids[code]
    return code


def read_texts(path: Path, max_n: Optional[int]) -> List[str]:
    """Read up to max_n non-blank passages from a jsonl file (blank/missing 'text' entries are skipped)."""
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Build client data first: python -m data.prepare_data")
    texts, n_blank = [], 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            text = json.loads(line).get("text")
            if not isinstance(text, str) or not text.strip():
                n_blank += 1
                continue
            texts.append(text)
            if max_n and len(texts) >= max_n:
                break
    if n_blank:
        log.warning(f"{path}: skipped {n_blank} blank passages")
    if not texts:
        raise ValueError(f"{path} contains no usable examples")
    return texts


def tokenize(tokenizer, texts: List[str], max_len: int) -> List[List[int]]:
    """Token ids, at most max_len long. EOS is appended only when the passage fits: a truncated passage
    is cut mid-text, and teaching the model to 'end' there would be wrong. Sequences with fewer than
    2 ids are dropped (they have no next-token target)."""
    out = []
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"]
        if len(ids) < max_len:
            ids = ids + [tokenizer.eos_token_id]
        if len(ids) >= 2:
            out.append(ids)
    return out


def make_collate(pad_id: int):
    def collate(batch):
        n = max(len(x) for x in batch)
        ids = torch.full((len(batch), n), pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), n), dtype=torch.long)
        for i, x in enumerate(batch):
            ids[i, :len(x)] = torch.tensor(x)
            mask[i, :len(x)] = 1
        return {"input_ids": ids, "attention_mask": mask}
    return collate


# ---------- loss / perplexity ----------
def build_labels(ids, mask):
    """Next-token labels aligned with logits[:, :-1]; padding positions become -100 (ignored)."""
    return ids[:, 1:].masked_fill(mask[:, 1:] == 0, -100)


def batch_loss_sum(model, batch, device):
    """Return (sum of token NLL, number of predicted tokens). Padding is excluded via the attention mask."""
    ids, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
    logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1].float()
    labels = build_labels(ids, mask)
    loss_sum = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                               ignore_index=-100, reduction="sum")
    return loss_sum, (labels != -100).sum()


def perplexity(loss: float) -> float:
    """exp(loss), returning inf instead of overflowing."""
    return math.exp(loss) if loss < 700 else float("inf")


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Token-weighted mean loss over the whole loader (not a mean of batch means)."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        s, n = batch_loss_sum(model, batch, device)
        total += s.item()
        count += n.item()
    model.train(was_training)
    if count == 0:
        raise ValueError("no valid tokens in validation set")
    loss = total / count
    return {"validation_loss": loss, "perplexity": perplexity(loss), "validation_tokens": count}


# ---------- training ----------
def train_client(cfg: dict, language: str, clients_dir: Optional[str] = None,
                 output_tag: Optional[str] = None) -> dict:
    seed, mcfg, dcfg, tcfg, ocfg = cfg["seed"], cfg["model"], cfg["data"], cfg["train"], cfg["output"]
    set_seed(seed)
    cid = resolve_client_id(language, dcfg.get("data_config"))
    cdir = Path(clients_dir or dcfg["clients_dir"]) / cid
    tag = output_tag or cid

    tok = load_tokenizer(mcfg)
    train_ids = tokenize(tok, read_texts(cdir / "train.jsonl", dcfg.get("max_train_samples")), mcfg["max_seq_len"])
    val_ids = tokenize(tok, read_texts(cdir / "val.jsonl", dcfg.get("max_validation_samples")), mcfg["max_seq_len"])
    if not train_ids or not val_ids:
        raise ValueError(f"[{cid}] no usable sequences after tokenization (train={len(train_ids)}, val={len(val_ids)})")
    collate = make_collate(tok.pad_token_id)
    g = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(train_ids, batch_size=tcfg["batch_size"], shuffle=True, collate_fn=collate, generator=g)
    val_loader = DataLoader(val_ids, batch_size=tcfg["batch_size"], shuffle=False, collate_fn=collate)

    base, device = load_base_model(mcfg, tok, cfg.get("device", "auto"), cfg.get("precision", "auto"))
    vocab_info = check_tokenizer_model_compat(tok, base)
    base_dtype = str(getattr(base, "dtype", None))
    model = attach_lora(base, cfg["lora"], mcfg.get("lora_target_modules"))
    setup = verify_lora_setup(model)
    if setup["n_lora_tensors"] == 0 or setup["n_lora_frozen"] or setup["n_non_lora_trainable"]:
        raise RuntimeError(f"[{cid}] LoRA/freezing setup is wrong: {setup}")
    params = count_parameters(model)
    log.info(f"[{cid}] device={device} model={mcfg['name']} trainable={params['trainable_parameters']:,} "
             f"/ total={params['total_parameters']:,} ({params['trainable_percent']}%)")

    accum = tcfg["grad_accum_steps"]
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = tcfg["max_steps"] if tcfg.get("max_steps") is not None else tcfg["epochs"] * steps_per_epoch
    if total_steps < 1:
        raise ValueError(f"[{cid}] total optimizer steps is {total_steps}; check train.max_steps / train.epochs")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tcfg["learning_rate"], weight_decay=tcfg["weight_decay"])

    (Path(ocfg["results_dir"]) / tag / "log.jsonl").unlink(missing_ok=True)   # one log per run, not appended
    exp = ExperimentLog(str(Path(ocfg["results_dir"]) / tag))
    initial = evaluate(model, val_loader, device)
    log.info(f"[{cid}] before training: val_loss={initial['validation_loss']:.4f} ppl={initial['perplexity']:.2f}")
    exp.log(event="initial_eval", **initial)

    lora_before = adapter_state(model)
    t0, step, micro, train_losses = time.time(), 0, 0, []
    run_sum, run_cnt = 0.0, 0
    model.train()
    opt.zero_grad()
    while step < total_steps:
        for batch in train_loader:
            s, n = batch_loss_sum(model, batch, device)
            (s / n.clamp(min=1) / accum).backward()          # mean token loss of this micro-batch, scaled for accumulation
            run_sum, run_cnt, micro = run_sum + s.item(), run_cnt + n.item(), micro + 1
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                               tcfg["max_grad_norm"])
                opt.step(); opt.zero_grad(); step += 1
                train_losses.append(run_sum / run_cnt); run_sum, run_cnt = 0.0, 0
                exp.log(event="step", step=step, train_loss=train_losses[-1])
                if step % 10 == 0 or step == total_steps:
                    log.info(f"[{cid}] step {step}/{total_steps} train_loss={train_losses[-1]:.4f}")
                ev = tcfg.get("eval_every_steps")
                if ev and step % ev == 0 and step < total_steps:
                    exp.log(event="eval", step=step, **evaluate(model, val_loader, device))
                if step >= total_steps:
                    break
    final = evaluate(model, val_loader, device)
    lora_after = adapter_state(model)
    lora_updated = any(not torch.equal(lora_before[k], lora_after[k]) for k in lora_before)
    ckpt = Path(ocfg["checkpoints_dir"]) / tag / "adapter"
    save_adapter(model, str(ckpt))

    metrics = {
        "language": language, "client_id": cid, "model": mcfg["name"], "device": device,
        "model_is_random_init": mcfg["name"] == TINY_OFFLINE, "seed": seed,
        "n_train_samples": len(train_ids), "n_validation_samples": len(val_ids),
        "optimizer_steps": step, "batch_size": tcfg["batch_size"], "grad_accum_steps": accum,
        "max_seq_len": mcfg["max_seq_len"], "lora": cfg["lora"],
        "initial_validation_loss": initial["validation_loss"], "initial_perplexity": initial["perplexity"],
        "validation_loss": final["validation_loss"], "perplexity": final["perplexity"],
        "validation_tokens": final["validation_tokens"],
        "first_train_loss": train_losses[0], "last_train_loss": train_losses[-1],
        "trainable_parameters": params["trainable_parameters"], "total_parameters": params["total_parameters"],
        "trainable_percent": params["trainable_percent"],
        "train_seconds": round(time.time() - t0, 2), "adapter_path": str(ckpt),
        "precision_setting": cfg.get("precision", "auto"), "base_dtype": base_dtype,
        "lora_summary": lora_summary(model), "lora_setup_check": setup, "lora_updated": lora_updated,
        **vocab_info, "environment": describe_environment(), "peak_memory": peak_memory_gb(),
    }
    save_json(metrics, str(Path(ocfg["results_dir"]) / tag / "metrics.json"))
    save_json(cfg, str(Path(ocfg["results_dir"]) / tag / "config_used.json"))
    log.info(f"[{cid}] done: val_loss={final['validation_loss']:.4f} ppl={final['perplexity']:.2f}")
    return metrics


def main():
    ap = argparse.ArgumentParser(description="Single-client LoRA training (any config value can be overridden)")
    ap.add_argument("--config", default="configs/single_client.yaml")
    ap.add_argument("--language", required=True, help="english|hindi|tamil|telugu|malayalam or en|hi|ta|te|ml")
    ap.add_argument("--clients-dir", default=None)
    ap.add_argument("--output-tag", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-seq-len", type=int, default=None)
    ap.add_argument("--device", default=None, help="auto|cpu|cuda")
    ap.add_argument("--precision", default=None, choices=["auto", "fp32", "bf16"])
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--lora-r", type=int, default=None)
    ap.add_argument("--lora-alpha", type=int, default=None)
    ap.add_argument("--lora-dropout", type=float, default=None)
    ap.add_argument("--max-train-samples", type=int, default=None)
    ap.add_argument("--max-val-samples", type=int, default=None)
    ap.add_argument("--results-dir", default=None)
    ap.add_argument("--checkpoints-dir", default=None)
    a = ap.parse_args()
    cfg = apply_overrides(load_config(a.config), **{k: getattr(a, k) for k in OVERRIDES})
    m = train_client(cfg, a.language, a.clients_dir, a.output_tag)
    print(json.dumps({k: m[k] for k in ("language", "model", "device", "initial_validation_loss", "validation_loss",
                                        "perplexity", "lora_updated", "trainable_parameters", "total_parameters")},
                     indent=2))


if __name__ == "__main__":
    main()
