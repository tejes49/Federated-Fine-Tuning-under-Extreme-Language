"""Federated LoRA fine-tuning with FedAvg (no clustering) across language clients.

Run:  python -m experiments.run_fedavg --config configs/baseline.yaml
      python -m experiments.run_fedavg --config configs/dev.yaml --rounds 3 --local-steps 5 \
             --clients-dir /some/dir --languages en hi ta te ml
Needs data/clients/<id>/{train,val}.jsonl (python -m data.prepare_data).

Per round: (a) the global LoRA adapter is loaded into each active client's freshly built LoRA model,
(b) clients/local_train.train_client trains it for `federation.local_steps` optimizer steps,
(c) the trained client adapters are collected and averaged with server.fedavg.aggregate_lora_adapters
(weights = client training-set size, or uniform), (d) the average becomes the new global adapter,
(e) the global adapter is evaluated on every language's validation set.

`train_client` is used unchanged. It builds its own model, so the global adapter is injected by wrapping the
`attach_lora` name inside clients.local_train for the duration of the call (see `injected_global_adapter`),
and the trained adapter is read back from that same model object. Optimizer state (AdamW) is local to a
round and is reset each round, as in plain FedAvg.

Outputs: results/fedavg/metrics.json, checkpoints/fedavg/global_adapter/ (PEFT format) and
checkpoints/fedavg/global_adapter.pt (raw LoRA tensors). Only measured values are written.
"""
import argparse
import contextlib
import copy
import gc
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
from torch.utils.data import DataLoader

import clients.local_train as lt
from clients.local_train import (evaluate, make_collate, perplexity, read_texts, resolve_client_id, tokenize,
                                 train_client)
from models.base_model import TINY_OFFLINE, load_base_model, load_tokenizer
from models.lora import adapter_state, attach_lora, save_adapter
from server.fedavg import adapter_num_bytes, aggregate_lora_adapters
from utils.checkpoints import load_config, save_json, save_state
from utils.config_overrides import OVERRIDES, apply_overrides
from utils.logging import get_logger
from utils.reproducibility import set_seed

log = get_logger()
DEFAULT_LANGUAGES = ["en", "hi", "ta", "te", "ml"]


# ---------- adapter plumbing ----------
def load_adapter_state(peft_model, state: Dict[str, torch.Tensor]) -> None:
    """Copy `state` (names as produced by models.lora.adapter_state) into the model's LoRA tensors."""
    params = {n: p for n, p in peft_model.named_parameters() if "lora_" in n}
    if set(params) != set(state):
        missing, extra = sorted(set(params) - set(state)), sorted(set(state) - set(params))
        raise ValueError(f"adapter/model mismatch: missing {missing[:3]}, unexpected {extra[:3]}")
    with torch.no_grad():
        for n, p in params.items():
            p.copy_(state[n].to(device=p.device, dtype=p.dtype))


@contextlib.contextmanager
def injected_global_adapter(global_state: Dict[str, torch.Tensor], holder: dict):
    """While active, clients.local_train.attach_lora loads `global_state` into every model it builds and
    stores the model in holder['model'], so the trained adapter can be read after train_client returns."""
    original = lt.attach_lora

    def attach_and_load(model, lora_cfg, target_modules=None):
        peft_model = original(model, lora_cfg, target_modules)
        load_adapter_state(peft_model, global_state)
        holder["model"] = peft_model
        return peft_model

    lt.attach_lora = attach_and_load
    try:
        yield
    finally:
        lt.attach_lora = original


def client_cfg(cfg: dict, local_steps: int, seed: int, results_dir: Path, ckpt_dir: Path) -> dict:
    c = copy.deepcopy(cfg)
    c["seed"] = seed
    c["train"]["max_steps"] = local_steps
    c["train"]["eval_every_steps"] = None
    c["output"] = {**c.get("output", {}), "results_dir": str(results_dir), "checkpoints_dir": str(ckpt_dir)}
    return c


# ---------- global evaluation ----------
class GlobalEvaluator:
    """Holds one LoRA model and the tokenized validation set of every language; evaluates any adapter state."""

    def __init__(self, cfg: dict, languages: List[str], clients_dir: str):
        mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
        set_seed(cfg["seed"])
        self.tok = load_tokenizer(mcfg)
        collate = make_collate(self.tok.pad_token_id)
        self.loaders, self.ids = {}, {}
        for lang in languages:
            cid = resolve_client_id(lang, dcfg.get("data_config"))
            val = tokenize(self.tok, read_texts(Path(clients_dir) / cid / "val.jsonl",
                                                dcfg.get("max_validation_samples")), mcfg["max_seq_len"])
            if not val:
                raise ValueError(f"[{cid}] no usable validation sequences")
            self.loaders[cid] = DataLoader(val, batch_size=tcfg["batch_size"], shuffle=False, collate_fn=collate)
        base, self.device = load_base_model(mcfg, self.tok, cfg.get("device", "auto"), cfg.get("precision", "auto"))
        self.model = attach_lora(base, cfg["lora"], mcfg.get("lora_target_modules"))

    def initial_state(self) -> Dict[str, torch.Tensor]:
        return adapter_state(self.model)

    def evaluate(self, state: Dict[str, torch.Tensor]) -> Dict[str, dict]:
        load_adapter_state(self.model, state)
        return {cid: evaluate(self.model, loader, self.device) for cid, loader in self.loaders.items()}

    def save(self, state: Dict[str, torch.Tensor], ckpt_dir: Path) -> None:
        load_adapter_state(self.model, state)
        save_adapter(self.model, str(ckpt_dir / "global_adapter"))
        save_state(state, str(ckpt_dir / "global_adapter.pt"))


def summarize_eval(per_lang: Dict[str, dict]) -> dict:
    """Per-language loss/ppl plus token-weighted and unweighted-over-languages means."""
    toks = sum(v["validation_tokens"] for v in per_lang.values())
    loss_tok = sum(v["validation_loss"] * v["validation_tokens"] for v in per_lang.values()) / toks
    loss_mean = sum(v["validation_loss"] for v in per_lang.values()) / len(per_lang)
    return {"per_language": per_lang,
            "mean_loss_token_weighted": loss_tok, "mean_perplexity_token_weighted": perplexity(loss_tok),
            "mean_loss_over_languages": loss_mean, "mean_perplexity_over_languages": perplexity(loss_mean)}


# ---------- orchestration ----------
def run_fedavg(cfg: dict, languages: Optional[List[str]] = None, clients_dir: Optional[str] = None,
               rounds: Optional[int] = None, local_steps: Optional[int] = None,
               clients_per_round: Optional[int] = None, weighting: str = "samples",
               keep_client_artifacts: bool = False) -> dict:
    fed = cfg.get("federation", {})
    languages = languages or DEFAULT_LANGUAGES
    rounds = rounds if rounds is not None else fed.get("num_rounds", 10)
    local_steps = local_steps if local_steps is not None else fed.get("local_steps", 20)
    cpr = clients_per_round if clients_per_round is not None else fed.get("clients_per_round")
    if weighting not in ("samples", "uniform"):
        raise ValueError("weighting must be 'samples' or 'uniform'")
    if rounds < 1 or local_steps < 1:
        raise ValueError("rounds and local_steps must be >= 1")
    if cpr is not None and not 1 <= cpr <= len(languages):
        raise ValueError(f"clients_per_round must be in [1, {len(languages)}] or null")
    clients_dir = clients_dir or cfg["data"]["clients_dir"]
    ocfg = cfg["output"]
    out_root = Path(ocfg.get("fedavg_results_dir", "results/fedavg"))
    ckpt_root = Path(ocfg.get("fedavg_checkpoints_dir", "checkpoints/fedavg"))
    scratch = ckpt_root / "_client_tmp"
    seed = cfg["seed"]
    rng = random.Random(seed)
    cids = [resolve_client_id(l, cfg["data"].get("data_config")) for l in languages]

    evaluator = GlobalEvaluator(cfg, languages, clients_dir)
    global_state = evaluator.initial_state()           # seeded LoRA init (A random, B zero)
    down_bytes_per_client = adapter_num_bytes(global_state)
    log.info(f"FedAvg: {len(cids)} clients {cids}, rounds={rounds}, local_steps={local_steps}, "
             f"clients/round={cpr or 'all'}, adapter={down_bytes_per_client/1e6:.3f} MB")

    t_start = time.time()
    initial_eval = summarize_eval(evaluator.evaluate(global_state))
    history, comm_total = [], 0
    for r in range(1, rounds + 1):
        t0 = time.time()
        active = sorted(rng.sample(range(len(cids)), cpr)) if cpr else list(range(len(cids)))
        trained, sizes, client_info = [], [], []
        for ci in active:
            cid, lang = cids[ci], languages[ci]
            ccfg = client_cfg(cfg, local_steps, seed + 1000 * r + ci, out_root / "client_runs",
                              scratch)
            holder: dict = {}
            with injected_global_adapter(global_state, holder):
                m = train_client(ccfg, lang, clients_dir, output_tag=f"round{r:03d}_{cid}")
            state = adapter_state(holder.pop("model"))
            trained.append(state); sizes.append(m["n_train_samples"])
            client_info.append({"client_id": cid, "n_train_samples": m["n_train_samples"],
                                "optimizer_steps": m["optimizer_steps"],
                                "local_initial_val_loss": m["initial_validation_loss"],
                                "local_final_val_loss": m["validation_loss"],
                                "local_first_train_loss": m["first_train_loss"],
                                "local_last_train_loss": m["last_train_loss"],
                                "lora_updated": m["lora_updated"], "train_seconds": m["train_seconds"],
                                "upload_bytes": adapter_num_bytes(state)})
            gc.collect()
        w = sizes if weighting == "samples" else None
        global_state = aggregate_lora_adapters(trained, w)
        norm_w = [s / sum(sizes) for s in sizes] if w else [1 / len(sizes)] * len(sizes)
        for info, nw in zip(client_info, norm_w):
            info["aggregation_weight"] = nw
        ev = summarize_eval(evaluator.evaluate(global_state))
        up = sum(c["upload_bytes"] for c in client_info)
        down = down_bytes_per_client * len(active)
        comm_total += up + down
        rec = {"round": r, "active_clients": [cids[i] for i in active], "clients": client_info,
               "global_eval": ev, "round_seconds": round(time.time() - t0, 2),
               "communication": {"upload_bytes": up, "download_bytes": down, "total_bytes": up + down,
                                 "cumulative_bytes": comm_total}}
        history.append(rec)
        log.info(f"round {r}/{rounds}: mean val loss {ev['mean_loss_over_languages']:.4f} "
                 f"({rec['round_seconds']}s) " +
                 " ".join(f"{c}={v['perplexity']:.1f}" for c, v in ev["per_language"].items()))
        _write_metrics(out_root, cfg, languages, cids, rounds, local_steps, cpr, weighting, initial_eval,
                       history, comm_total, t_start, finished=(r == rounds))

    evaluator.save(global_state, ckpt_root)
    if not keep_client_artifacts:
        shutil.rmtree(scratch, ignore_errors=True)
    save_json(cfg, str(out_root / "config_used.json"))
    return load_metrics(out_root)


def _write_metrics(out_root, cfg, languages, cids, rounds, local_steps, cpr, weighting, initial_eval,
                   history, comm_total, t_start, finished):
    save_json({
        "method": "fedavg_lora", "finished": finished, "model": cfg["model"]["name"],
        "model_is_random_init": cfg["model"]["name"] == TINY_OFFLINE, "seed": cfg["seed"],
        "languages": languages, "client_ids": cids, "rounds_planned": rounds, "rounds_completed": len(history),
        "local_steps": local_steps, "clients_per_round": cpr, "weighting": weighting, "lora": cfg["lora"],
        "train": cfg["train"], "initial_global_eval": initial_eval, "rounds": history,
        "final_global_eval": history[-1]["global_eval"] if history else None,
        "total_communication_bytes": comm_total, "total_seconds": round(time.time() - t_start, 2),
    }, str(Path(out_root) / "metrics.json"))


def load_metrics(out_root: Path) -> dict:
    import json
    with open(Path(out_root) / "metrics.json", encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="FedAvg + LoRA across language clients")
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--languages", nargs="+", default=None, help="default: en hi ta te ml")
    ap.add_argument("--clients-dir", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--local-steps", type=int, default=None)
    ap.add_argument("--clients-per-round", type=int, default=None)
    ap.add_argument("--weighting", choices=["samples", "uniform"], default="samples")
    ap.add_argument("--keep-client-artifacts", action="store_true")
    ap.add_argument("--fedavg-results-dir", default=None)
    ap.add_argument("--fedavg-checkpoints-dir", default=None)
    for name, typ in (("model", str), ("max_seq_len", int), ("device", str), ("precision", str), ("seed", int),
                      ("batch_size", int), ("grad_accum", int), ("lr", float), ("lora_r", int),
                      ("lora_alpha", int), ("lora_dropout", float), ("max_train_samples", int),
                      ("max_val_samples", int)):
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=typ, default=None)
    a = ap.parse_args()
    cfg = apply_overrides(load_config(a.config), **{k: getattr(a, k, None) for k in OVERRIDES})
    cfg.setdefault("output", {})
    if a.fedavg_results_dir:
        cfg["output"]["fedavg_results_dir"] = a.fedavg_results_dir
    if a.fedavg_checkpoints_dir:
        cfg["output"]["fedavg_checkpoints_dir"] = a.fedavg_checkpoints_dir
    m = run_fedavg(cfg, a.languages, a.clients_dir, a.rounds, a.local_steps, a.clients_per_round,
                   a.weighting, a.keep_client_artifacts)
    fin = m["final_global_eval"]
    print({"rounds": m["rounds_completed"], "total_communication_bytes": m["total_communication_bytes"],
           "initial_mean_loss": m["initial_global_eval"]["mean_loss_over_languages"],
           "final_mean_loss": fin["mean_loss_over_languages"],
           "final_perplexity": {k: v["perplexity"] for k, v in fin["per_language"].items()}})


if __name__ == "__main__":
    main()
