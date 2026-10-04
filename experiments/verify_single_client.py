"""End-to-end verification of the single-client LoRA pipeline (real model, real prepared data).

    python -m data.prepare_data --config configs/data.yaml            # once, needs internet
    python -m experiments.verify_single_client --config configs/dev_real.yaml

Stages: tokenizer -> model + forward pass -> per-language data checks + base-model loss -> LoRA attach /
freezing / forward / backward -> tiny training run (train_client) -> adapter reload + independent
perplexity cross-check + adapter size.

Every value in results/verification/report.json is measured by this run. A stage that fails records its
exact error (status "error") and the script carries on; nothing is filled in or estimated.
"""
import argparse
import contextlib
import copy
import gc
import json
import math
import time
import traceback
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from clients.local_train import (batch_loss_sum, build_labels, evaluate, make_collate, read_texts,
                                 resolve_client_id, tokenize, train_client)
from data.checks import check_batch, check_texts, check_tokenized, tokenizer_fertility
from models.base_model import check_tokenizer_model_compat, load_base_model, load_tokenizer
from models.lora import attach_lora, count_parameters, load_adapter, lora_summary, verify_lora_setup
from utils.checkpoints import load_config, save_json
from utils.config_overrides import apply_overrides
from utils.logging import get_logger
from utils.reproducibility import set_seed
from utils.resources import describe_environment, peak_memory_gb

log = get_logger()
ALL_LANGUAGES = ["english", "hindi", "tamil", "telugu", "malayalam"]


@contextlib.contextmanager
def stage(container: dict, name: str):
    """Run a block; record status/error/seconds under container[name] and never let it crash the script."""
    rec = {"status": "error"}
    container[name] = rec
    t0 = time.time()
    try:
        yield rec
        rec["status"] = "ok"
    except Exception as e:  # noqa: BLE001 - the point is to record the exact failure
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc(limit=6)
        log.error(f"stage '{name}' failed: {rec['error']}")
    finally:
        rec["seconds"] = round(time.time() - t0, 2)


def require(obj, what: str):
    if obj is None:
        raise RuntimeError(f"needs '{what}', but the earlier stage that creates it failed")


def free(*objs):
    del objs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/dev_real.yaml")
    ap.add_argument("--languages", nargs="+", default=ALL_LANGUAGES)
    ap.add_argument("--train-language", default=None, help="default: first language whose data stage succeeded")
    ap.add_argument("--clients-dir", default=None)
    ap.add_argument("--report-dir", default="results/verification")
    ap.add_argument("--n-samples", type=int, default=16, help="validation passages per language for the checks")
    for flag, typ in (("--model", str), ("--device", str), ("--precision", str), ("--max-steps", int),
                      ("--max-seq-len", int), ("--batch-size", int), ("--seed", int),
                      ("--max-train-samples", int), ("--max-val-samples", int)):
        ap.add_argument(flag, type=typ, default=None)
    a = ap.parse_args()

    cfg = apply_overrides(load_config(a.config), model=a.model, device=a.device, precision=a.precision,
                          max_steps=a.max_steps, max_seq_len=a.max_seq_len, batch_size=a.batch_size,
                          seed=a.seed, max_train_samples=a.max_train_samples, max_val_samples=a.max_val_samples)
    mcfg, dcfg, tcfg = cfg["model"], cfg["data"], cfg["train"]
    clients_dir = Path(a.clients_dir or dcfg["clients_dir"])
    max_len, bs = mcfg["max_seq_len"], tcfg["batch_size"]
    scripts = {c["id"]: c["script"] for c in load_config(dcfg["data_config"])["clients"]}
    set_seed(cfg["seed"])

    report = {"config_file": a.config, "config_used": cfg, "languages_requested": a.languages,
              "environment": describe_environment()}
    tok = model = device = probe = peft = None

    # 1. tokenizer -----------------------------------------------------------------------------
    with stage(report, "1_tokenizer") as r:
        tok = load_tokenizer(mcfg)
        r.update(model_name=mcfg["name"], tokenizer_class=type(tok).__name__, vocab_size=len(tok),
                 pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, unk_token_id=tok.unk_token_id)

    # 2. model + forward pass ------------------------------------------------------------------
    with stage(report, "2_model_and_forward") as r:
        require(tok, "tokenizer")
        model, device = load_base_model(mcfg, tok, cfg.get("device", "auto"), cfg.get("precision", "auto"))
        r.update(device=device, dtype=str(model.dtype), model_type=model.config.model_type,
                 **check_tokenizer_model_compat(tok, model),
                 total_parameters=count_parameters(model)["total_parameters"])
        enc = tok("The quick brown fox", return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**enc)
        r["logits_shape"] = list(out.logits.shape)
        r["logits_all_finite"] = bool(torch.isfinite(out.logits).all())

    # 3. real data, per language, through the BASE model (no LoRA yet) --------------------------------
    langs = {}
    report["3_languages"] = langs
    for lang in a.languages:
        with stage(langs, lang) as r:
            require(model, "model")
            cid = resolve_client_id(lang, dcfg.get("data_config"))
            texts = read_texts(clients_dir / cid / "val.jsonl", a.n_samples)
            seqs = tokenize(tok, texts, max_len)
            coll = make_collate(tok.pad_token_id)
            batch = coll(seqs[:bs])
            r["client_id"] = cid
            r["texts"] = check_texts(texts, scripts[cid])
            r["tokens"] = check_tokenized(seqs, max_len, len(tok))
            r["fertility"] = tokenizer_fertility(tok, texts)
            r["batch_problems"] = check_batch(batch, len(tok))
            r["batch_shape"] = list(batch["input_ids"].shape)
            r["label_shape"] = list(build_labels(batch["input_ids"], batch["attention_mask"]).shape)
            r["padded_positions_in_batch"] = int((batch["attention_mask"] == 0).sum())
            r["base_model_eval"] = evaluate(model, DataLoader(seqs, batch_size=bs, collate_fn=coll), device)
            if probe is None:
                s, n = batch_loss_sum(model, batch, device)
                probe = {"lang": lang, "batch": batch, "base_loss": s.item() / n.item()}

    # 4. LoRA on the loaded model ------------------------------------------------------------------
    with stage(report, "4_lora") as r:
        require(model, "model")
        require(probe, "a language whose data stage succeeded")
        peft = attach_lora(model, cfg["lora"], mcfg.get("lora_target_modules"))
        r["lora"] = lora_summary(peft)
        r["parameters"] = count_parameters(peft)
        r["freezing"] = verify_lora_setup(peft)
        peft.eval()
        with torch.no_grad():
            s, n = batch_loss_sum(peft, probe["batch"], device)
        r["probe_language"] = probe["lang"]
        r["loss_at_init_with_lora"] = s.item() / n.item()
        r["loss_base_without_lora"] = probe["base_loss"]
        r["equals_base_at_init"] = math.isclose(r["loss_at_init_with_lora"], probe["base_loss"], rel_tol=1e-2)
        peft.train()
        peft.zero_grad()
        s, n = batch_loss_sum(peft, probe["batch"], device)
        (s / n).backward()
        grads_b = [p.grad for nm, p in peft.named_parameters() if "lora_B" in nm]
        r["backward_ok"] = bool(grads_b) and all(g is not None for g in grads_b)
        r["lora_B_grad_nonzero"] = any(g is not None and bool(g.abs().sum() > 0) for g in grads_b)
        r["frozen_params_with_grad"] = sum(1 for nm, p in peft.named_parameters()
                                           if "lora_" not in nm and p.grad is not None)
        peft.zero_grad()
    peft = model = None      # drop references so train_client's fresh model fits in memory
    free()

    # 5. tiny real training run (fresh model inside train_client) ------------------------------------
    metrics = None
    with stage(report, "5_tiny_training") as r:
        train_lang = a.train_language or (probe["lang"] if probe else a.languages[0])
        cid = resolve_client_id(train_lang, dcfg.get("data_config"))
        tag = f"verify_{cid}"
        metrics = train_client(copy.deepcopy(cfg), train_lang, a.clients_dir, output_tag=tag)
        keep = ("language", "model", "device", "base_dtype", "optimizer_steps", "batch_size", "grad_accum_steps",
                "max_seq_len", "n_train_samples", "n_validation_samples", "initial_validation_loss",
                "initial_perplexity", "validation_loss", "perplexity", "first_train_loss", "last_train_loss",
                "lora_updated", "lora_summary", "lora_setup_check", "trainable_parameters", "total_parameters",
                "trainable_percent", "train_seconds", "peak_memory", "adapter_path")
        r["metrics"] = {k: metrics[k] for k in keep}
        recs = [json.loads(line) for line in open(Path(cfg["output"]["results_dir"]) / tag / "log.jsonl",
                                                  encoding="utf-8")]
        r["step_losses"] = [[x["step"], x["train_loss"]] for x in recs if x["event"] == "step"]
        r["first_vs_last_train_loss_lower"] = metrics["last_train_loss"] < metrics["first_train_loss"]  # informational only

    # 6. adapter reload + independent perplexity cross-check ----------------------------------------
    with stage(report, "6_adapter_reload_and_perplexity") as r:
        require(metrics, "training metrics")
        cid = metrics["client_id"]
        val_seqs = tokenize(tok, read_texts(clients_dir / cid / "val.jsonl", dcfg.get("max_validation_samples")), max_len)
        coll = make_collate(tok.pad_token_id)
        base2, dev2 = load_base_model(mcfg, tok, cfg.get("device", "auto"), cfg.get("precision", "auto"))
        base_bytes = sum(p.numel() * p.element_size() for p in base2.parameters())
        tol = 1e-4 if next(base2.parameters()).dtype == torch.float32 else 3e-2
        m2 = load_adapter(base2, metrics["adapter_path"])
        m2.eval()
        ev = evaluate(m2, DataLoader(val_seqs, batch_size=bs, collate_fn=coll), dev2)
        r["trained_validation_loss"] = metrics["validation_loss"]
        r["reloaded_validation_loss"] = ev["validation_loss"]
        r["reload_matches_trained"] = math.isclose(ev["validation_loss"], metrics["validation_loss"], rel_tol=tol)
        # independent reference: one sequence at a time (no padding), plain log-softmax NLL
        tot, cnt = 0.0, 0
        with torch.no_grad():
            for ids in val_seqs:
                x = torch.tensor([ids], device=dev2)
                logp = torch.log_softmax(m2(input_ids=x).logits[0, :-1].float(), dim=-1)
                tot += -logp.gather(1, x[0, 1:, None]).sum().item()
                cnt += x.size(1) - 1
        r["reference_loss_unpadded"] = tot / cnt
        r["reference_perplexity"] = math.exp(tot / cnt) if tot / cnt < 700 else float("inf")
        r["reference_tokens"] = cnt
        r["batched_tokens"] = ev["validation_tokens"]
        r["reference_matches_batched_padded_eval"] = (cnt == ev["validation_tokens"] and
                                                      math.isclose(tot / cnt, ev["validation_loss"], rel_tol=tol))
        files = {f.name: f.stat().st_size for f in Path(metrics["adapter_path"]).iterdir()}
        r["adapter_files_bytes"] = files
        r["base_model_bytes"] = base_bytes
        r["adapter_weights_fraction_of_base"] = files.get("adapter_model.safetensors", 0) / base_bytes
        r["full_base_weights_saved_in_adapter_dir"] = any(n.startswith(("model", "pytorch_model")) for n in files)

    report["peak_memory_whole_script"] = peak_memory_gb()
    out = Path(a.report_dir) / "report.json"
    save_json(report, str(out))
    print(f"\nreport written to {out}")
    for name in ("1_tokenizer", "2_model_and_forward", "4_lora", "5_tiny_training", "6_adapter_reload_and_perplexity"):
        print(f"  {name}: {report[name]['status']}" + (f"  ({report[name]['error']})" if "error" in report[name] else ""))
    for lang, rec in langs.items():
        print(f"  language {lang}: {rec['status']}" + (f"  ({rec['error']})" if "error" in rec else ""))


if __name__ == "__main__":
    main()
