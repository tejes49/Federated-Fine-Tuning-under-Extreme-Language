"""Proposed method: language-aware clustering + per-cluster FedAvg + alpha-mixed global adapter + tokenizer alignment.

Run:  python -m experiments.run_proposed --config configs/proposed.yaml
      python -m experiments.run_proposed --config configs/proposed_dev.yaml --clients-dir /some/dir
Needs data/clients/<id>/{train,val}.jsonl (python -m data.prepare_data). proposed.yaml uses the LaBSE encoder
(Hugging Face download); proposed_dev.yaml uses the offline hashing encoder and the tiny random-init model.

Setup (once): language fingerprints from a sample of each client's training passages -> language clusters
(clustering/language_cluster.py); per-client linear subword projection fitted against the encoder's sentence
embeddings (models/tokenizer_align.py, diagnostic only - see that module).
Each round: active clients start from their CLUSTER adapter (federation.cluster_init=cluster) or from the GLOBAL
adapter (=global), train with the unchanged `train_client`, adapters are FedAvg-ed within each cluster, and the global
adapter is updated with  alpha * sum_c N_c/N * cluster_c + (1 - alpha) * global_prev  (server/cluster_fedavg.py).
Evaluation per round, per language: perplexity under the language's cluster adapter and under the global adapter,
fairness gap = max PPL - min PPL over languages, communication bytes.

Outputs: results/proposed/metrics.json, checkpoints/proposed/{cluster_<k>,global_adapter}/ (+ clusters.json).
"""
import argparse
import gc
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from clients.local_train import LANG_ALIASES, read_texts, resolve_client_id, tokenize, train_client
from clustering.language_cluster import cluster_clients, language_fingerprint, make_embedder
from experiments.run_fedavg import (GlobalEvaluator, client_cfg, injected_global_adapter, load_adapter_state,
                                    summarize_eval)
from clients.local_train import evaluate
from models.base_model import TINY_OFFLINE
from models.lora import adapter_state, save_adapter
from models.tokenizer_align import fit_client_projection
from server.cluster_fedavg import cluster_round, save_cluster_checkpoints
from server.fedavg import adapter_num_bytes
from utils.checkpoints import load_config, save_json, save_state
from utils.config_overrides import OVERRIDES, apply_overrides
from utils.logging import get_logger

log = get_logger()
DEFAULT_LANGUAGES = ["en", "hi", "ta", "te", "ml"]
DEFAULT_CLUSTERING = {"encoder": "sentence-transformers/LaBSE", "fingerprint_samples": 100, "similarity_threshold": 0.5,
                      "family_weight": 0.5, "n_clusters": None, "center": True, "families": None, "batch_size": 32}
DEFAULT_ALIGNMENT = {"enabled": True, "ridge": 1e-2}


def fairness_gap(per_language: Dict[str, dict]) -> dict:
    """max PPL - min PPL over language cohorts (plus which languages, and the max/min ratio)."""
    ppl = {k: v["perplexity"] for k, v in per_language.items()}
    hi, lo = max(ppl, key=ppl.get), min(ppl, key=ppl.get)
    return {"gap": ppl[hi] - ppl[lo], "max_perplexity": ppl[hi], "max_language": hi,
            "min_perplexity": ppl[lo], "min_language": lo, "ratio": ppl[hi] / ppl[lo]}


class ProposedEvaluator(GlobalEvaluator):
    """GlobalEvaluator plus evaluation restricted to a subset of language cohorts, and PEFT-format saving."""

    def evaluate_subset(self, state, cids: List[str]) -> Dict[str, dict]:
        load_adapter_state(self.model, state)
        return {c: evaluate(self.model, self.loaders[c], self.device) for c in cids}

    def save_peft(self, state, path: Path) -> None:
        load_adapter_state(self.model, state)
        save_adapter(self.model, str(path))


def declared_language(cid: str, data_config: Optional[str]) -> str:
    """Declared language code of a client id (from configs/data.yaml when present, else the id itself)."""
    if data_config and Path(data_config).exists():
        for c in load_config(data_config).get("clients", []):
            if c["id"] == cid:
                return c["lang"]
    return cid


def evaluate_assignment(evaluator, cluster_states, global_state, clusters, cids) -> dict:
    """Per-language eval under (a) the language's own cluster adapter, (b) the global adapter."""
    per_cluster = {}
    for k, members in clusters.items():
        per_cluster.update(evaluator.evaluate_subset(cluster_states[k], members))
    order = {c: per_cluster[c] for c in cids}
    glob = evaluator.evaluate_subset(global_state, cids)
    ce, ge = summarize_eval(order), summarize_eval(glob)
    ce["fairness_gap"], ge["fairness_gap"] = fairness_gap(order), fairness_gap(glob)
    return {"cluster_adapter": ce, "global_adapter": ge}


def run_proposed(cfg: dict, languages: Optional[List[str]] = None, clients_dir: Optional[str] = None,
                 rounds: Optional[int] = None, local_steps: Optional[int] = None,
                 clients_per_round: Optional[int] = None, alpha: Optional[float] = None,
                 cluster_init: Optional[str] = None, keep_client_artifacts: bool = False,
                 embedder=None) -> dict:
    fed = cfg.get("federation", {})
    ccfg_ = {**DEFAULT_CLUSTERING, **(cfg.get("language_clustering") or {})}
    acfg = {**DEFAULT_ALIGNMENT, **(cfg.get("alignment") or {})}
    languages = languages or DEFAULT_LANGUAGES
    rounds = rounds if rounds is not None else fed.get("num_rounds", 10)
    local_steps = local_steps if local_steps is not None else fed.get("local_steps", 20)
    cpr = clients_per_round if clients_per_round is not None else fed.get("clients_per_round")
    alpha = alpha if alpha is not None else fed.get("mixing_alpha", 0.5)
    cluster_init = cluster_init or fed.get("cluster_init", "cluster")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if cluster_init not in ("cluster", "global"):
        raise ValueError("cluster_init must be 'cluster' or 'global'")
    if rounds < 1 or local_steps < 1:
        raise ValueError("rounds and local_steps must be >= 1")
    if cpr is not None and not 1 <= cpr <= len(languages):
        raise ValueError(f"clients_per_round must be in [1, {len(languages)}] or null")
    dcfg, mcfg = cfg["data"], cfg["model"]
    clients_dir = clients_dir or dcfg["clients_dir"]
    ocfg = cfg["output"]
    out_root = Path(ocfg.get("proposed_results_dir", "results/proposed"))
    ckpt_root = Path(ocfg.get("proposed_checkpoints_dir", "checkpoints/proposed"))
    scratch = ckpt_root / "_client_tmp"
    seed = cfg["seed"]
    rng = random.Random(seed)
    cids = [resolve_client_id(l, dcfg.get("data_config")) for l in languages]
    declared = {c: declared_language(c, dcfg.get("data_config")) for c in cids}
    t_start = time.time()

    evaluator = ProposedEvaluator(cfg, languages, clients_dir)
    tok = evaluator.tok

    # ---- setup: fingerprints, clusters, tokenizer alignment, client sizes ----
    embedder = embedder or make_embedder(ccfg_["encoder"], batch_size=ccfg_["batch_size"])
    n_fp = ccfg_["fingerprint_samples"]
    train_texts = {c: read_texts(Path(clients_dir) / c / "train.jsonl", n_fp) for c in cids}
    train_emb = {c: embedder.encode(t) for c, t in train_texts.items()}
    fingerprints = {c: language_fingerprint(e) for c, e in train_emb.items()}
    assignment = cluster_clients(cids, declared, fingerprints, families=ccfg_["families"],
                                 similarity_threshold=ccfg_["similarity_threshold"],
                                 family_weight=ccfg_["family_weight"], n_clusters=ccfg_["n_clusters"],
                                 center=ccfg_["center"])
    clusters, cluster_of = assignment.clusters, assignment.cluster_of
    log.info(f"clusters: {clusters}")

    alignment = None
    if acfg["enabled"]:
        emb_matrix = evaluator.model.get_input_embeddings().weight.detach().cpu()
        alignment = {}
        for c in cids:
            val_texts = read_texts(Path(clients_dir) / c / "val.jsonl", n_fp)
            r = fit_client_projection(emb_matrix, tok, train_texts[c], train_emb[c], val_texts,
                                      embedder.encode(val_texts), mcfg["max_seq_len"], acfg["ridge"])
            save_state(r["projection"].state_dict(), str(ckpt_root / "alignment" / f"{c}.pt"))
            alignment[c] = {k: v for k, v in r.items() if k != "projection"}
            alignment[c]["d_in"], alignment[c]["d_common"] = r["projection"].d_in, r["projection"].d_common

    n_train = {c: len(tokenize(tok, read_texts(Path(clients_dir) / c / "train.jsonl", dcfg.get("max_train_samples")),
                               mcfg["max_seq_len"])) for c in cids}
    cluster_sizes = {k: float(sum(n_train[c] for c in members)) for k, members in clusters.items()}
    fp_dim = int(next(iter(fingerprints.values())).shape[0])
    setup_bytes = len(cids) * fp_dim * 4        # each client uploads one float32 fingerprint, once

    global_state = evaluator.initial_state()
    cluster_states = {k: {n: t.clone() for n, t in global_state.items()} for k in clusters}
    bytes_per_adapter = adapter_num_bytes(global_state)
    log.info(f"proposed: clients={cids}, clusters={len(clusters)}, rounds={rounds}, local_steps={local_steps}, "
             f"alpha={alpha}, cluster_init={cluster_init}, encoder={embedder.name}")

    initial_eval = evaluate_assignment(evaluator, cluster_states, global_state, clusters, cids)
    history, comm_total = [], setup_bytes
    meta = {"model": mcfg["name"], "model_is_random_init": mcfg["name"] == TINY_OFFLINE, "seed": seed,
            "languages": languages, "client_ids": cids, "declared_languages": declared,
            "rounds_planned": rounds, "local_steps": local_steps, "clients_per_round": cpr, "alpha": alpha,
            "cluster_init": cluster_init, "lora": cfg["lora"], "train": cfg["train"],
            "fingerprint_encoder": embedder.name, "encoder_is_semantic": bool(getattr(embedder, "is_semantic", True)),
            "clustering": {**assignment.to_dict(), "cluster_sizes": {str(k): v for k, v in cluster_sizes.items()},
                           "client_train_samples": n_train},
            "tokenizer_alignment": alignment, "setup_communication_bytes": setup_bytes}

    for r in range(1, rounds + 1):
        t0 = time.time()
        active = sorted(rng.sample(range(len(cids)), cpr)) if cpr else list(range(len(cids)))
        trained, weights, ks, info = [], [], [], []
        for ci in active:
            cid, k = cids[ci], cluster_of[cids[ci]]
            start = cluster_states[k] if cluster_init == "cluster" else global_state
            ccfg = client_cfg(cfg, local_steps, seed + 1000 * r + ci, out_root / "client_runs", scratch)
            holder: dict = {}
            with injected_global_adapter(start, holder):
                m = train_client(ccfg, languages[ci], clients_dir, output_tag=f"round{r:03d}_{cid}")
            st = adapter_state(holder.pop("model"))
            trained.append(st); weights.append(m["n_train_samples"]); ks.append(k)
            info.append({"client_id": cid, "cluster": k, "n_train_samples": m["n_train_samples"],
                         "optimizer_steps": m["optimizer_steps"], "local_initial_val_loss": m["initial_validation_loss"],
                         "local_final_val_loss": m["validation_loss"], "lora_updated": m["lora_updated"],
                         "train_seconds": m["train_seconds"], "upload_bytes": adapter_num_bytes(st)})
            gc.collect()
        cluster_states, global_state = cluster_round(trained, weights, ks, cluster_states, cluster_sizes,
                                                     global_state, alpha)
        ev = evaluate_assignment(evaluator, cluster_states, global_state, clusters, cids)
        up, down = sum(i["upload_bytes"] for i in info), bytes_per_adapter * len(active)
        comm_total += up + down
        rec = {"round": r, "active_clients": [cids[i] for i in active], "clients": info, "eval": ev,
               "round_seconds": round(time.time() - t0, 2),
               "communication": {"upload_bytes": up, "download_bytes": down, "total_bytes": up + down,
                                 "cumulative_bytes": comm_total}}
        history.append(rec)
        fa = ev["cluster_adapter"]["fairness_gap"]["gap"], ev["global_adapter"]["fairness_gap"]["gap"]
        log.info(f"round {r}/{rounds}: fairness gap cluster={fa[0]:.2f} global={fa[1]:.2f} ({rec['round_seconds']}s)")
        _write(out_root, meta, initial_eval, history, comm_total, t_start, finished=(r == rounds))

    save_cluster_checkpoints(str(ckpt_root), cluster_states, global_state, clusters, cluster_sizes, alpha,
                             {"cluster_of": cluster_of, "cluster_init": cluster_init})
    for k, st in cluster_states.items():
        evaluator.save_peft(st, ckpt_root / f"cluster_{k}")
    evaluator.save_peft(global_state, ckpt_root / "global_adapter")
    if not keep_client_artifacts:
        shutil.rmtree(scratch, ignore_errors=True)
    save_json(cfg, str(out_root / "config_used.json"))
    import json
    return json.loads((out_root / "metrics.json").read_text(encoding="utf-8"))


def _write(out_root, meta, initial_eval, history, comm_total, t_start, finished):
    save_json({"method": "proposed_language_aware", "finished": finished, **meta, "rounds_completed": len(history),
               "initial_eval": initial_eval, "rounds": history,
               "final_eval": history[-1]["eval"] if history else None,
               "total_communication_bytes": comm_total, "total_seconds": round(time.time() - t_start, 2)},
              str(Path(out_root) / "metrics.json"))


def main():
    ap = argparse.ArgumentParser(description="Proposed method: language-aware clustering + alpha-mixing")
    ap.add_argument("--config", default="configs/proposed.yaml")
    ap.add_argument("--languages", nargs="+", default=None)
    ap.add_argument("--clients-dir", default=None)
    ap.add_argument("--rounds", type=int, default=None)
    ap.add_argument("--local-steps", type=int, default=None)
    ap.add_argument("--clients-per-round", type=int, default=None)
    ap.add_argument("--alpha", type=float, default=None, help="mixing coefficient in [0,1]")
    ap.add_argument("--cluster-init", choices=["cluster", "global"], default=None)
    ap.add_argument("--encoder", default=None, help="sentence-transformers name/path, or offline-hash")
    ap.add_argument("--n-clusters", type=int, default=None)
    ap.add_argument("--similarity-threshold", type=float, default=None)
    ap.add_argument("--no-alignment", action="store_true")
    ap.add_argument("--keep-client-artifacts", action="store_true")
    ap.add_argument("--proposed-results-dir", default=None)
    ap.add_argument("--proposed-checkpoints-dir", default=None)
    for name, typ in (("model", str), ("max_seq_len", int), ("device", str), ("precision", str), ("seed", int),
                      ("batch_size", int), ("grad_accum", int), ("lr", float), ("lora_r", int),
                      ("lora_alpha", int), ("lora_dropout", float), ("max_train_samples", int),
                      ("max_val_samples", int)):
        ap.add_argument("--" + name.replace("_", "-"), dest=name, type=typ, default=None)
    a = ap.parse_args()
    cfg = apply_overrides(load_config(a.config), **{k: getattr(a, k, None) for k in OVERRIDES})
    cfg.setdefault("output", {})
    cfg.setdefault("language_clustering", {})
    cfg.setdefault("alignment", {})
    for key, val in (("proposed_results_dir", a.proposed_results_dir), ("proposed_checkpoints_dir", a.proposed_checkpoints_dir)):
        if val:
            cfg["output"][key] = val
    if a.encoder:
        cfg["language_clustering"]["encoder"] = a.encoder
    if a.n_clusters is not None:
        cfg["language_clustering"]["n_clusters"] = a.n_clusters
    if a.similarity_threshold is not None:
        cfg["language_clustering"]["similarity_threshold"] = a.similarity_threshold
    if a.no_alignment:
        cfg["alignment"]["enabled"] = False
    m = run_proposed(cfg, a.languages, a.clients_dir, a.rounds, a.local_steps, a.clients_per_round, a.alpha,
                     a.cluster_init, a.keep_client_artifacts)
    f = m["final_eval"]
    print({"clusters": m["clustering"]["clusters"], "rounds": m["rounds_completed"],
           "fairness_gap_cluster_adapter": f["cluster_adapter"]["fairness_gap"]["gap"],
           "fairness_gap_global_adapter": f["global_adapter"]["fairness_gap"]["gap"],
           "total_communication_bytes": m["total_communication_bytes"]})


if __name__ == "__main__":
    main()
