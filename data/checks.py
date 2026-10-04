"""Sanity checks for prepared client data and its tokenization.

Pure Python (no torch import): the tokenizer is passed in, batches only need .shape/.min()/.max()/... .
CLI (needs transformers, and the data from `python -m data.prepare_data`):
    python -m data.checks --language tamil [--config configs/single_client.yaml] [--max-samples 100]
"""
import argparse
import json
from pathlib import Path
from typing import List, Optional

from data.prepare_data import script_fraction


def check_texts(texts: List[str], script: str, min_script_fraction: float = 0.5) -> dict:
    """Empty / duplicate passages and possible language mixing (share of letters in the expected script)."""
    nonempty = [t for t in texts if isinstance(t, str) and t.strip()]
    fracs = [script_fraction(t, script) for t in nonempty]
    lens = sorted(len(t) for t in nonempty)
    return {
        "n_texts": len(texts),
        "n_empty": len(texts) - len(nonempty),
        "n_duplicates": len(nonempty) - len(set(nonempty)),
        "expected_script": script,
        "mean_script_fraction": round(sum(fracs) / len(fracs), 4) if fracs else None,
        "n_below_script_threshold": sum(1 for f in fracs if f < min_script_fraction),
        "script_threshold": min_script_fraction,
        "chars_min": lens[0] if lens else None,
        "chars_max": lens[-1] if lens else None,
    }


def check_tokenized(seqs: List[List[int]], max_len: int, vocab_size: int) -> dict:
    """Length and id-range checks on already-tokenized sequences."""
    lens = [len(s) for s in seqs]
    return {
        "n_sequences": len(seqs),
        "len_min": min(lens) if lens else None,
        "len_max": max(lens) if lens else None,
        "len_mean": round(sum(lens) / len(lens), 2) if lens else None,
        "n_over_max_len": sum(1 for n in lens if n > max_len),
        "n_at_max_len": sum(1 for n in lens if n == max_len),   # truncated (or exactly full)
        "n_shorter_than_2": sum(1 for n in lens if n < 2),      # no next-token target
        "n_invalid_token_ids": sum(1 for s in seqs for i in s if i < 0 or i >= vocab_size),
    }


def tokenizer_fertility(tokenizer, texts: List[str]) -> dict:
    """How many tokens the tokenizer spends per word / character (before truncation)."""
    toks = chars = words = unk = 0
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for t in texts:
        ids = tokenizer(t, add_special_tokens=False)["input_ids"]
        toks += len(ids)
        chars += len(t)
        words += len(t.split())
        if unk_id is not None:
            unk += sum(1 for i in ids if i == unk_id)
    return {"tokens_per_word": round(toks / words, 3) if words else None,
            "tokens_per_char": round(toks / chars, 3) if chars else None,
            "unk_tokens": unk, "unk_rate": round(unk / toks, 6) if toks else None}


def check_batch(batch: dict, vocab_size: int) -> List[str]:
    """Return a list of problems (empty list = batch is fine). Expects right-padded 2-D int batches."""
    ids, mask = batch["input_ids"], batch["attention_mask"]
    problems = []
    if ids.ndim != 2:
        problems.append(f"input_ids must be 2-D, got shape {tuple(ids.shape)}")
    if tuple(ids.shape) != tuple(mask.shape):
        problems.append(f"input_ids {tuple(ids.shape)} and attention_mask {tuple(mask.shape)} differ")
        return problems
    if int(ids.min()) < 0 or int(ids.max()) >= vocab_size:
        problems.append("token id outside [0, vocab_size)")
    if not bool(((mask == 0) | (mask == 1)).all()):
        problems.append("attention_mask contains values other than 0/1")
    if bool((mask.sum(1) < 2).any()):
        problems.append("a sequence has fewer than 2 real tokens (no next-token target)")
    if bool((mask[:, 1:] > mask[:, :-1]).any()):
        problems.append("padding is not right-aligned (a 0 is followed by a 1)")
    return problems


def _read_texts(path: Path, max_n: Optional[int]) -> List[str]:
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                texts.append(json.loads(line).get("text", ""))
            if max_n and len(texts) >= max_n:
                break
    return texts


def main():
    # heavy imports only here so the helpers above stay importable without torch/transformers
    from clients.local_train import make_collate, resolve_client_id, tokenize
    from models.base_model import load_tokenizer
    from utils.checkpoints import load_config

    ap = argparse.ArgumentParser(description="Check prepared data + tokenization for one language")
    ap.add_argument("--language", required=True)
    ap.add_argument("--config", default="configs/single_client.yaml")
    ap.add_argument("--clients-dir", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--max-samples", type=int, default=200)
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.model:
        cfg["model"]["name"] = a.model
    cid = resolve_client_id(a.language, cfg["data"].get("data_config"))
    clients = {c["id"]: c for c in load_config(cfg["data"]["data_config"])["clients"]}
    script = clients[cid]["script"]
    cdir = Path(a.clients_dir or cfg["data"]["clients_dir"]) / cid
    tok = load_tokenizer(cfg["model"])
    report = {"language": a.language, "client_id": cid, "tokenizer": cfg["model"]["name"],
              "tokenizer_size": len(tok)}
    for split in ("train", "val"):
        texts = _read_texts(cdir / f"{split}.jsonl", a.max_samples)
        seqs = tokenize(tok, texts, cfg["model"]["max_seq_len"])
        rep = {"texts": check_texts(texts, script),
               "tokens": check_tokenized(seqs, cfg["model"]["max_seq_len"], len(tok)),
               "fertility": tokenizer_fertility(tok, [t for t in texts if t.strip()])}
        if seqs:
            rep["batch_problems"] = check_batch(make_collate(tok.pad_token_id)(seqs[:8]), len(tok))
        report[split] = rep
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
