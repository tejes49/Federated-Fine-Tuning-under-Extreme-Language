"""Phase 2: build simulated federated clients from Wikipedia.

Run (Colab or local):  python -m data.prepare_data --config configs/data.yaml
Output: data/clients/<id>/{train,val,test}.jsonl, data/clients/manifest.json
Each client's text stays in its own folder; later phases only read its own folder.
"""
import argparse
import hashlib
import json
import random
import unicodedata
from pathlib import Path

from utils.checkpoints import load_config, save_json
from utils.logging import get_logger
from utils.reproducibility import set_seed

log = get_logger()


# ---------- pure, offline-testable helpers ----------
def assign_split(key: str, seed: int, test_pct: int, val_pct: int) -> str:
    """Deterministic article -> split by hash, so no article leaks across splits."""
    b = int(hashlib.md5(f"{seed}:{key}".encode()).hexdigest(), 16) % 100
    if b < test_pct:
        return "test"
    if b < test_pct + val_pct:
        return "val"
    return "train"


def extract_passages(text: str, min_chars: int, max_chars: int) -> list:
    """Split an article into paragraph passages; chunk long ones at whitespace."""
    out = []
    for para in text.split("\n"):
        para = para.strip()
        while len(para) > max_chars:
            cut = para.rfind(" ", 0, max_chars)
            cut = cut if cut > max_chars // 2 else max_chars
            out.append(para[:cut].strip())
            para = para[cut:].strip()
        if para:
            out.append(para)
    return [p for p in out if len(p) >= min_chars]


def script_fraction(text: str, script: str) -> float:
    """Fraction of alphabetic characters belonging to `script` (e.g. 'TAMIL')."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return 0.0
    hits = sum(1 for c in letters if unicodedata.name(c, "").startswith(script))
    return hits / len(letters)


def collect(articles, need: dict, seed: int, test_pct: int, val_pct: int,
            min_chars: int, max_chars: int, max_articles: int) -> dict:
    """Stream articles until each split has >= need[split] passages."""
    got = {"train": [], "val": [], "test": []}
    for n, art in enumerate(articles):
        if n >= max_articles or all(len(got[s]) >= need[s] for s in got):
            break
        key = str(art.get("id", art.get("title", n)))
        split = assign_split(key, seed, test_pct, val_pct)
        if len(got[split]) >= need[split]:
            continue
        for p in extract_passages(art["text"], min_chars, max_chars):
            got[split].append({"text": p, "article_id": key})
    return got


def finalize(got: dict, need: dict, seed: int) -> dict:
    rng = random.Random(seed)
    out = {}
    for s, items in got.items():
        items = list(items)
        rng.shuffle(items)
        out[s] = items[: need[s]]
    return out


# ---------- I/O ----------
def write_jsonl(path: Path, rows: list, lang: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({**r, "lang": lang}, ensure_ascii=False) + "\n")


def stream_wikipedia(dump: str, lang: str, seed: int):
    from datasets import load_dataset  # imported lazily so tests run offline
    ds = load_dataset("wikimedia/wikipedia", f"{dump}.{lang}", split="train", streaming=True)
    return ds.shuffle(seed=seed, buffer_size=5000)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/data.yaml")
    ap.add_argument("--force", action="store_true", help="rebuild even if files exist")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    d, root = cfg["dataset"], Path(cfg["paths"]["clients"])
    manifest = []
    for c in cfg["clients"]:
        cdir = root / c["id"]
        if (cdir / "train.jsonl").exists() and not args.force:
            log.info(f"[{c['id']}] exists, skipping (use --force to rebuild)")
        else:
            log.info(f"[{c['id']}] streaming {d['dump']}.{c['lang']} ...")
            need = {"train": c["n_train"], "val": d["n_val"], "test": d["n_test"]}
            got = collect(stream_wikipedia(d["dump"], c["lang"], cfg["seed"]), need, cfg["seed"],
                          d["test_pct"], d["val_pct"], d["min_chars"], d["max_chars"], d["max_articles"])
            data = finalize(got, need, cfg["seed"])
            for s, rows in data.items():
                write_jsonl(cdir / f"{s}.jsonl", rows, c["lang"])
        stats = {**c}
        for s in ("train", "val", "test"):
            rows = [json.loads(l) for l in open(cdir / f"{s}.jsonl", encoding="utf-8")]
            txt = " ".join(r["text"] for r in rows)
            stats[f"n_{s}_actual"] = len(rows)
            stats[f"chars_{s}"] = len(txt)
            if s == "train":
                stats["script_fraction_train"] = round(script_fraction(txt, c["script"]), 4)
        manifest.append(stats)
        log.info(f"[{c['id']}] {stats['n_train_actual']}/{stats['n_val_actual']}/{stats['n_test_actual']} "
                 f"train/val/test passages, script match {stats['script_fraction_train']}")
    save_json({"seed": cfg["seed"], "dataset": d, "clients": manifest}, str(root / "manifest.json"))
    log.info(f"wrote {root / 'manifest.json'}")


if __name__ == "__main__":
    main()
