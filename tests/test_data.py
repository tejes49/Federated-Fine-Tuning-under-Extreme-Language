"""Phase 2 tests (offline; synthetic articles, no network)."""
from pathlib import Path

from data.prepare_data import assign_split, collect, extract_passages, finalize, script_fraction
from utils.checkpoints import load_config

CFG = Path(__file__).resolve().parent.parent / "configs" / "data.yaml"


def test_split_is_deterministic_and_covers_all():
    a = [assign_split(str(i), 42, 10, 10) for i in range(1000)]
    assert a == [assign_split(str(i), 42, 10, 10) for i in range(1000)]
    assert {"train", "val", "test"} == set(a)
    assert 50 < a.count("test") < 150  # ~10%


def test_extract_passages_length_rules():
    text = "short\n" + ("word " * 100) + "\n" + ("w " * 1500)
    ps = extract_passages(text, 200, 1000)
    assert ps and all(200 <= len(p) <= 1000 for p in ps)


def test_script_fraction():
    assert script_fraction("இது தமிழ் உரை", "TAMIL") > 0.95
    assert script_fraction("hello world", "LATIN") == 1.0
    assert script_fraction("hello world", "TAMIL") == 0.0


def test_collect_finalize_no_leakage_and_counts():
    arts = [{"id": i, "text": ("sentence number %d " % i) * 30} for i in range(5000)]
    need = {"train": 50, "val": 10, "test": 20}
    got = collect(arts, need, 42, 10, 10, 200, 1000, 5000)
    out = finalize(got, need, 42)
    assert {s: len(v) for s, v in out.items()} == need
    ids = {s: {r["article_id"] for r in v} for s, v in out.items()}
    assert not (ids["train"] & ids["val"] or ids["train"] & ids["test"] or ids["val"] & ids["test"])


def test_finalize_reproducible():
    arts = [{"id": i, "text": ("abc %d " % i) * 60} for i in range(2000)]
    need = {"train": 30, "val": 5, "test": 5}
    g = collect(arts, need, 42, 10, 10, 200, 1000, 2000)
    assert finalize(g, need, 42) == finalize(g, need, 42)


def test_data_config_valid():
    cfg = load_config(str(CFG))
    ids = [c["id"] for c in cfg["clients"]]
    assert len(ids) == len(set(ids)) == 5
    n = {c["resource"]: c["n_train"] for c in cfg["clients"]}
    assert n["high"] > n["medium"] > n["low"]
    assert {c["lang"] for c in cfg["clients"]} == {"en", "hi", "ta", "te", "ml"}
