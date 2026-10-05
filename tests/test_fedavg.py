"""FedAvg tests. Aggregation tests are pure torch; round tests use the tiny random-init model (fully offline)."""
import json
from pathlib import Path

import pytest
import torch

from experiments.run_fedavg import (GlobalEvaluator, injected_global_adapter, load_adapter_state, run_fedavg)
from models.lora import adapter_state
from server.fedavg import adapter_num_bytes, aggregate_lora_adapters, normalize_weights
from utils.checkpoints import load_config

ROOT = Path(__file__).resolve().parent.parent
LANG_TEXT = {
    "en": "The quick brown fox jumps over the lazy dog near the river bank. ",
    "hi": "हिन्दी भारत की एक प्रमुख भाषा है और बहुत लोग इसे बोलते हैं। ",
    "ta": "தமிழ் ஒரு செம்மொழி ஆகும் இது மிகவும் பழமையானது. ",
    "te": "తెలుగు ఒక ద్రావిడ భాష ఇది ఆంధ్ర ప్రదేశ్ లో మాట్లాడతారు. ",
    "ml": "മലയാളം കേരളത്തിലെ ഔദ്യോഗിക ഭാഷയാണ് ഇത് ദ്രാവിഡ ഭാഷയാണ്. ",
}
N_TRAIN = {"en": 24, "hi": 12, "ta": 12, "te": 8, "ml": 8}


def _state(v, dtype=torch.float32):
    return {"a.lora_A.w": torch.full((2, 3), float(v), dtype=dtype), "a.lora_B.w": torch.full((3, 2), 2.0 * v, dtype=dtype)}


# ---------- aggregation ----------
def test_uniform_average():
    out = aggregate_lora_adapters([_state(1), _state(3)])
    assert torch.allclose(out["a.lora_A.w"], torch.full((2, 3), 2.0))
    assert torch.allclose(out["a.lora_B.w"], torch.full((3, 2), 4.0))


def test_weighted_average_normalises_weights():
    a = aggregate_lora_adapters([_state(0), _state(10)], weights=[1, 3])
    b = aggregate_lora_adapters([_state(0), _state(10)], weights=[0.25, 0.75])
    assert torch.allclose(a["a.lora_A.w"], torch.full((2, 3), 7.5))
    assert torch.equal(a["a.lora_A.w"], b["a.lora_A.w"])


def test_single_client_is_identity_and_inputs_untouched():
    s = _state(5)
    before = {k: v.clone() for k, v in s.items()}
    out = aggregate_lora_adapters([s])
    assert all(torch.equal(out[k], before[k]) for k in s)
    assert all(torch.equal(s[k], before[k]) for k in s)
    assert out["a.lora_A.w"] is not s["a.lora_A.w"]


def test_zero_weight_client_is_ignored():
    out = aggregate_lora_adapters([_state(1), _state(100)], weights=[1, 0])
    assert torch.allclose(out["a.lora_A.w"], torch.ones(2, 3))


def test_dtype_preserved_and_order_invariant():
    cl = [_state(1, torch.bfloat16), _state(2, torch.bfloat16), _state(4, torch.bfloat16)]
    w = [1, 2, 3]
    out = aggregate_lora_adapters(cl, w)
    assert out["a.lora_A.w"].dtype == torch.bfloat16
    rev = aggregate_lora_adapters(cl[::-1], w[::-1])
    assert torch.equal(out["a.lora_A.w"], rev["a.lora_A.w"])


def test_random_tensors_match_manual_formula():
    g = torch.Generator().manual_seed(0)
    cl = [{"p": torch.randn(4, 5, generator=g)} for _ in range(4)]
    w = [3.0, 1.0, 5.0, 2.0]
    expect = sum(wi / sum(w) * c["p"] for wi, c in zip(w, cl))
    assert torch.allclose(aggregate_lora_adapters(cl, w)["p"], expect, atol=1e-6)


@pytest.mark.parametrize("weights", [[1], [1, 2, 3], [-1, 2], [0, 0], [float("nan"), 1], [float("inf"), 1]])
def test_bad_weights_rejected(weights):
    with pytest.raises(ValueError):
        aggregate_lora_adapters([_state(1), _state(2)], weights)


def test_incompatible_adapters_rejected():
    with pytest.raises(ValueError):
        aggregate_lora_adapters([])
    with pytest.raises(ValueError):
        aggregate_lora_adapters([_state(1), {"other": torch.zeros(1)}])
    bad = _state(1); bad["a.lora_A.w"] = torch.zeros(5, 5)
    with pytest.raises(ValueError, match="shape mismatch"):
        aggregate_lora_adapters([_state(1), bad])
    nan = _state(1); nan["a.lora_A.w"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        aggregate_lora_adapters([_state(1), nan])


def test_helpers():
    assert normalize_weights(None, 4) == [0.25] * 4
    assert adapter_num_bytes(_state(1)) == (6 + 6) * 4


# ---------- federated rounds (tiny offline model) ----------
@pytest.fixture(scope="module")
def cfg():
    return load_config(str(ROOT / "configs" / "fedavg_dev.yaml"))


@pytest.fixture()
def env(tmp_path, cfg):
    clients = tmp_path / "clients"
    for lang, text in LANG_TEXT.items():
        d = clients / lang; d.mkdir(parents=True)
        for split, n in (("train", N_TRAIN[lang]), ("val", 4)):
            with open(d / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for i in range(n):
                    f.write(json.dumps({"text": text * (2 + i % 3), "lang": lang}, ensure_ascii=False) + "\n")
    c = json.loads(json.dumps(cfg))
    c["output"]["fedavg_results_dir"] = str(tmp_path / "results" / "fedavg")
    c["output"]["fedavg_checkpoints_dir"] = str(tmp_path / "checkpoints" / "fedavg")
    c["data"]["max_train_samples"] = 24
    c["data"]["data_config"] = None
    return c, str(clients), tmp_path


def test_injection_loads_global_adapter_and_restores_attach(env):
    import clients.local_train as lt
    c, clients, _ = env
    ev = GlobalEvaluator(c, ["ta"], clients)
    state = ev.initial_state()
    marked = {k: torch.full_like(v, 0.125) for k, v in state.items()}
    original = lt.attach_lora
    holder = {}
    with injected_global_adapter(marked, holder):
        m = lt.train_client({**c, "train": {**c["train"], "max_steps": 1}, "output": {
            "results_dir": str(env[2] / "r"), "checkpoints_dir": str(env[2] / "c")}}, "ta", clients)
    assert lt.attach_lora is original
    # one tiny step from the marked start -> the result stays near it (would be ~0/random if not injected)
    after = adapter_state(holder["model"])
    assert all((after[k] - 0.125).abs().max() < 0.05 for k in after)
    assert m["lora_updated"]


def test_run_fedavg_end_to_end(env):
    c, clients, tmp = env
    m = run_fedavg(c, ["en", "hi", "ta", "te", "ml"], clients, rounds=2, local_steps=3)
    assert m["finished"] and m["rounds_completed"] == 2
    for r in m["rounds"]:
        assert r["active_clients"] == ["en", "hi", "ta", "te", "ml"]
        assert abs(sum(cl["aggregation_weight"] for cl in r["clients"]) - 1) < 1e-9
        assert all(cl["lora_updated"] and cl["optimizer_steps"] == 3 for cl in r["clients"])
        assert set(r["global_eval"]["per_language"]) == {"en", "hi", "ta", "te", "ml"}
        for v in r["global_eval"]["per_language"].values():
            assert v["validation_loss"] > 0 and v["perplexity"] > 1
        assert r["round_seconds"] > 0
        com = r["communication"]
        assert com["upload_bytes"] == sum(cl["upload_bytes"] for cl in r["clients"])
        assert com["download_bytes"] > 0 and com["total_bytes"] == com["upload_bytes"] + com["download_bytes"]
    assert m["total_communication_bytes"] == m["rounds"][-1]["communication"]["cumulative_bytes"]
    assert m["rounds"][0]["clients"][0]["aggregation_weight"] == pytest.approx(24 / 64)  # en: 24 of 64 samples
    out = tmp / "results" / "fedavg" / "metrics.json"
    assert json.loads(out.read_text(encoding="utf-8"))["rounds_completed"] == 2
    ck = tmp / "checkpoints" / "fedavg"
    assert (ck / "global_adapter" / "adapter_config.json").exists() and (ck / "global_adapter.pt").exists()
    assert not (ck / "_client_tmp").exists()
    # saved global == what the last round evaluated
    saved = torch.load(ck / "global_adapter.pt")
    ev = GlobalEvaluator(c, list(LANG_TEXT), clients).evaluate(saved)
    final = m["final_global_eval"]["per_language"]
    for k in final:
        assert ev[k]["validation_loss"] == pytest.approx(final[k]["validation_loss"], rel=1e-5)


def test_global_is_weighted_mean_of_clients_one_round(env):
    """Reproduce round 1 by hand with train_client + aggregate and compare with run_fedavg's global adapter."""
    import clients.local_train as lt
    c, clients, tmp = env
    langs = ["en", "ta"]
    run_fedavg(c, langs, clients, rounds=1, local_steps=2)
    g = torch.load(tmp / "checkpoints" / "fedavg" / "global_adapter.pt")
    ev = GlobalEvaluator(c, langs, clients)
    start, trained = ev.initial_state(), []
    for i, lang in enumerate(langs):
        cc = json.loads(json.dumps(c)); cc["seed"] = c["seed"] + 1000 + i; cc["train"]["max_steps"] = 2
        cc["output"] = {"results_dir": str(tmp / "h"), "checkpoints_dir": str(tmp / "hc")}
        holder = {}
        with injected_global_adapter(start, holder):
            lt.train_client(cc, lang, clients, output_tag=lang)
        trained.append(adapter_state(holder["model"]))
    expect = aggregate_lora_adapters(trained, [24, 12])
    assert all(torch.allclose(g[k], expect[k], atol=1e-5) for k in g)


def test_client_sampling_and_uniform_weighting(env):
    c, clients, _ = env
    m = run_fedavg(c, ["en", "hi", "ta", "te", "ml"], clients, rounds=3, local_steps=1,
                   clients_per_round=2, weighting="uniform")
    for r in m["rounds"]:
        assert len(r["active_clients"]) == 2
        assert all(cl["aggregation_weight"] == pytest.approx(0.5) for cl in r["clients"])
        assert len(r["global_eval"]["per_language"]) == 5   # all cohorts evaluated even if not sampled


def test_invalid_run_arguments(env):
    c, clients, _ = env
    with pytest.raises(ValueError):
        run_fedavg(c, ["en"], clients, rounds=0)
    with pytest.raises(ValueError):
        run_fedavg(c, ["en"], clients, rounds=1, clients_per_round=3)
    with pytest.raises(ValueError):
        run_fedavg(c, ["en"], clients, rounds=1, weighting="bogus")


def test_load_adapter_state_rejects_mismatch(env):
    c, clients, _ = env
    ev = GlobalEvaluator(c, ["en"], clients)
    s = ev.initial_state(); s.pop(next(iter(s)))
    with pytest.raises(ValueError, match="mismatch"):
        load_adapter_state(ev.model, s)
