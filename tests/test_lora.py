"""Single-client LoRA tests. Fully offline: tiny random-init model + local byte tokenizer + tiny synthetic jsonl."""
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from clients.local_train import (batch_loss_sum, build_labels, evaluate, make_collate, perplexity, read_texts,
                                 resolve_client_id, tokenize, train_client)
from data.checks import check_batch, check_texts, check_tokenized
from models.base_model import check_tokenizer_model_compat, load_base_model, load_tokenizer
from models.lora import (attach_lora, count_parameters, load_adapter, lora_summary, save_adapter,
                         verify_lora_setup)
from torch.utils.data import DataLoader
from utils.checkpoints import load_config

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "configs" / "dev.yaml"
DATA = ROOT / "data" / "clients"
TAMIL = ["தமிழ் ஒரு செம்மொழி ஆகும் " * 6, "இது ஒரு சோதனை உரை " * 8, "சென்னை தமிழ்நாட்டின் தலைநகரம் " * 6] * 6


@pytest.fixture(scope="module")
def cfg():
    return load_config(str(CFG))


@pytest.fixture()
def clients_dir(tmp_path):
    for split, n in (("train", 18), ("val", 6)):
        d = tmp_path / "ta"; d.mkdir(exist_ok=True)
        with open(d / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for t in TAMIL[:n]:
                f.write(json.dumps({"text": t, "lang": "ta"}, ensure_ascii=False) + "\n")
    return tmp_path


def _fresh(cfg):
    tok = load_tokenizer(cfg["model"])
    base, dev = load_base_model(cfg["model"], tok, "cpu")
    return tok, base, dev


def test_1_model_loads(cfg):
    tok, base, _ = _fresh(cfg)
    ids = tok("தமிழ்", add_special_tokens=False)["input_ids"]
    assert tok.decode(ids) == "தமிழ்"          # byte tokenizer round-trips non-Latin script
    out = base(input_ids=torch.tensor([ids]))
    assert out.logits.shape[-1] == len(tok)


def test_2_lora_attaches(cfg):
    _, base, _ = _fresh(cfg)
    m = attach_lora(base, cfg["lora"])
    assert any("lora_A" in n for n, _ in m.named_parameters())


def test_3_base_params_frozen(cfg):
    _, base, _ = _fresh(cfg)
    m = attach_lora(base, cfg["lora"])
    assert all(not p.requires_grad for n, p in m.named_parameters() if "lora_" not in n)


def test_4_lora_params_trainable(cfg):
    _, base, _ = _fresh(cfg)
    m = attach_lora(base, cfg["lora"])
    lora = [p for n, p in m.named_parameters() if "lora_" in n]
    assert lora and all(p.requires_grad for p in lora)


def test_5_trainable_smaller_than_total(cfg):
    _, base, _ = _fresh(cfg)
    c = count_parameters(attach_lora(base, cfg["lora"]))
    assert 0 < c["trainable_parameters"] < c["total_parameters"]


def test_6_tiny_training_run(cfg, clients_dir, tmp_path):
    cfg = json.loads(json.dumps(cfg))
    cfg["train"]["max_steps"] = 25
    cfg["output"] = {"results_dir": str(tmp_path / "res"), "checkpoints_dir": str(tmp_path / "ck")}
    m = train_client(cfg, "tamil", clients_dir=str(clients_dir))
    assert m["optimizer_steps"] == 25
    assert m["lora_updated"] is True                            # optimizer really changed the LoRA weights
    assert m["lora_setup_check"]["n_non_lora_trainable"] == 0   # nothing but LoRA was trainable
    assert all(math.isfinite(m[k]) for k in ("first_train_loss", "last_train_loss", "validation_loss", "perplexity"))
    assert (tmp_path / "res" / "ta" / "metrics.json").exists()
    assert (tmp_path / "ck" / "ta" / "adapter" / "adapter_config.json").exists()
    steps = [json.loads(l) for l in open(tmp_path / "res" / "ta" / "log.jsonl", encoding="utf-8")]
    assert sum(1 for r in steps if r["event"] == "step") == 25
    # NOTE: no "loss must go down" assertion here: with a frozen random-init model that is not guaranteed.
    # The real loss curve is recorded by experiments/verify_single_client.py and must be read, not assumed.


def test_7_loss_and_perplexity(cfg, clients_dir):
    tok, base, dev = _fresh(cfg)
    m = attach_lora(base, cfg["lora"])
    data = tokenize(tok, TAMIL[:6], 64)
    r = evaluate(m, DataLoader(data, batch_size=3, collate_fn=make_collate(tok.pad_token_id)), dev)
    assert r["validation_loss"] > 0 and math.isclose(r["perplexity"], math.exp(r["validation_loss"]))
    assert r["validation_tokens"] == sum(len(x) - 1 for x in data)   # padding excluded from the count
    assert perplexity(1e6) == float("inf")                            # no overflow crash


def test_8_9_adapter_save_and_load(cfg, tmp_path):
    import copy
    tok, base, dev = _fresh(cfg)
    pristine = copy.deepcopy(base)                          # identical base weights for the reload
    m = attach_lora(base, cfg["lora"])
    with torch.no_grad():                                   # make LoRA non-trivial (B starts at 0)
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.add_(0.05)
    data = tokenize(tok, TAMIL[:4], 64)
    loader = DataLoader(data, batch_size=4, collate_fn=make_collate(tok.pad_token_id))
    before = evaluate(m, loader, dev)["validation_loss"]
    save_adapter(m, str(tmp_path / "ad"))                   # test 8
    assert (tmp_path / "ad" / "adapter_config.json").exists()
    m2 = load_adapter(pristine, str(tmp_path / "ad"))       # test 9
    after = evaluate(m2, loader, dev)["validation_loss"]
    assert math.isclose(before, after, rel_tol=1e-5)
    a = {n: p for n, p in m.named_parameters() if "lora_" in n}
    b = {n: p for n, p in m2.named_parameters() if "lora_" in n}
    assert a.keys() == b.keys() and all(torch.allclose(a[k], b[k]) for k in a)


def test_client_id_resolution():
    assert resolve_client_id("Tamil", "configs/data.yaml") == "ta"
    assert resolve_client_id("ml") == "ml"


# ---------- additional hardening tests ----------
class _Uniform(nn.Module):
    """Fake LM with a uniform next-token distribution: NLL per predicted token is exactly ln(V)."""

    def __init__(self, vocab):
        super().__init__()
        self.vocab = vocab

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=torch.zeros(*input_ids.shape, self.vocab))


def test_perplexity_exact_for_uniform_model_and_padding_ignored():
    V = 50
    seqs = [[0, 2, 3, 4, 5], [6, 0], [8, 9, 10]]           # pad id 0 also occurs as a REAL token
    r = evaluate(_Uniform(V), DataLoader(seqs, batch_size=3, collate_fn=make_collate(0)), "cpu")
    assert r["validation_tokens"] == 4 + 1 + 2              # padding positions not counted
    assert math.isclose(r["validation_loss"], math.log(V), rel_tol=1e-6)
    assert math.isclose(r["perplexity"], V, rel_tol=1e-5)


def test_labels_shifted_and_padding_masked():
    batch = make_collate(0)([[5, 6, 7], [8, 9]])
    labels = build_labels(batch["input_ids"], batch["attention_mask"])
    assert labels.tolist() == [[6, 7], [9, -100]]


def test_batched_padded_eval_equals_unpadded_eval(cfg):
    tok, base, dev = _fresh(cfg)
    m = attach_lora(base, cfg["lora"])
    with torch.no_grad():
        for n, p in m.named_parameters():
            if "lora_B" in n:
                p.add_(0.05)
    data = tokenize(tok, [TAMIL[0][:k] for k in (3, 10, 25, 60)], 64)
    assert len({len(x) for x in data}) > 1                  # genuinely different lengths => padding happens
    coll = make_collate(tok.pad_token_id)
    a = evaluate(m, DataLoader(data, batch_size=4, collate_fn=coll), dev)
    b = evaluate(m, DataLoader(data, batch_size=1, collate_fn=coll), dev)
    assert a["validation_tokens"] == b["validation_tokens"]
    assert math.isclose(a["validation_loss"], b["validation_loss"], rel_tol=1e-4)


def test_tokenize_eos_only_when_not_truncated_and_drops_empty(cfg):
    tok = load_tokenizer(cfg["model"])
    short = tokenize(tok, ["abc"], 16)[0]
    assert len(short) == 4 and short[-1] == tok.eos_token_id
    long = tokenize(tok, ["a" * 100], 16)[0]
    assert len(long) == 16 and long[-1] != tok.eos_token_id      # cut mid-text: no fake end-of-text
    assert tokenize(tok, [""], 16) == []                          # no next-token target => dropped


def test_read_texts_skips_blank_passages(tmp_path):
    f = tmp_path / "x.jsonl"
    f.write_text('{"text": "hello"}\n{"text": "   "}\n{"text": ""}\n\n{"lang": "x"}\n{"text": "world"}\n', encoding="utf-8")
    assert read_texts(f, None) == ["hello", "world"]


def test_lora_settings_come_from_config_and_freezing_report(cfg):
    _, base, _ = _fresh(cfg)
    m = attach_lora(base, cfg["lora"])
    s = lora_summary(m)
    assert (s["r"], s["alpha"], s["dropout"]) == (cfg["lora"]["r"], cfg["lora"]["alpha"], cfg["lora"]["dropout"])
    assert s["target_modules"] == ["c_attn"]
    v = verify_lora_setup(m)
    assert v["n_lora_tensors"] > 0 and v["n_lora_frozen"] == 0 and v["n_non_lora_trainable"] == 0


def test_lora_forward_equals_base_at_init(cfg):
    tok, base, dev = _fresh(cfg)
    ids = torch.tensor([tokenize(tok, [TAMIL[0]], 32)[0]])
    base.eval()
    with torch.no_grad():
        before = base(input_ids=ids).logits.clone()
        m = attach_lora(base, cfg["lora"])
        m.eval()
        after = m(input_ids=ids).logits
    assert torch.allclose(before, after, atol=1e-6)              # LoRA B starts at zero => same function


def test_backward_reaches_only_lora_and_optimizer_updates_only_lora(cfg):
    tok, base, dev = _fresh(cfg)
    m = attach_lora(base, cfg["lora"])
    frozen_before = {n: p.detach().clone() for n, p in m.named_parameters() if "lora_" not in n}
    lora_before = {n: p.detach().clone() for n, p in m.named_parameters() if "lora_" in n}
    batch = make_collate(tok.pad_token_id)(tokenize(tok, TAMIL[:3], 32))
    assert check_batch(batch, len(tok)) == []
    m.train()
    s, n = batch_loss_sum(m, batch, dev)
    assert torch.isfinite(s)
    (s / n).backward()
    grads_b = [p.grad for nm, p in m.named_parameters() if "lora_B" in nm]
    assert grads_b and all(g is not None for g in grads_b) and any(g.abs().sum() > 0 for g in grads_b)
    assert all(p.grad is None for nm, p in m.named_parameters() if "lora_" not in nm)
    torch.optim.AdamW([p for p in m.parameters() if p.requires_grad], lr=1e-2).step()
    assert any(not torch.equal(lora_before[k], p) for k, p in m.named_parameters() if k in lora_before)
    assert all(torch.equal(frozen_before[k], p) for k, p in m.named_parameters() if k in frozen_before)


def test_adapter_dir_holds_only_the_adapter(cfg, tmp_path):
    _, base, _ = _fresh(cfg)
    base_bytes = sum(p.numel() * p.element_size() for p in base.parameters())
    m = attach_lora(base, cfg["lora"])
    save_adapter(m, str(tmp_path / "ad"))
    files = {f.name: f.stat().st_size for f in (tmp_path / "ad").iterdir()}
    assert "adapter_config.json" in files and "adapter_model.safetensors" in files
    assert not any(n.startswith(("model", "pytorch_model")) for n in files)   # no full base-model weights
    assert files["adapter_model.safetensors"] < 0.25 * base_bytes


def test_tokenizer_model_vocab_compat(cfg):
    tok, base, _ = _fresh(cfg)
    info = check_tokenizer_model_compat(tok, base)
    assert info["tokenizer_size"] <= info["embedding_size"]


@pytest.mark.skipif(not (DATA / "manifest.json").exists(),
                    reason="prepared data not present (run: python -m data.prepare_data)")
def test_prepared_real_data_flows_through_tokenizer_and_model(cfg):
    tok, base, dev = _fresh(cfg)
    coll = make_collate(tok.pad_token_id)
    for c in json.load(open(DATA / "manifest.json", encoding="utf-8"))["clients"]:
        texts = read_texts(DATA / c["id"] / "val.jsonl", 8)
        assert check_texts(texts, c["script"])["n_empty"] == 0
        seqs = tokenize(tok, texts, 64)
        rep = check_tokenized(seqs, 64, len(tok))
        assert rep["n_sequences"] > 0 and rep["n_invalid_token_ids"] == 0 and rep["n_over_max_len"] == 0
        batch = coll(seqs[:4])
        assert check_batch(batch, len(tok)) == []
        s, n = batch_loss_sum(base, batch, dev)
        assert torch.isfinite(s) and n > 0


@pytest.mark.real_model
@pytest.mark.skipif(os.environ.get("RUN_REAL_MODEL") != "1", reason="set RUN_REAL_MODEL=1 (needs the model download)")
def test_real_model_tokenizer_forward_and_lora():
    rcfg = load_config(str(ROOT / "configs" / "dev_real.yaml"))
    tok = load_tokenizer(rcfg["model"])
    base, dev = load_base_model(rcfg["model"], tok, "cpu", "fp32")
    info = check_tokenizer_model_compat(tok, base)
    enc = tok("தமிழ் ஒரு செம்மொழி", return_tensors="pt")
    with torch.no_grad():
        out = base(**enc)
    assert out.logits.shape[-1] >= info["tokenizer_size"] and torch.isfinite(out.logits).all()
    c = count_parameters(attach_lora(base, rcfg["lora"], rcfg["model"].get("lora_target_modules")))
    assert 0 < c["trainable_percent"] < 5
