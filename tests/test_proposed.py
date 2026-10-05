"""Phase C tests: language clustering, tokenizer-alignment projection, cluster aggregation, alpha-mixing, and the
orchestration run. Offline: tiny random-init model, offline hashing encoder / fake semantic encoder, synthetic data."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from clustering.language_cluster import (HashingEmbedder, SentenceTransformerEmbedder, cluster_clients,
                                         compute_fingerprints, language_fingerprint, make_embedder)
from experiments.run_fedavg import GlobalEvaluator
from experiments.run_proposed import fairness_gap, run_proposed
from models.base_model import load_tokenizer
from models.tokenizer_align import (SubwordProjection, alignment_quality, fit_client_projection, fit_projection,
                                    pool_subword_embeddings, project_token_embeddings)
from server.cluster_fedavg import aggregate_within_clusters, alpha_mix, cluster_round, save_cluster_checkpoints
from server.fedavg import aggregate_lora_adapters
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
LANGS = ["en", "hi", "ta", "te", "ml"]


# ---------- helpers ----------
def _fp(*vals):
    v = np.array(vals, dtype=float)
    return v / np.linalg.norm(v)


def synthetic_fingerprints():
    """en and hi each point their own way; the three Dravidian languages are close to each other."""
    return {"en": _fp(1, 0, 0, 0), "hi": _fp(0, 1, 0, 0), "ta": _fp(0, 0, 1, 0.05), "te": _fp(0, 0, 1, -0.05),
            "ml": _fp(0, 0, 1, 0.02)}


def _lang_of(ids):
    return {c: c for c in ids}


def _state(v, dtype=torch.float32):
    return {"x.lora_A": torch.full((2, 3), float(v), dtype=dtype), "x.lora_B": torch.full((3, 2), 2.0 * v, dtype=dtype)}


class FakeSemanticEncoder:
    """Stands in for LaBSE: same-family scripts get nearby vectors (deterministic, no download)."""
    name, is_semantic = "fake-semantic", True

    def encode(self, texts):
        out = []
        for t in texts:
            o = ord(t[0])
            base = np.zeros(8)
            base[0 if o < 0x900 else 1 if o < 0xB00 else 2] = 1.0         # latin | devanagari | dravidian scripts
            base[3 + (o % 4)] += 0.02                                     # tiny per-text variation
            out.append(base / np.linalg.norm(base))
        return np.stack(out)


# ---------- language clustering ----------
def test_dravidian_languages_cluster_together():
    a = cluster_clients(LANGS, _lang_of(LANGS), synthetic_fingerprints())
    assert a.clusters == {0: ["en"], 1: ["hi"], 2: ["ta", "te", "ml"]}
    assert a.cluster_of["te"] == a.cluster_of["ml"] == 2 and a.cluster_of["en"] == 0


def test_same_declared_language_is_always_one_cluster():
    ids = ["ta_a", "ta_b", "en_a"]
    langs = {"ta_a": "ta", "ta_b": "ta", "en_a": "en"}
    fps = {"ta_a": _fp(0, 1), "ta_b": _fp(1, 0), "en_a": _fp(1, 1)}    # ta fingerprints disagree, metadata wins
    a = cluster_clients(ids, langs, fps, similarity_threshold=2.0)      # threshold too high to merge anything else
    assert a.cluster_of["ta_a"] == a.cluster_of["ta_b"] != a.cluster_of["en_a"]


def test_threshold_extremes_and_n_clusters():
    fps = synthetic_fingerprints()
    assert len(cluster_clients(LANGS, _lang_of(LANGS), fps, similarity_threshold=5.0).clusters) == 5
    assert len(cluster_clients(LANGS, _lang_of(LANGS), fps, similarity_threshold=-5.0).clusters) == 1
    for n in (1, 2, 3, 4, 5):
        assert len(cluster_clients(LANGS, _lang_of(LANGS), fps, n_clusters=n).clusters) == n
    two = cluster_clients(LANGS, _lang_of(LANGS), fps, n_clusters=3)
    assert two.clusters[2] == ["ta", "te", "ml"]
    with pytest.raises(ValueError):
        cluster_clients(LANGS, _lang_of(LANGS), fps, n_clusters=6)


def test_embedding_similarity_matters_without_metadata():
    fps = synthetic_fingerprints()
    a = cluster_clients(LANGS, _lang_of(LANGS), fps, family_weight=0.0, similarity_threshold=0.5)
    assert a.clusters[a.cluster_of["ta"]] == ["ta", "te", "ml"]          # grouped by fingerprints alone
    # metadata alone (fingerprints identical): the Dravidian family still groups together
    same = {c: _fp(1, 1, 1, 1) for c in LANGS}
    b = cluster_clients(LANGS, _lang_of(LANGS), same, family_weight=1.0, similarity_threshold=0.5, center=False)
    assert b.clusters[b.cluster_of["ta"]] == ["ta", "te", "ml"] and b.cluster_of["en"] != b.cluster_of["hi"]


def test_cluster_assignment_is_deterministic_complete_and_serialisable():
    fps = synthetic_fingerprints()
    a = cluster_clients(LANGS, _lang_of(LANGS), fps)
    b = cluster_clients(LANGS, _lang_of(LANGS), fps)
    assert a.cluster_of == b.cluster_of
    assert sorted(c for m in a.clusters.values() for c in m) == sorted(LANGS)
    json.dumps(a.to_dict())


def test_cluster_input_validation():
    with pytest.raises(ValueError):
        cluster_clients([], {}, {})
    with pytest.raises(ValueError):
        cluster_clients(["en"], {"en": "en"}, {})
    with pytest.raises(ValueError):
        cluster_clients(["en"], {"en": "en"}, {"en": _fp(1, 0)}, family_weight=2.0)


def test_fingerprints_and_encoders():
    e = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert np.allclose(language_fingerprint(e), _fp(1, 1))
    with pytest.raises(ValueError):
        language_fingerprint(np.zeros((0, 2)))
    h = HashingEmbedder(64)
    x1, x2 = h.encode([LANG_TEXT["ta"], LANG_TEXT["en"]]), h.encode([LANG_TEXT["ta"], LANG_TEXT["en"]])
    assert np.array_equal(x1, x2) and np.allclose(np.linalg.norm(x1, axis=1), 1.0)
    assert float(x1[0] @ x1[1]) < 0.5                                     # different scripts are not near-identical
    fps = compute_fingerprints(h, {"en": [LANG_TEXT["en"]] * 3, "ta": [LANG_TEXT["ta"]] * 3})
    assert set(fps) == {"en", "ta"} and fps["en"].shape == (64,)
    assert make_embedder("offline-hash", dim=32).dim == 32


def test_sentence_transformer_wrapper_with_injected_model():
    class M:
        def encode(self, texts, **kw):
            return np.array([[len(t), 1.0] for t in texts])
    emb = SentenceTransformerEmbedder("fake", model=M())
    out = emb.encode(["a", "bbb"])
    assert out.shape == (2, 2) and np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_sentence_transformer_wrapper_with_real_local_model(tmp_path):
    """Exercises the real sentence-transformers code path with a tiny locally built (random) model."""
    st = pytest.importorskip("sentence_transformers")
    from transformers import GPT2Config, GPT2LMHeadModel
    tok = load_tokenizer({"name": "tiny-offline"})
    d = tmp_path / "tiny"
    GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=64, n_embd=32, n_layer=1, n_head=2,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)).save_pretrained(d)
    tok.save_pretrained(d)
    try:
        mods = getattr(st, "models", None)
        if mods is None:                                  # sentence-transformers >= 6 moved the modules
            from sentence_transformers.sentence_transformer import modules as mods
        model = st.SentenceTransformer(modules=[mods.Transformer(str(d), max_seq_length=32), mods.Pooling(32, "mean")])
    except Exception as e:  # noqa: BLE001 - library-version differences in building modules offline
        pytest.skip(f"cannot build a local sentence-transformers model here: {type(e).__name__}: {e}")
    emb = SentenceTransformerEmbedder("tiny-local", model=model)
    v = emb.encode([LANG_TEXT["ta"], LANG_TEXT["en"]])
    assert v.shape == (2, 32) and np.allclose(np.linalg.norm(v, axis=1), 1.0, atol=1e-5)


# ---------- projection layer ----------
def test_projection_maps_dims_and_has_no_bias():
    p = SubwordProjection(10, 4)
    assert p(torch.randn(7, 10)).shape == (7, 4) and p(torch.randn(2, 3, 10)).shape == (2, 3, 4)
    assert p.proj.bias is None and p.num_parameters == 40


@pytest.mark.parametrize("n,d", [(40, 6), (6, 40)])             # primal and dual (n < d) solves
def test_fit_projection_recovers_linear_map(n, d):
    g = torch.Generator().manual_seed(0)
    x, w = torch.randn(n, d, generator=g), torch.randn(d, 3, generator=g)
    y = x @ w
    p = fit_projection(x, y, ridge=1e-8)
    assert torch.allclose(p(x), y, atol=1e-3)
    if n > d:
        assert torch.allclose(p.proj.weight.T, w, atol=1e-3)


def test_fit_projection_ridge_shrinks_and_validates():
    g = torch.Generator().manual_seed(1)
    x, y = torch.randn(30, 5, generator=g), torch.randn(30, 3, generator=g)
    assert fit_projection(x, y, 100.0).proj.weight.norm() < fit_projection(x, y, 1e-3).proj.weight.norm()
    with pytest.raises(ValueError):
        fit_projection(x, y[:10])
    with pytest.raises(ValueError):
        fit_projection(x, y, ridge=0.0)


def test_project_token_embeddings_matches_matmul_and_chunks():
    g = torch.Generator().manual_seed(2)
    table, p = torch.randn(25, 6, generator=g), SubwordProjection(6, 3)
    out = project_token_embeddings(p, table, chunk=7)
    assert out.shape == (25, 3) and torch.allclose(out, table @ p.proj.weight.T, atol=1e-6)


def test_pool_subword_embeddings():
    table = torch.arange(12.0).reshape(4, 3)
    assert torch.allclose(pool_subword_embeddings(table, [[0, 2]]), table[[0, 2]].mean(0, keepdim=True))
    with pytest.raises(ValueError):
        pool_subword_embeddings(table, [[]])


def test_client_projection_aligns_heldout_and_leaves_tokenizer_untouched():
    tok = load_tokenizer({"name": "tiny-offline"})
    vocab_before, size_before = dict(tok.get_vocab()), len(tok)
    enc = FakeSemanticEncoder()
    g = torch.Generator().manual_seed(3)
    table = torch.randn(len(tok), 64, generator=g)
    texts = {l: [LANG_TEXT[l] * (1 + i % 3) for i in range(12)] for l in LANGS}
    tr = [t for l in LANGS for t in texts[l][:9]]
    va = [t for l in LANGS for t in texts[l][9:]]
    r = fit_client_projection(table, tok, tr, enc.encode(tr), va, enc.encode(va), max_len=64, ridge=1e-1)
    assert r["projection"].d_in == 64 and r["projection"].d_common == 8
    assert r["heldout"]["mean_cosine"] > 0.95
    assert r["heldout"]["mean_cosine"] > r["heldout"]["shuffled_baseline_cosine"] + 0.1
    assert r["n_train"] == len(tr) and r["parameters"] == 64 * 8
    assert dict(tok.get_vocab()) == vocab_before and len(tok) == size_before     # tokenizer unmodified
    q = alignment_quality(r["projection"], pool_subword_embeddings(table, [[1, 2]]), torch.ones(1, 8))
    assert "shuffled_baseline_cosine" not in q


# ---------- cluster aggregation ----------
def test_aggregate_within_clusters_matches_per_cluster_fedavg():
    cl = [_state(1), _state(3), _state(10), _state(20)]
    w, ks = [1, 3, 2, 2], [0, 0, 1, 1]
    out = aggregate_within_clusters(cl, w, ks)
    assert set(out) == {0, 1}
    assert torch.allclose(out[0]["x.lora_A"], torch.full((2, 3), (1 * 1 + 3 * 3) / 4))
    assert torch.allclose(out[1]["x.lora_A"], torch.full((2, 3), 15.0))
    with pytest.raises(ValueError):
        aggregate_within_clusters(cl, w[:3], ks)


def test_cluster_aggregation_isolates_clusters():
    out = aggregate_within_clusters([_state(1), _state(1000)], [1, 1], [0, 1])
    assert torch.equal(out[0]["x.lora_A"], _state(1)["x.lora_A"]) and torch.equal(out[1]["x.lora_A"], _state(1000)["x.lora_A"])


# ---------- alpha mixing ----------
def test_alpha_mix_matches_formula():
    clusters = {0: _state(2), 1: _state(10)}
    sizes = {0: 30.0, 1: 10.0}                      # N_0/N = .75, N_1/N = .25
    prev = _state(100)
    avg = 0.75 * 2 + 0.25 * 10
    for alpha in (0.0, 0.3, 0.5, 1.0):
        out = alpha_mix(clusters, sizes, prev, alpha)
        expect = alpha * avg + (1 - alpha) * 100
        assert torch.allclose(out["x.lora_A"], torch.full((2, 3), expect), atol=1e-5)
        assert torch.allclose(out["x.lora_B"], torch.full((3, 2), 2 * expect), atol=1e-5)


def test_alpha_one_is_size_weighted_fedavg_and_zero_keeps_previous():
    clusters = {0: _state(1), 1: _state(5), 2: _state(9)}
    sizes = {0: 1.0, 1: 2.0, 2: 3.0}
    prev = _state(-7)
    ref = aggregate_lora_adapters([clusters[k] for k in (0, 1, 2)], [1, 2, 3])
    one = alpha_mix(clusters, sizes, prev, 1.0)
    zero = alpha_mix(clusters, sizes, prev, 0.0)
    assert all(torch.allclose(one[k], ref[k], atol=1e-6) for k in ref)
    assert all(torch.equal(zero[k], prev[k]) for k in prev)


def test_alpha_mix_dtype_inputs_untouched_and_validation():
    clusters = {0: _state(1, torch.bfloat16)}
    prev = _state(3, torch.bfloat16)
    before = {k: v.clone() for k, v in prev.items()}
    out = alpha_mix(clusters, {0: 5.0}, prev, 0.5)
    assert out["x.lora_A"].dtype == torch.bfloat16 and torch.allclose(out["x.lora_A"].float(), torch.full((2, 3), 2.0))
    assert all(torch.equal(prev[k], before[k]) for k in prev)
    with pytest.raises(ValueError):
        alpha_mix(clusters, {0: 5.0}, prev, 1.5)
    with pytest.raises(ValueError):
        alpha_mix(clusters, {1: 5.0}, prev, 0.5)
    with pytest.raises(ValueError):
        alpha_mix(clusters, {0: 5.0}, {"other": torch.zeros(1)}, 0.5)


def test_cluster_round_keeps_stale_cluster_and_mixes_with_static_sizes():
    prev_clusters = {0: _state(0), 1: _state(8)}
    sizes = {0: 1.0, 1: 3.0}
    new_c, new_g = cluster_round([_state(4)], [10], [0], prev_clusters, sizes, _state(0), alpha=1.0)
    assert torch.equal(new_c[1]["x.lora_A"], prev_clusters[1]["x.lora_A"])        # cluster 1 had no clients
    assert torch.allclose(new_g["x.lora_A"], torch.full((2, 3), 0.25 * 4 + 0.75 * 8))


def test_save_cluster_checkpoints(tmp_path):
    save_cluster_checkpoints(str(tmp_path), {0: _state(1), 1: _state(2)}, _state(3), {0: ["en"], 1: ["ta", "te"]},
                             {0: 24.0, 1: 20.0}, 0.5)
    assert torch.equal(torch.load(tmp_path / "cluster_1" / "adapter.pt")["x.lora_A"], _state(2)["x.lora_A"])
    assert torch.equal(torch.load(tmp_path / "global_adapter" / "adapter.pt")["x.lora_A"], _state(3)["x.lora_A"])
    meta = json.loads((tmp_path / "clusters.json").read_text())
    assert meta["clusters"]["1"] == ["ta", "te"] and meta["alpha"] == 0.5


def test_fairness_gap():
    g = fairness_gap({"en": {"perplexity": 10.0}, "ta": {"perplexity": 50.0}, "hi": {"perplexity": 20.0}})
    assert g["gap"] == 40.0 and g["max_language"] == "ta" and g["min_language"] == "en" and g["ratio"] == 5.0


# ---------- orchestration (tiny offline model) ----------
@pytest.fixture()
def env(tmp_path):
    clients = tmp_path / "clients"
    for lang, text in LANG_TEXT.items():
        d = clients / lang
        d.mkdir(parents=True)
        for split, n in (("train", N_TRAIN[lang]), ("val", 4)):
            with open(d / f"{split}.jsonl", "w", encoding="utf-8") as f:
                for i in range(n):
                    f.write(json.dumps({"text": text * (2 + i % 3), "lang": lang}, ensure_ascii=False) + "\n")
    c = load_config(str(ROOT / "configs" / "proposed_dev.yaml"))
    c["output"]["proposed_results_dir"] = str(tmp_path / "results" / "proposed")
    c["output"]["proposed_checkpoints_dir"] = str(tmp_path / "checkpoints" / "proposed")
    c["data"]["max_train_samples"] = 24
    c["data"]["data_config"] = None
    c["alignment"]["ridge"] = 1e-1
    return c, str(clients), tmp_path


def test_run_proposed_end_to_end_and_alpha_mixing(env):
    c, clients, tmp = env
    alpha = 0.4
    m = run_proposed(c, LANGS, clients, rounds=1, local_steps=3, alpha=alpha, embedder=FakeSemanticEncoder())
    assert m["finished"] and m["rounds_completed"] == 1 and m["fingerprint_encoder"] == "fake-semantic"
    cl = m["clustering"]["clusters"]
    assert cl == {"0": ["en"], "1": ["hi"], "2": ["ta", "te", "ml"]}
    assert m["clustering"]["cluster_sizes"] == {"0": 24.0, "1": 12.0, "2": 28.0}
    r = m["rounds"][0]
    assert all(i["lora_updated"] and i["optimizer_steps"] == 3 for i in r["clients"])
    for key in ("cluster_adapter", "global_adapter"):
        ev = r["eval"][key]
        assert set(ev["per_language"]) == set(LANGS)
        ppl = [v["perplexity"] for v in ev["per_language"].values()]
        assert ev["fairness_gap"]["gap"] == pytest.approx(max(ppl) - min(ppl))
    com = r["communication"]
    assert com["upload_bytes"] == sum(i["upload_bytes"] for i in r["clients"]) and com["download_bytes"] > 0
    assert m["total_communication_bytes"] == m["setup_communication_bytes"] + com["total_bytes"]
    assert set(m["tokenizer_alignment"]) == set(LANGS) and "heldout" in m["tokenizer_alignment"]["ta"]
    # checkpoints exist and the saved global equals alpha*sum_c N_c/N cluster_c + (1-alpha)*initial
    ck = tmp / "checkpoints" / "proposed"
    for k in ("0", "1", "2"):
        assert (ck / f"cluster_{k}" / "adapter.pt").exists() and (ck / f"cluster_{k}" / "adapter_config.json").exists()
    assert (ck / "global_adapter" / "adapter_config.json").exists() and (ck / "clusters.json").exists()
    assert not (ck / "_client_tmp").exists()
    states = {k: torch.load(ck / f"cluster_{k}" / "adapter.pt") for k in (0, 1, 2)}
    sizes = {0: 24.0, 1: 12.0, 2: 28.0}
    init = GlobalEvaluator(c, LANGS, clients).initial_state()
    avg = aggregate_lora_adapters([states[k] for k in (0, 1, 2)], [sizes[k] for k in (0, 1, 2)])
    g = torch.load(ck / "global_adapter" / "adapter.pt")
    for name in g:
        assert torch.allclose(g[name], alpha * avg[name] + (1 - alpha) * init[name], atol=1e-5)
    assert json.loads((tmp / "results" / "proposed" / "metrics.json").read_text())["method"] == "proposed_language_aware"


def test_clusters_actually_aggregate_their_members(env):
    """Cluster 2 (ta, te, ml) adapter = sample-weighted mean of those three clients' trained adapters."""
    import clients.local_train as lt
    from experiments.run_fedavg import injected_global_adapter
    from models.lora import adapter_state
    c, clients, tmp = env
    run_proposed(c, LANGS, clients, rounds=1, local_steps=2, alpha=1.0, embedder=FakeSemanticEncoder())
    saved = torch.load(tmp / "checkpoints" / "proposed" / "cluster_2" / "adapter.pt")
    start = GlobalEvaluator(c, LANGS, clients).initial_state()
    trained = []
    for ci, lang in enumerate(LANGS):
        if lang not in ("ta", "te", "ml"):
            continue
        cc = json.loads(json.dumps(c)); cc["seed"] = c["seed"] + 1000 + ci; cc["train"]["max_steps"] = 2
        cc["output"] = {"results_dir": str(tmp / "h"), "checkpoints_dir": str(tmp / "hc")}
        holder = {}
        with injected_global_adapter(start, holder):
            lt.train_client(cc, lang, clients, output_tag=lang)
        trained.append(adapter_state(holder["model"]))
    expect = aggregate_lora_adapters(trained, [12, 8, 8])
    assert all(torch.allclose(saved[k], expect[k], atol=1e-5) for k in saved)


def test_cluster_init_global_and_client_sampling(env):
    c, clients, _ = env
    m = run_proposed(c, LANGS, clients, rounds=2, local_steps=1, clients_per_round=2, cluster_init="global",
                     embedder=FakeSemanticEncoder())
    assert m["cluster_init"] == "global"
    for r in m["rounds"]:
        assert len(r["active_clients"]) == 2
        assert set(r["eval"]["cluster_adapter"]["per_language"]) == set(LANGS)   # all cohorts still evaluated


def test_offline_hash_dev_config_and_no_alignment(env):
    c, clients, _ = env
    c["alignment"]["enabled"] = False
    m = run_proposed(c, LANGS, clients, rounds=1, local_steps=1)               # default encoder = offline-hash
    assert m["fingerprint_encoder"] == "offline-hash" and m["encoder_is_semantic"] is False
    assert m["tokenizer_alignment"] is None


def test_invalid_arguments(env):
    c, clients, _ = env
    for kw in ({"alpha": 1.5}, {"alpha": -0.1}, {"cluster_init": "bogus"}, {"rounds": 0}, {"clients_per_round": 9}):
        with pytest.raises(ValueError):
            run_proposed(c, ["en"], clients, **{"rounds": 1, **kw})
