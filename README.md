# Federated Fine-Tuning under Extreme Language Heterogeneity

Implementation of the paper by Dhayanidhimaran R, Dhivakar G, Jetlin C P (St. Joseph's Institute of Technology).

**Research integrity:** all numbers in the paper are placeholders. Nothing in this repo reports a result
unless it was produced by an executed experiment saved under `results/`. Anything not run is **NOT YET EVALUATED**.

## Current milestone: single-client multilingual LoRA (no federation yet)

Three status words are used below and they are not interchangeable:

* **IMPLEMENTED** - the code exists.
* **EXECUTED AND VERIFIED** - it was actually run and the outcome observed (where, and how, is stated).
* **NOT YET IMPLEMENTED** - no code.

### IMPLEMENTED
Config system with `extends`, seeding, device/precision selection, logging; Wikipedia client-data preparation;
base-model + tokenizer loading (`models/base_model.py`); LoRA attach / freeze / count / save / load
(`models/lora.py`); single-client training with token-weighted validation loss and perplexity
(`clients/local_train.py`); real-data sanity checks (`data/checks.py`); an end-to-end verification script
(`experiments/verify_single_client.py`); resource reporting (`utils/resources.py`); CLI overrides for every
tunable (`utils/config_overrides.py`); tests for all of the above.

### EXECUTED AND VERIFIED
Only the following has been run, in the last hardening session, in a sandbox with **1 CPU core, ~3.9 GB RAM, no GPU,
no PyPI / Hugging Face access, and no torch / transformers / peft / pytest installed**:

* The torch-free tests: `tests/test_setup.py`, `tests/test_data.py`, `tests/test_data_checks.py`,
  `tests/test_configs.py` - **21 test cases, 21 passed, 0 failed**. pytest was not installable, so they were run with a
  small hand-written runner that emulates `pytest.mark.parametrize` / `pytest.raises` (the tests themselves are plain
  pytest and unchanged by it).
* `python -m py_compile` on every modified Python file.
* Every config used by those tests loads and `extends` resolves; `configs/dev_real.yaml` resolves to `bigscience/bloom-560m`.

### NOT YET EXECUTED (implemented, but never run by the person or tool that wrote it)
Everything that imports torch / transformers / peft. In particular: `tests/test_lora.py` (21 tests, including the new
ones), `python -m clients.local_train`, `python -m data.checks`, `python -m experiments.verify_single_client`. Earlier
notes in this repo said the pipeline had been verified on a tiny random-init model; that was **not reproduced** in the
hardening session (no torch available), so treat it as unconfirmed until `python -m pytest -q` passes on your machine.

Also NOT done: Wikipedia download (`data/clients/` is empty), download or forward pass of any real multilingual model,
any real loss / perplexity value, any check of Hindi / Tamil / Telugu / Malayalam tokenizer coverage.
**Real multilingual model execution was not completed in this environment.**

### NOT YET IMPLEMENTED
FedAvg, federated client orchestration, client sampling, language clustering, sentence embeddings, cluster
aggregation, global adapter, tokenizer alignment, communication experiments, final research evaluation.

## Phase B: FedAvg + LoRA baseline (IMPLEMENTED; tested only on the tiny offline model with synthetic data)

* `server/fedavg.py` - `aggregate_lora_adapters(client_adapters, weights=None)`: weighted FedAvg of LoRA state_dicts.
* `experiments/run_fedavg.py` - multi-round orchestration over en/hi/ta/te/ml using the unchanged `train_client`;
  writes `results/fedavg/metrics.json` and `checkpoints/fedavg/`.
* Run: `python -m experiments.run_fedavg --config configs/fedavg.yaml` (needs `data/clients/<id>/`; use
  `configs/fedavg_dev.yaml` for a no-download pipeline check). Any number in `metrics.json` is real only when produced by such a run.
* NOT YET EXECUTED: any FedAvg run with the real multilingual model on real Wikipedia client data.

## Phase C: proposed method (IMPLEMENTED; tested only on the tiny offline model with synthetic data)

* `clustering/language_cluster.py` - language fingerprints (mean sentence embedding; LaBSE intended, offline hashing encoder for dev)
  and clustering from declared language + family metadata + fingerprint cosine.
* `models/tokenizer_align.py` - linear subword-embedding -> common-space projection (closed-form ridge); tokenizers untouched.
  Diagnostic only: the shared-tokenizer LoRA training does not consume it.
* `server/cluster_fedavg.py` - per-cluster FedAvg and `Delta_global = alpha*sum_c(N_c/N)*Delta_c + (1-alpha)*Delta_global_prev`
  (applied to the LoRA tensors, which are the delta on the frozen base); cluster + global checkpoints.
* `experiments/run_proposed.py` - orchestration; per-language PPL (cluster adapter and global adapter), fairness gap
  (max PPL - min PPL), communication bytes -> `results/proposed/metrics.json`, checkpoints in `checkpoints/proposed/`.
* Run: `python -m experiments.run_proposed --config configs/proposed.yaml` (dev: `configs/proposed_dev.yaml`).
* NOT YET EXECUTED: LaBSE fingerprints, any run with the real multilingual model / real data. Clustering threshold,
  family weight and alpha are uncalibrated defaults.

## Setup

    python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
    pip install -r requirements.txt
    python check_env.py

## Prepare data (needs internet; streams Wikipedia)

    python -m data.prepare_data --config configs/data.yaml

Writes `data/clients/<id>/{train,val,test}.jsonl` and `data/clients/manifest.json` (ids: en, hi, ta, te, ml).
Sanity-check one language (empty passages, script/language mixing, tokens per word, sequence lengths, id range):

    python -m data.checks --language tamil

## Run the tests

    python -m pytest -q                       # lightweight: offline, tiny random-init model + byte tokenizer
    RUN_REAL_MODEL=1 python -m pytest -q -m real_model     # optional: also loads the real model (download)

Tests that need prepared data skip themselves when `data/clients/manifest.json` is absent.

## Run the single-client experiment

    python -m clients.local_train --language tamil                      # research settings (configs/single_client.yaml)
    python -m clients.local_train --config configs/dev_real.yaml --language tamil    # tiny run, real model

Every tunable can be set in the YAML or overridden on the command line: `--model --max-seq-len --device --precision
--seed --epochs --max-steps --batch-size --grad-accum --lr --lora-r --lora-alpha --lora-dropout --max-train-samples
--max-val-samples --results-dir --checkpoints-dir --clients-dir`.
Precision is `auto | fp32 | bf16` (`auto` = bf16 on CUDA GPUs that support it, otherwise fp32; fp16 is rejected on
purpose because the training loop has no loss scaling).

Outputs per client id (e.g. `ta`): `results/.../ta/metrics.json` (includes environment, peak memory, resolved LoRA
settings, whether the LoRA weights actually changed), `log.jsonl` (rewritten each run), `config_used.json`; the LoRA
adapter (adapter only, not the base model) in `checkpoints/.../ta/adapter/`.

## Full verification on a machine with model access (produces the real evidence)

    python -m data.prepare_data --config configs/data.yaml
    python -m experiments.verify_single_client --config configs/dev_real.yaml \
        --languages english hindi tamil telugu malayalam

Writes `results/verification/report.json`. Stages: tokenizer; model load + forward pass; per-language data checks and
**base-model** loss/perplexity; LoRA attach, freezing, init-equals-base, backward; a tiny real training run (loss per
step); adapter reload; an independent unpadded perplexity computation compared with the batched padded one; adapter
size vs base size. A failing stage records its exact error and does not stop the others. If the model cannot be
downloaded, the report will say so; do not fill in numbers by hand.

## Configurations

| File | Purpose |
|---|---|
| `configs/base.yaml` | shared defaults: seed, device, precision, paths, LoRA r/alpha/dropout, federation placeholders |
| `configs/single_client.yaml` | **research** single-client settings: model `bigscience/bloom-560m`, seq len 128, batch 4 x accum 2, lr 2e-4 |
| `configs/dev_real.yaml` | **development**: same real model, tiny settings (seq 64, batch 2, 5 steps, 32/16 samples), separate output dirs |
| `configs/dev.yaml` | **development, offline**: `tiny-offline` random-init GPT-2 + byte tokenizer; checks code paths only |
| `configs/data.yaml` | client languages, scripts, passage counts (resource level is simulated by `n_train`) |

**Development model.** If `bloom-560m` does not fit in memory, override it, e.g. `--model <smaller causal LM>`; LoRA
targets are auto-chosen for gpt2, bloom, llama, qwen2, mistral, gemma, opt, gpt_neox (otherwise set
`model.lora_target_modules`). No alternative model has been tested here, so none is claimed to work.
**Research model.** `bigscience/bloom-560m` (LoRA target `query_key_value`). Its coverage of Hindi/Tamil/Telugu/Malayalam
is documented by its authors but has **not been verified in this repo**; `verify_single_client` measures tokens-per-word
and base-model perplexity per language so that this can be checked.
Memory rule of thumb (arithmetic from the name, not a measurement): ~0.56 B parameters x 4 bytes (fp32) is ~2.2 GB of
weights before activations, LoRA optimizer state and the framework. The real peak is recorded in `metrics.json`.

## Notes on what the numbers mean

* Perplexity = exp(token-weighted mean validation loss); padding positions are masked out via the attention mask (not by
  pad id, so a real EOS or a pad-equal-to-EOS tokenizer is handled correctly).
* Perplexity is per *subword token*. Different tokenizers, and different languages under one tokenizer, split text into
  different numbers of tokens, so perplexities are **not comparable across languages** without normalising (e.g. per
  character/byte). Do not read a language ranking off them.
* A truncated passage is no longer given an artificial EOS at the cut point; EOS is appended only when the passage fits.
* A falling training loss shows the optimizer works; it does not show the proposed federated method is effective.
