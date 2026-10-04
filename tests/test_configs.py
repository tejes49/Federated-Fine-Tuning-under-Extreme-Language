"""Torch-free tests: configs load, research vs dev separation, CLI overrides, resource reporting."""
from pathlib import Path

import pytest

from utils.checkpoints import load_config
from utils.config_overrides import OVERRIDES, apply_overrides
from utils.resources import describe_environment, peak_memory_gb

CFG = Path(__file__).resolve().parent.parent / "configs"
REQUIRED = [("model", "name"), ("model", "max_seq_len"), ("lora", "r"), ("lora", "alpha"), ("lora", "dropout"),
            ("train", "batch_size"), ("train", "grad_accum_steps"), ("train", "learning_rate"),
            ("train", "epochs"), ("train", "max_steps"), ("data", "clients_dir"), ("data", "max_train_samples"),
            ("output", "results_dir"), ("output", "checkpoints_dir"), ("seed",), ("device",), ("precision",)]


def _get(cfg, path):
    for k in path:
        cfg = cfg[k]
    return cfg


@pytest.mark.parametrize("name", ["single_client.yaml", "dev.yaml", "dev_real.yaml"])
def test_training_configs_define_every_tunable(name):
    cfg = load_config(str(CFG / name))
    for path in REQUIRED:
        _get(cfg, path)          # KeyError = a parameter is not configurable


def test_research_and_dev_configs_are_separate():
    research, dev, dev_real = (load_config(str(CFG / n)) for n in ("single_client.yaml", "dev.yaml", "dev_real.yaml"))
    assert research["model"]["name"] == "bigscience/bloom-560m"
    assert dev["model"]["name"] == "tiny-offline"                      # random-init offline model
    assert dev_real["model"]["name"] == research["model"]["name"]      # real model, tiny settings
    assert dev_real["train"]["max_steps"] is not None and research["train"]["max_steps"] is None
    assert dev_real["output"]["results_dir"] != research["output"]["results_dir"]   # dev runs never overwrite research runs


def test_apply_overrides():
    cfg = load_config(str(CFG / "single_client.yaml"))
    apply_overrides(cfg, model="x/y", lora_r=4, lr=1e-3, seed=0, max_steps=None, precision="fp32")
    assert cfg["model"]["name"] == "x/y" and cfg["lora"]["r"] == 4 and cfg["seed"] == 0
    assert cfg["train"]["learning_rate"] == 1e-3 and cfg["train"]["max_steps"] is None   # None = keep
    assert cfg["precision"] == "fp32"
    with pytest.raises(KeyError):
        apply_overrides(cfg, not_an_option=1)
    assert all(isinstance(p, tuple) for p in OVERRIDES.values())


def test_resource_reports_have_expected_keys():
    env = describe_environment()
    assert {"python", "cpu_count", "ram_total_gb", "torch", "cuda_available", "gpu_name"} <= set(env)
    assert set(peak_memory_gb()) == {"peak_process_rss_gb", "peak_cuda_allocated_gb"}
