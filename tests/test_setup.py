"""Phase 1 tests: config inheritance, seeding, device detection."""
import random
from pathlib import Path

import numpy as np

from utils.checkpoints import load_config
from utils.reproducibility import get_device, set_seed

CFG = Path(__file__).resolve().parent.parent / "configs"


def test_baseline_extends_base():
    cfg = load_config(str(CFG / "baseline.yaml"))
    assert cfg["seed"] == 42 and cfg["lora"]["r"] == 8   # inherited from base
    assert cfg["clustering"] is False and cfg["method"] == "fedavg_lora"


def test_full_model_flags():
    cfg = load_config(str(CFG / "full_model.yaml"))
    assert cfg["clustering"] and cfg["global_adapter"] and cfg["tokenizer_alignment"]


def test_seed_reproducible():
    set_seed(1); a = (random.random(), np.random.rand())
    set_seed(1); b = (random.random(), np.random.rand())
    assert a == b


def test_device_is_valid():
    assert get_device("auto") in ("cpu", "cuda")
    assert get_device("cpu") == "cpu"
