"""Seeding and device selection (Rules 4 and 5)."""
import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    """Seed python, numpy and (if installed) torch for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:  # ImportError or a broken install (OSError)
        pass


def get_device(preference: str = "auto") -> str:
    """Return 'cuda' or 'cpu'. 'auto' picks CUDA when available, else CPU."""
    if preference == "cpu":
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if preference == "cuda":
            print("[warn] cuda requested but unavailable; falling back to cpu")
    except Exception:  # ImportError or a broken install (OSError)
        pass
    return "cpu"
