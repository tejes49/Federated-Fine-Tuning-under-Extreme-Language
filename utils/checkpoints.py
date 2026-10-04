"""Config loading (with `extends`) and simple JSON/tensor checkpoint helpers."""
import json
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    """Load a YAML config; if it has `extends: other.yaml`, merge over that file."""
    p = Path(path)
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("extends", None)
    if parent:
        base = load_config(str(p.parent / parent))
        return _deep_merge(base, cfg)
    return cfg


def _deep_merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in b.items():
        out[k] = _deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def save_json(obj, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def save_state(state_dict, path: str) -> None:
    import torch
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state_dict, path)
