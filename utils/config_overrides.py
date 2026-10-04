"""Command-line overrides for config values (kept torch-free so it is unit-testable anywhere)."""

# CLI attribute name -> path inside the loaded config dict
OVERRIDES = {
    "model": ("model", "name"),
    "max_seq_len": ("model", "max_seq_len"),
    "device": ("device",),
    "precision": ("precision",),
    "seed": ("seed",),
    "epochs": ("train", "epochs"),
    "max_steps": ("train", "max_steps"),
    "batch_size": ("train", "batch_size"),
    "grad_accum": ("train", "grad_accum_steps"),
    "lr": ("train", "learning_rate"),
    "lora_r": ("lora", "r"),
    "lora_alpha": ("lora", "alpha"),
    "lora_dropout": ("lora", "dropout"),
    "max_train_samples": ("data", "max_train_samples"),
    "max_val_samples": ("data", "max_validation_samples"),
    "results_dir": ("output", "results_dir"),
    "checkpoints_dir": ("output", "checkpoints_dir"),
}


def apply_overrides(cfg: dict, **values) -> dict:
    """Set cfg[path] = value for every non-None value. Unknown names raise KeyError."""
    for name, value in values.items():
        path = OVERRIDES[name]
        if value is None:
            continue
        node = cfg
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = value
    return cfg
