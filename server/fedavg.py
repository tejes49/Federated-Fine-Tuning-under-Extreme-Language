"""Server-side FedAvg for LoRA adapters.

`aggregate_lora_adapters` performs the standard FedAvg weighted parameter average (McMahan et al., 2017)
independently on every LoRA tensor (lora_A and lora_B are averaged separately, as in plain FedAvg-LoRA;
the average of products is NOT the product of averages, which is a known property of this baseline).

Only torch is needed here (no peft / transformers), so it is cheap to import and test.
"""
import math
from typing import Dict, List, Mapping, Optional, Sequence

import torch

AdapterState = Mapping[str, torch.Tensor]


def normalize_weights(weights: Optional[Sequence[float]], n: int) -> List[float]:
    """Return n non-negative weights summing to 1 (uniform when `weights` is None)."""
    if n < 1:
        raise ValueError("need at least one client adapter to aggregate")
    if weights is None:
        return [1.0 / n] * n
    w = [float(x) for x in weights]
    if len(w) != n:
        raise ValueError(f"got {len(w)} weights for {n} client adapters")
    if any(math.isnan(x) or math.isinf(x) or x < 0 for x in w):
        raise ValueError(f"weights must be finite and non-negative, got {w}")
    total = sum(w)
    if total <= 0:
        raise ValueError("weights must sum to a positive number")
    return [x / total for x in w]


def _check_compatible(client_adapters: Sequence[AdapterState]) -> None:
    ref = client_adapters[0]
    ref_keys = set(ref.keys())
    if not ref_keys:
        raise ValueError("client adapter 0 is empty")
    for i, a in enumerate(client_adapters[1:], start=1):
        keys = set(a.keys())
        if keys != ref_keys:
            missing, extra = sorted(ref_keys - keys), sorted(keys - ref_keys)
            raise ValueError(f"client adapter {i} has different parameter names "
                             f"(missing {missing[:3]}, unexpected {extra[:3]})")
    for k in ref_keys:
        for i, a in enumerate(client_adapters):
            if tuple(a[k].shape) != tuple(ref[k].shape):
                raise ValueError(f"shape mismatch for '{k}': client {i} has {tuple(a[k].shape)}, "
                                 f"client 0 has {tuple(ref[k].shape)}")


def aggregate_lora_adapters(client_adapters: Sequence[AdapterState],
                            weights: Optional[Sequence[float]] = None) -> Dict[str, torch.Tensor]:
    """Weighted average of client LoRA state_dicts.

    client_adapters: list of {param_name: tensor} (e.g. from models.lora.adapter_state).
    weights: one non-negative number per client (e.g. number of training samples); normalised to sum to 1.
             None = uniform average.
    Returns a new dict on CPU with the dtype of client 0's tensors; inputs are never modified.
    Accumulation is done in float64 so the result does not depend on client order beyond rounding.
    """
    adapters = list(client_adapters)
    w = normalize_weights(weights, len(adapters))
    _check_compatible(adapters)
    out: Dict[str, torch.Tensor] = {}
    for k, ref in adapters[0].items():
        acc = torch.zeros(ref.shape, dtype=torch.float64)
        for wi, a in zip(w, adapters):
            t = a[k].detach().to("cpu", torch.float64)
            if not torch.isfinite(t).all():
                raise ValueError(f"non-finite values in client adapter for '{k}'")
            acc.add_(t, alpha=wi)
        out[k] = acc.to(ref.dtype)
    return out


def adapter_num_bytes(state: AdapterState) -> int:
    """Size of an adapter payload in bytes (what one upload / download would transfer, uncompressed)."""
    return sum(t.numel() * t.element_size() for t in state.values())
