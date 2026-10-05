"""Server side of the proposed method: per-cluster FedAvg + alpha-mixed global adapter.

Interpretation (stated explicitly because the update symbol is overloaded): the LoRA tensors ARE the learned
delta on top of the frozen base model, so Delta_c is cluster c's aggregated LoRA tensor set and Delta_global is the
global LoRA tensor set. With N_c = training samples held by cluster c and N = sum_c N_c:

    Delta_global(t) = alpha * sum_c (N_c / N) * Delta_c(t)  +  (1 - alpha) * Delta_global(t-1)

alpha = 1 gives plain size-weighted FedAvg of the cluster adapters; alpha = 0 freezes the global adapter.
Torch-only (no peft / transformers).
"""
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from server.fedavg import AdapterState, aggregate_lora_adapters
from utils.checkpoints import save_json, save_state


def aggregate_within_clusters(client_adapters: Sequence[AdapterState], client_weights: Sequence[float],
                              client_cluster: Sequence[int]) -> Dict[int, Dict[str, torch.Tensor]]:
    """FedAvg inside every cluster: {cluster id: weighted average of its clients' adapters}.
    client_weights are typically training-set sizes; each cluster's weights are normalised on their own."""
    if not (len(client_adapters) == len(client_weights) == len(client_cluster)):
        raise ValueError("client_adapters, client_weights and client_cluster must have the same length")
    out = {}
    for k in sorted(set(client_cluster)):
        idx = [i for i, c in enumerate(client_cluster) if c == k]
        out[k] = aggregate_lora_adapters([client_adapters[i] for i in idx], [client_weights[i] for i in idx])
    return out


def alpha_mix(cluster_adapters: Mapping[int, AdapterState], cluster_sizes: Mapping[int, float],
              previous_global: AdapterState, alpha: float) -> Dict[str, torch.Tensor]:
    """Delta_global = alpha * sum_c (N_c/N) Delta_c + (1 - alpha) * previous_global  (float64 math, dtype preserved)."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    ks = sorted(cluster_adapters)
    if set(ks) != set(cluster_sizes):
        raise ValueError("cluster_adapters and cluster_sizes must have the same cluster ids")
    sizes = [float(cluster_sizes[k]) for k in ks]            # normalised (N_c / N) inside aggregate_lora_adapters
    cluster_avg = aggregate_lora_adapters([cluster_adapters[k] for k in ks], sizes)
    if set(previous_global) != set(cluster_avg):
        raise ValueError("previous_global has different parameter names than the cluster adapters")
    out = {}
    for k, avg in cluster_avg.items():
        prev = previous_global[k]
        if tuple(prev.shape) != tuple(avg.shape):
            raise ValueError(f"shape mismatch for '{k}': previous_global {tuple(prev.shape)} vs {tuple(avg.shape)}")
        out[k] = (alpha * avg.double() + (1.0 - alpha) * prev.detach().to("cpu", torch.float64)).to(avg.dtype)
    return out


def cluster_round(client_adapters: Sequence[AdapterState], client_weights: Sequence[float], client_cluster: Sequence[int],
                  previous_cluster_adapters: Mapping[int, AdapterState], cluster_sizes: Mapping[int, float],
                  previous_global: AdapterState, alpha: float):
    """One server step. Clusters that had no client this round keep their previous adapter (stale) and still
    contribute to the global mix with their static size N_c. Returns (new_cluster_adapters, new_global)."""
    updated = aggregate_within_clusters(client_adapters, client_weights, client_cluster) if client_adapters else {}
    new_clusters = {k: updated.get(k, previous_cluster_adapters[k]) for k in cluster_sizes}
    return new_clusters, alpha_mix(new_clusters, cluster_sizes, previous_global, alpha)


def save_cluster_checkpoints(ckpt_dir: str, cluster_adapters: Mapping[int, AdapterState], global_adapter: AdapterState,
                             clusters: Mapping[int, List[str]], cluster_sizes: Mapping[int, float], alpha: float,
                             extra: Optional[dict] = None) -> None:
    """Raw LoRA tensors: <dir>/cluster_<k>/adapter.pt, <dir>/global_adapter/adapter.pt, plus <dir>/clusters.json."""
    root = Path(ckpt_dir)
    for k, st in cluster_adapters.items():
        save_state(dict(st), str(root / f"cluster_{k}" / "adapter.pt"))
    save_state(dict(global_adapter), str(root / "global_adapter" / "adapter.pt"))
    save_json({"clusters": {str(k): v for k, v in clusters.items()},
               "cluster_sizes": {str(k): v for k, v in cluster_sizes.items()}, "alpha": alpha, **(extra or {})},
              str(root / "clusters.json"))
