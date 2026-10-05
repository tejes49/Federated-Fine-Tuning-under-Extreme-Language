"""Tokenizer alignment (proposed method, step 2): a lightweight linear projection from a client's subword
embedding space into a common multilingual embedding space.

Local tokenizers are never touched or re-trained. A client keeps its own tokenizer and embedding table; the only
thing learned is a d_in x d_common matrix W. Sentence-level supervision is used: the mean of a passage's subword
embeddings (client tokenizer + client embedding table) is regressed onto a reference sentence embedding of the same
passage (e.g. LaBSE). `project_token_embeddings` then maps the whole client vocabulary into the common space.

This module is a standalone component: the shared LoRA training in this repo does not consume the projection
(all clients here use one tokenizer), so it is evaluated by alignment quality only.
"""
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn


class SubwordProjection(nn.Module):
    """y = x @ W^T, no bias. Works on [..., d_in] tensors; trainable by gradient or fit in closed form."""

    def __init__(self, d_in: int, d_common: int):
        super().__init__()
        self.d_in, self.d_common = d_in, d_common
        self.proj = nn.Linear(d_in, d_common, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)

    @property
    def num_parameters(self) -> int:
        return self.d_in * self.d_common


def pool_subword_embeddings(embedding_matrix: torch.Tensor, id_lists: Sequence[Sequence[int]]) -> torch.Tensor:
    """Mean subword embedding per passage -> [n_passages, d_in] (float32). Empty passages are rejected."""
    rows = []
    for ids in id_lists:
        if len(ids) == 0:
            raise ValueError("empty token id list")
        rows.append(embedding_matrix[torch.as_tensor(list(ids), dtype=torch.long)].float().mean(dim=0))
    return torch.stack(rows)


def fit_projection(x: torch.Tensor, y: torch.Tensor, ridge: float = 1e-2) -> SubwordProjection:
    """Closed-form ridge regression  W = argmin ||x W^T - y||^2 + ridge ||W||^2  (float64 solve).
    Uses the smaller of the primal / dual systems, so it also works when n_passages < d_in."""
    if x.ndim != 2 or y.ndim != 2 or x.shape[0] != y.shape[0]:
        raise ValueError(f"expected x [n, d_in] and y [n, d_common] with equal n, got {tuple(x.shape)} / {tuple(y.shape)}")
    if ridge <= 0:
        raise ValueError("ridge must be > 0")
    X, Y = x.double(), y.double()
    n, d = X.shape
    if d <= n:
        w = torch.linalg.solve(X.T @ X + ridge * torch.eye(d, dtype=torch.float64), X.T @ Y)   # [d_in, d_common]
    else:
        w = X.T @ torch.linalg.solve(X @ X.T + ridge * torch.eye(n, dtype=torch.float64), Y)
    p = SubwordProjection(d, Y.shape[1])
    with torch.no_grad():
        p.proj.weight.copy_(w.T.float())
    return p


@torch.no_grad()
def project_token_embeddings(proj: SubwordProjection, embedding_matrix: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """Whole-vocabulary map: [V, d_in] -> [V, d_common] (chunked so a 250k-token table does not spike memory)."""
    return torch.cat([proj(embedding_matrix[i:i + chunk].float()) for i in range(0, embedding_matrix.shape[0], chunk)])


@torch.no_grad()
def alignment_quality(proj: SubwordProjection, x: torch.Tensor, y: torch.Tensor) -> Dict[str, float]:
    """Mean cosine between projected pooled embeddings and the reference sentence embeddings, plus a
    shuffled-pair baseline (each passage compared with another passage's target, offset by n//2) for context."""
    p = nn.functional.normalize(proj(x.float()), dim=-1)
    t = nn.functional.normalize(y.float(), dim=-1)
    out = {"mean_cosine": float((p * t).sum(-1).mean())}
    if len(p) > 1:
        out["shuffled_baseline_cosine"] = float((p * t.roll(max(1, len(t) // 2), dims=0)).sum(-1).mean())
    return out


def fit_client_projection(embedding_matrix: torch.Tensor, tokenizer, train_texts: List[str], train_targets: np.ndarray,
                          val_texts: List[str], val_targets: np.ndarray, max_len: int = 128,
                          ridge: float = 1e-2) -> Dict[str, object]:
    """Fit one client's projection using ITS OWN tokenizer (unchanged) and embedding table; report held-out quality."""
    def ids_of(texts):
        out = [tokenizer(t, add_special_tokens=False, truncation=True, max_length=max_len)["input_ids"] for t in texts]
        keep = [i for i, ids in enumerate(out) if ids]
        return [out[i] for i in keep], keep

    tr_ids, tr_keep = ids_of(train_texts)
    va_ids, va_keep = ids_of(val_texts)
    if not tr_ids or not va_ids:
        raise ValueError("no tokenizable passages for alignment")
    xt = pool_subword_embeddings(embedding_matrix, tr_ids)
    yt = torch.as_tensor(np.asarray(train_targets)[tr_keep], dtype=torch.float32)
    xv = pool_subword_embeddings(embedding_matrix, va_ids)
    yv = torch.as_tensor(np.asarray(val_targets)[va_keep], dtype=torch.float32)
    proj = fit_projection(xt, yt, ridge)
    return {"projection": proj, "train": alignment_quality(proj, xt, yt), "heldout": alignment_quality(proj, xv, yv),
            "n_train": len(tr_ids), "n_heldout": len(va_ids), "parameters": proj.num_parameters}
