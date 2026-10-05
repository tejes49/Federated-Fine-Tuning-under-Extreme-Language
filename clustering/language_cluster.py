"""Language-aware client clustering (proposed method, step 1).

Each client gets a *language fingerprint*: the mean (L2-normalised) sentence embedding of a sample of its
training passages. Clients are then grouped into language clusters using BOTH

  * declared metadata  - clients that declare the same language are always in the same cluster, and
                         languages of the same family get a similarity bonus (`family_weight`);
  * embedding similarity - cosine similarity between the language fingerprints.

Clustering is average-linkage agglomerative over *language groups*, stopping when the best remaining
combined similarity is below `similarity_threshold` (or when `n_clusters` is reached, if given).

Encoders:
  * `SentenceTransformerEmbedder("sentence-transformers/LaBSE")` - the intended encoder (LaBSE covers
    hi/ta/te/ml; `paraphrase-multilingual-MiniLM-L12-v2` does NOT list Tamil, Telugu or Malayalam). Needs a
    Hugging Face download.
  * `HashingEmbedder` ("offline-hash") - deterministic character-n-gram hashing, no download. For development
    and tests only; it carries no semantic information, so clusters from it say nothing about languages.

The similarity threshold / family weight are NOT calibrated on any data yet.
"""
import hashlib
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

OFFLINE_HASH = "offline-hash"

# Declared linguistic family per language code (metadata; override via config clustering.families).
LANG_FAMILY = {"en": "germanic", "hi": "indo-aryan", "ta": "dravidian", "te": "dravidian", "ml": "dravidian"}


# ---------- encoders ----------
class HashingEmbedder:
    """Signed character 1-3-gram hashing into `dim` buckets, L2-normalised. Deterministic, offline, NOT semantic."""

    name = OFFLINE_HASH
    is_semantic = False

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for i, t in enumerate(texts):
            for n in (1, 2, 3):
                for j in range(len(t) - n + 1):
                    h = int.from_bytes(hashlib.blake2b(t[j:j + n].encode("utf-8"), digest_size=8).digest(), "little")
                    out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        return _l2(out)


class SentenceTransformerEmbedder:
    """Thin wrapper over sentence_transformers.SentenceTransformer (name or local path)."""

    is_semantic = True

    def __init__(self, name: str, device: Optional[str] = None, batch_size: int = 32, model=None):
        self.name, self.batch_size = name, batch_size
        if model is None:
            from sentence_transformers import SentenceTransformer  # imported lazily: heavy, optional
            model = SentenceTransformer(name, device=device)
        self.model = model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        e = self.model.encode(list(texts), batch_size=self.batch_size, convert_to_numpy=True,
                              show_progress_bar=False, normalize_embeddings=False)
        return _l2(np.asarray(e, dtype=np.float64))


def make_embedder(name: str, **kw):
    return HashingEmbedder(kw.get("dim", 256)) if name == OFFLINE_HASH else SentenceTransformerEmbedder(
        name, device=kw.get("device"), batch_size=kw.get("batch_size", 32))


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


# ---------- fingerprints ----------
def language_fingerprint(embeddings: np.ndarray) -> np.ndarray:
    """Mean of the passage embeddings, re-normalised. embeddings: [n_passages, d]."""
    e = np.asarray(embeddings, dtype=np.float64)
    if e.ndim != 2 or e.shape[0] == 0:
        raise ValueError("need a non-empty [n_passages, d] embedding matrix")
    return _l2(e.mean(axis=0, keepdims=True))[0]


def compute_fingerprints(embedder, texts_by_client: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
    return {cid: language_fingerprint(embedder.encode(texts)) for cid, texts in texts_by_client.items()}


# ---------- clustering ----------
@dataclass
class ClusterAssignment:
    cluster_of: Dict[str, int]                      # client id -> cluster index
    clusters: Dict[int, List[str]]                  # cluster index -> client ids
    similarity: Dict[str, Dict[str, float]] = field(default_factory=dict)   # combined group similarity (languages)
    cosine: Dict[str, Dict[str, float]] = field(default_factory=dict)       # raw fingerprint cosine (languages)
    settings: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"cluster_of": self.cluster_of, "clusters": {str(k): v for k, v in self.clusters.items()},
                "similarity": self.similarity, "cosine": self.cosine, "settings": self.settings}


def cluster_clients(client_ids: Sequence[str], languages: Dict[str, str], fingerprints: Dict[str, np.ndarray],
                    families: Optional[Dict[str, str]] = None, similarity_threshold: float = 0.5,
                    family_weight: float = 0.5, n_clusters: Optional[int] = None,
                    center: bool = True) -> ClusterAssignment:
    """Group clients into language clusters (see module docstring).

    languages: client id -> declared language code. fingerprints: client id -> vector.
    combined similarity between two language groups = (1 - family_weight) * cosine + family_weight * same_family,
    where cosine is computed on fingerprints (mean-centred across language groups when `center`, which removes the
    direction all texts share) and same_family is 1 if their declared families match, else 0.
    """
    if not 0.0 <= family_weight <= 1.0:
        raise ValueError("family_weight must be in [0, 1]")
    ids = list(client_ids)
    if not ids:
        raise ValueError("no clients to cluster")
    missing = [c for c in ids if c not in languages or c not in fingerprints]
    if missing:
        raise ValueError(f"missing language/fingerprint for clients {missing}")
    fam = {**LANG_FAMILY, **(families or {})}

    langs: List[str] = []                           # language groups, in first-seen order
    for c in ids:
        if languages[c] not in langs:
            langs.append(languages[c])
    members = {l: [c for c in ids if languages[c] == l] for l in langs}
    fp = np.stack([_l2(np.mean([_l2(np.asarray(fingerprints[c], dtype=np.float64)[None])[0] for c in members[l]],
                               axis=0, keepdims=True))[0] for l in langs])
    if center and len(langs) > 2:                   # centring is degenerate for <=2 groups
        fp = _l2(fp - fp.mean(axis=0, keepdims=True))
    cos = fp @ fp.T
    same = np.array([[1.0 if fam.get(a) is not None and fam.get(a) == fam.get(b) else 0.0 for b in langs] for a in langs])
    sim = (1.0 - family_weight) * cos + family_weight * same

    groups: List[List[int]] = [[i] for i in range(len(langs))]
    target = n_clusters if n_clusters is not None else 1
    if n_clusters is not None and not 1 <= n_clusters <= len(langs):
        raise ValueError(f"n_clusters must be in [1, {len(langs)}] (number of distinct declared languages)")
    while len(groups) > target:
        best, pair = -math.inf, None
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                s = float(np.mean([sim[i, j] for i in groups[a] for j in groups[b]]))
                if s > best + 1e-12:
                    best, pair = s, (a, b)
        if n_clusters is None and best < similarity_threshold:
            break
        a, b = pair
        groups[a] = groups[a] + groups[b]
        del groups[b]

    order = sorted(groups, key=min)                 # deterministic cluster ids: by first language group
    clusters: Dict[int, List[str]] = {}
    for k, g in enumerate(order):
        clusters[k] = [c for i in sorted(g) for c in members[langs[i]]]
    cluster_of = {c: k for k, cs in clusters.items() for c in cs}
    mat = lambda m: {a: {b: float(m[i, j]) for j, b in enumerate(langs)} for i, a in enumerate(langs)}
    return ClusterAssignment(cluster_of, clusters, mat(sim), mat(cos),
                             {"similarity_threshold": similarity_threshold, "family_weight": family_weight,
                              "n_clusters": n_clusters, "center": center, "languages": langs})
