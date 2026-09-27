"""Frozen functions from existing run_experiments.py; see provenance.json. Only optional diagnostic hooks added."""
from __future__ import annotations
from typing import Sequence, Tuple
import numpy as np

def normalize_columns(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(A, axis=0, keepdims=True)
    return A / np.maximum(norms, eps)

def kmeans(
    X: np.ndarray,
    k: int,
    seed: int,
    n_init: int = 20,
    max_iter: int = 200,
) -> np.ndarray:
    """Small deterministic k-means for rows of X."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    best_labels = None
    best_obj = np.inf
    for _ in range(n_init):
        centers = np.empty((k, X.shape[1]), dtype=np.float64)
        first = rng.integers(n)
        centers[0] = X[first]
        dist2 = np.sum((X - centers[0]) ** 2, axis=1)
        for j in range(1, k):
            probs = dist2 / max(dist2.sum(), 1e-12)
            idx = rng.choice(n, p=probs)
            centers[j] = X[idx]
            dist2 = np.minimum(dist2, np.sum((X - centers[j]) ** 2, axis=1))

        labels = np.zeros(n, dtype=int)
        for _iter in range(max_iter):
            d2 = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
            new_labels = d2.argmin(axis=1)
            if np.array_equal(labels, new_labels) and _iter > 0:
                break
            labels = new_labels
            for j in range(k):
                mask = labels == j
                if mask.any():
                    centers[j] = X[mask].mean(axis=0)
                else:
                    centers[j] = X[rng.integers(n)]
        obj = ((X - centers[labels]) ** 2).sum()
        if obj < best_obj:
            best_obj = obj
            best_labels = labels.copy()
    assert best_labels is not None
    return best_labels

def geometric_median(
    points: np.ndarray,
    max_iter: int = 300,
    tol: float = 1e-7,
    diagnostics=None,
) -> np.ndarray:
    """Weiszfeld geometric median for rows of points."""
    if points.shape[0] == 1:
        if diagnostics is not None:
            diagnostics.append(dict(iterations=0, converged=True, reason="singleton", tol=tol, max_iter=max_iter))
        return points[0].copy()
    q = np.median(points, axis=0)
    converged = False
    for _ in range(max_iter):
        dist = np.linalg.norm(points - q, axis=1)
        if dist.min() < 1e-12:
            if diagnostics is not None:
                diagnostics.append(dict(iterations=_+1, converged=True, reason="point_hit", tol=tol, max_iter=max_iter))
            return points[dist.argmin()].copy()
        w = 1.0 / np.maximum(dist, 1e-12)
        q_new = (points * w[:, None]).sum(axis=0) / w.sum()
        if np.linalg.norm(q_new - q) <= tol * max(1.0, np.linalg.norm(q)):
            q = q_new
            converged = True
            break
        q = q_new
    if diagnostics is not None:
        diagnostics.append(dict(iterations=_+1, converged=converged, reason="tolerance" if converged else "max_iter", tol=tol, max_iter=max_iter))
    return q

def prepare_srf_clusters(
    atoms: Sequence[np.ndarray],
    rank: int,
    seed: int,
    diagnostics=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute one spectral partition and its sign-aligned atom matrix."""
    Ac = normalize_columns(np.concatenate(atoms, axis=1))
    R = Ac.shape[1]
    M = np.abs(Ac.T @ Ac)
    vals, vecs = np.linalg.eigh(M)
    top = np.argsort(vals)[-rank:][::-1]
    U = vecs[:, top]
    D = vals[top]
    Q = U * D[None, :]
    labels = kmeans(Q, rank, seed=seed)
    if diagnostics is not None:
        diagnostics.update(affinity=M, embedding=Q, eigenvalues=D)
    aligned = Ac.copy()
    for a in range(rank):
        idx = np.flatnonzero(labels == a)
        if len(idx) > 1:
            u, _, _ = np.linalg.svd(Ac[:, idx], full_matrices=False)
            signs = np.sign(u[:, 0] @ Ac[:, idx])
            signs[signs == 0] = 1.0
            aligned[:, idx] *= signs[None, :]
    return Ac, labels, aligned

def aggregate_srf_clusters(
    Ac: np.ndarray, labels: np.ndarray, rank: int, aggregate: str = "gm", diagnostics=None
) -> np.ndarray:
    """Reduce an existing partition without changing assignments or signs."""
    out = []
    R = Ac.shape[1]
    all_indices = np.arange(R)
    for a in range(rank):
        idx = all_indices[labels == a]
        if len(idx) == 0:
            idx = np.array([all_indices[a % R]])
        block = Ac[:, idx].copy()
        if aggregate == "mean":
            atom = block.mean(axis=1)
        elif aggregate == "gm":
            atom = geometric_median(block.T, diagnostics=diagnostics)
        else:
            raise ValueError(f"Unknown aggregate {aggregate}")
        out.append(atom)
    return normalize_columns(np.column_stack(out))

def aggregate_naive(
    atoms: Sequence[np.ndarray],
    method: str = "mean",
    diagnostics=None,
) -> np.ndarray:
    stack = np.stack(atoms, axis=0)
    if method == "mean":
        A = stack.mean(axis=0)
    elif method == "median":
        A = np.column_stack(
            [geometric_median(stack[:, :, j], diagnostics=diagnostics) for j in range(stack.shape[2])]
        )
    else:
        raise ValueError(method)
    return normalize_columns(A)
