"""Symmetric non-negative matrix factorization for clustering."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatMatrix = NDArray[np.float64]


def _as_points(data: ArrayLike) -> FloatMatrix:
    points = np.asarray(data, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2 or points.shape[1] < 1:
        raise ValueError("data must be a matrix with at least two rows and one column")
    if not np.isfinite(points).all():
        raise ValueError("data must contain only finite values")
    return points


def similarity_matrix(data: ArrayLike) -> FloatMatrix:
    """Return the dense Gaussian similarity matrix with a zero diagonal."""
    points = _as_points(data)
    squared_norms = np.einsum("ij,ij->i", points, points)
    distances = (
        squared_norms[:, None]
        + squared_norms[None, :]
        - 2.0 * points @ points.T
    )
    np.maximum(distances, 0.0, out=distances)
    similarities = np.exp(-distances / 2.0)
    np.fill_diagonal(similarities, 0.0)
    return similarities


def degree_matrix(similarities: ArrayLike) -> FloatMatrix:
    """Return the diagonal matrix of row sums of a similarity matrix."""
    matrix = _as_similarity(similarities)
    return np.diag(matrix.sum(axis=1))


def normalized_similarity_matrix(similarities: ArrayLike) -> FloatMatrix:
    """Return D^-1/2 A D^-1/2, using zeros for isolated vertices."""
    matrix = _as_similarity(similarities)
    degrees = matrix.sum(axis=1)
    inverse_sqrt = np.zeros_like(degrees)
    positive = degrees > 0.0
    inverse_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    return inverse_sqrt[:, None] * matrix * inverse_sqrt[None, :]


def _as_similarity(data: ArrayLike) -> FloatMatrix:
    matrix = np.asarray(data, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError("similarities must be a square matrix of size at least two")
    if not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise ValueError("similarities must contain finite, non-negative values")
    if not np.allclose(matrix, matrix.T):
        raise ValueError("similarities must be symmetric")
    return matrix


def symnmf_factorization(
    similarities: ArrayLike,
    k: int,
    *,
    max_iter: int = 300,
    tol: float = 1e-4,
    beta: float = 0.5,
    seed: int | None = 0,
) -> FloatMatrix:
    """Factor a symmetric non-negative matrix W into H @ H.T.

    Uses the multiplicative SymNMF update with a convex step size beta.
    """
    matrix = _as_similarity(similarities)
    n = matrix.shape[0]
    if (
        not isinstance(k, (int, np.integer))
        or isinstance(k, (bool, np.bool_))
        or not 1 <= k <= n
    ):
        raise ValueError("k must be an integer between 1 and the number of rows")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol <= 0.0:
        raise ValueError("tol must be finite and positive")
    if not np.isfinite(beta) or not 0.0 < beta <= 1.0:
        raise ValueError("beta must be in (0, 1]")

    scale = float(np.sqrt(matrix.mean() / k))
    if scale == 0.0:
        return np.zeros((n, k), dtype=np.float64)

    rng = np.random.default_rng(seed)
    h = rng.uniform(0.0, 2.0 * scale, size=(n, k))
    epsilon = np.finfo(np.float64).tiny

    for _ in range(max_iter):
        denominator = (h @ h.T) @ h
        ratio = (matrix @ h) / np.maximum(denominator, epsilon)
        updated = h * ((1.0 - beta) + beta * ratio)
        if np.linalg.norm(updated - h) < tol:
            h = updated
            break
        h = updated

    return h


def assign_clusters(factors: ArrayLike) -> NDArray[np.int64]:
    """Assign each row to the factor column with the largest value."""
    matrix = np.asarray(factors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("factors must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all() or (matrix < 0.0).any():
        raise ValueError("factors must contain finite, non-negative values")
    return np.argmax(matrix, axis=1)


def symnmf(
    data: ArrayLike,
    k: int,
    *,
    max_iter: int = 300,
    tol: float = 1e-4,
    beta: float = 0.5,
    seed: int | None = 0,
) -> FloatMatrix:
    """Build the normalized similarity matrix and factor it into k clusters."""
    normalized = normalized_similarity_matrix(similarity_matrix(data))
    return symnmf_factorization(
        normalized, k, max_iter=max_iter, tol=tol, beta=beta, seed=seed
    )
