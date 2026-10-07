"""A deterministic, NumPy-based Lloyd k-means baseline."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatMatrix = NDArray[np.float64]
IntVector = NDArray[np.int64]


def _as_points(data: ArrayLike) -> FloatMatrix:
    points = np.asarray(data, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 1 or points.shape[1] < 1:
        raise ValueError("data must be a non-empty two-dimensional matrix")
    if not np.isfinite(points).all():
        raise ValueError("data must contain only finite values")
    return points


def _squared_distances(points: FloatMatrix, centers: FloatMatrix) -> FloatMatrix:
    distances = (
        np.einsum("ij,ij->i", points, points)[:, None]
        + np.einsum("ij,ij->i", centers, centers)[None, :]
        - 2.0 * points @ centers.T
    )
    return np.maximum(distances, 0.0)


def _initialize_centers(
    points: FloatMatrix, k: int, rng: np.random.Generator
) -> FloatMatrix:
    n = points.shape[0]
    chosen = [int(rng.integers(n))]
    closest = _squared_distances(points, points[chosen]).ravel()

    while len(chosen) < k:
        weights = closest.copy()
        weights[chosen] = 0.0
        total = float(weights.sum())
        if total > 0.0 and np.isfinite(total):
            index = int(rng.choice(n, p=weights / total))
        else:
            index = next(i for i in range(n) if i not in chosen)
        chosen.append(index)
        closest = np.minimum(
            closest, _squared_distances(points, points[[index]]).ravel()
        )

    return points[chosen].copy()


def kmeans(
    data: ArrayLike,
    k: int,
    *,
    max_iter: int = 300,
    tol: float = 1e-4,
    seed: int | None = 0,
) -> tuple[IntVector, FloatMatrix]:
    """Return cluster labels and centroids using Lloyd's algorithm."""
    points = _as_points(data)
    n = points.shape[0]
    if (
        not isinstance(k, (int, np.integer))
        or isinstance(k, (bool, np.bool_))
        or not 1 <= k <= n
    ):
        raise ValueError("k must be an integer between 1 and the number of rows")
    if not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be a positive integer")
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError("tol must be finite and non-negative")

    rng = np.random.default_rng(seed)
    centers = _initialize_centers(points, k, rng)
    labels = np.full(n, -1, dtype=np.int64)

    for _ in range(max_iter):
        distances = _squared_distances(points, centers)
        new_labels = np.argmin(distances, axis=1).astype(np.int64)
        errors = distances[np.arange(n), new_labels]
        new_centers = np.empty_like(centers)

        for cluster in range(k):
            members = points[new_labels == cluster]
            if members.size:
                new_centers[cluster] = members.mean(axis=0)
            else:
                new_centers[cluster] = points[int(np.argmax(errors))]

        shift = float(np.linalg.norm(new_centers - centers))
        unchanged = np.array_equal(new_labels, labels)
        labels, centers = new_labels, new_centers
        if unchanged and shift <= tol:
            break

    return labels, centers
