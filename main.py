"""Command-line entry point for SymNMF and k-means clustering."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from kmeans import kmeans
from symnmf import (
    assign_clusters,
    degree_matrix,
    normalized_similarity_matrix,
    similarity_matrix,
    symnmf_factorization,
)


def load_csv(path: str | Path) -> np.ndarray:
    """Load a headerless, comma-separated matrix of finite numbers."""
    try:
        data = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    except (OSError, ValueError) as error:
        raise ValueError(
            f"cannot read numeric CSV data from '{path}': {error}"
        ) from error
    if data.shape[0] < 1 or data.shape[1] < 1:
        raise ValueError("input data must not be empty")
    if not np.isfinite(data).all():
        raise ValueError("input data must contain only finite values")
    return data


def format_matrix(matrix: ArrayLike) -> str:
    """Format matrix rows as comma-separated values rounded to four decimals."""
    values = np.asarray(matrix, dtype=np.float64)
    return "\n".join(",".join(f"{value:.4f}" for value in row) for row in values)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run SymNMF or k-means on a headerless numeric CSV file."
    )
    parser.add_argument(
        "k", type=int, help="number of clusters (ignored for sym/ddg/norm)"
    )
    parser.add_argument(
        "goal",
        choices=("sym", "ddg", "norm", "symnmf", "kmeans", "symnmf-labels"),
        help="matrix operation, factorization, or clustering algorithm",
    )
    parser.add_argument("input", help="path to the input CSV file")
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    parser.add_argument(
        "--max-iter", type=int, default=300, help="maximum update iterations"
    )
    parser.add_argument("--tol", type=float, default=1e-4, help="convergence tolerance")
    return parser


def run(
    k: int,
    goal: str,
    data: np.ndarray,
    *,
    seed: int = 0,
    max_iter: int = 300,
    tol: float = 1e-4,
) -> str:
    if goal == "sym":
        return format_matrix(similarity_matrix(data))
    if goal == "ddg":
        return format_matrix(degree_matrix(similarity_matrix(data)))
    if goal == "norm":
        return format_matrix(normalized_similarity_matrix(similarity_matrix(data)))
    if goal in ("symnmf", "symnmf-labels"):
        factors = symnmf_factorization(
            normalized_similarity_matrix(similarity_matrix(data)),
            k,
            max_iter=max_iter,
            tol=tol,
            seed=seed,
        )
        if goal == "symnmf-labels":
            return "\n".join(str(label) for label in assign_clusters(factors))
        return format_matrix(factors)
    if goal == "kmeans":
        labels, _ = kmeans(data, k, max_iter=max_iter, tol=tol, seed=seed)
        return "\n".join(str(label) for label in labels)
    raise ValueError(f"unsupported goal: {goal}")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        data = load_csv(args.input)
        output = run(
            args.k,
            args.goal,
            data,
            seed=args.seed,
            max_iter=args.max_iter,
            tol=args.tol,
        )
    except (OSError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
