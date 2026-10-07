# SymNMF clustering

A small Python implementation of the symmetric non-negative matrix
factorization (SymNMF) clustering project, with a k-means baseline.

## Setup

Use Python 3.10 or later:

```text
python -m pip install -r requirements.txt
```

Input files are headerless CSV matrices: one observation per row and one
numeric feature per column.

## Command line

```text
python main.py K GOAL input.csv [--seed N] [--max-iter N] [--tol VALUE]
```

Available goals:

- `sym`: Gaussian similarity matrix, with `a[i,j] = exp(-||x[i]-x[j]||^2/2)`
  and a zero diagonal.
- `ddg`: diagonal degree matrix of the similarities.
- `norm`: normalized similarities `D^-1/2 A D^-1/2`.
- `symnmf`: non-negative factor matrix `H` for the normalized similarities.
- `symnmf-labels`: cluster index for each row, assigned by the largest value
  in its row of `H`.
- `kmeans`: cluster index for each row using Lloyd's algorithm.

`K` is ignored by `sym`, `ddg`, and `norm`. Matrix output is comma-separated
and rounded to four decimal places. Cluster indices are zero-based. Randomized
operations use seed `0` by default and can be made reproducible with `--seed`.

The Python API is available directly from `symnmf.py` and `kmeans.py`.

## Tests

```text
python -m unittest discover -s tests
```
