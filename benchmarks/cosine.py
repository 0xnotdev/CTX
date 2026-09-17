"""Repeatable in-process cosine baseline; not a performance promise."""

from __future__ import annotations

from time import perf_counter

import numpy as np

ROWS = 20_000
DIMENSIONS = 384
QUERIES = 10

rng = np.random.default_rng(42)
matrix = rng.standard_normal((ROWS, DIMENSIONS), dtype=np.float32)
matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
queries = rng.standard_normal((QUERIES, DIMENSIONS), dtype=np.float32)
queries /= np.linalg.norm(queries, axis=1, keepdims=True)

# warm-up
_ = matrix @ queries[0]
start = perf_counter()
for query in queries:
    _ = np.argpartition(matrix @ query, -10)[-10:]
elapsed = perf_counter() - start
print(f"numpy={np.__version__}")
print(f"vectors_mib={matrix.nbytes / 1024 / 1024:.1f}")
print(f"queries={QUERIES} total_s={elapsed:.4f} ms_per_query={elapsed / QUERIES * 1000:.2f}")
