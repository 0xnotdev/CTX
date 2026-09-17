# CP-05 embedding selection and brute-force measurement

Selected production default: **`BAAI/bge-small-en-v1.5` through FastEmbed/ONNX CPU**.
This is an explicit, cached local model: `ctx` never calls an embedding API. Normal operation
uses `local_files_only`; a user must opt in to the initial model download.

## Evidence and licensing

FastEmbed's generated supported-model metadata reports this model as 384 dimensions, English,
512-token input, about 0.067 GB, and MIT licensed. The upstream BGE model card reports its
measured MTEB retrieval average (NDCG@10) as 51.68, versus 49.54 for the older `bge-small-en`;
that measured retrieval result, small footprint, permissive license, and direct FastEmbed
support drove selection. This is evidence for a reasonable V0 default, not a claim that MTEB
perfectly represents source-code specifications.

Authoritative references:

- FastEmbed supported model metadata: <https://qdrant.github.io/fastembed/examples/Supported_Models/>
- Model card, evaluation table, and MIT license metadata:
  <https://huggingface.co/BAAI/bge-small-en-v1.5>
- FastEmbed package is Apache-2.0; ONNX Runtime is MIT. Model terms are separate from package
  terms. Users redistributing cached weights must retain/review the model's own license.

The deterministic `HashEmbedding` exists only for tests and offline fixtures; it is not
presented as a semantic-quality substitute.

## NumPy brute-force baseline

Run `python benchmarks/cosine.py`. On the recorded development host (2025-09-17, NumPy 2.5.3),
10 warm searches across 20,000 normalized 384-float vectors took **0.093 s total / 9.31 ms per
query** and consumed about **29.3 MiB** of vector data. Host load and BLAS matter; rerun rather
than treating this as a guarantee. This supports simple in-process cosine for expected V0 scale
and gives no measured reason to add ANN infrastructure.
