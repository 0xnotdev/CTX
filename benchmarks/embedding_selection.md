# Production embedding selection

Selected default: **BAAI/bge-small-en-v1.5 through FastEmbed/ONNX CPU**. Normal ctx operations
open only an explicitly installed, checksum-verified local cache with `local_files_only=True`.
Only `ctx model download` can enable network access.

## Identity tested on 2026-09-18

- model: `BAAI/bge-small-en-v1.5`
- FastEmbed/Qdrant ONNX artifact revision:
  `52398278842ec682c6f32300af41344b1c0b0bb2`
- aggregate manifest SHA-256:
  `3d57fa27448b19e85e3ad1abb0fa5d4938dc01487f26fe0c531b52e049ab90c2`
- dimensions: 384
- FastEmbed 0.8.0; ONNX Runtime 1.30.0
- chunker: `ctx-ast-byte-safe:2`; embedding text: `heading-path-prefix:1`

FastEmbed's supported-model metadata describes the model as English, 384-dimensional, 512-token
input, roughly 0.067 GB and MIT licensed. The upstream BGE card reports MTEB retrieval NDCG@10
51.68 for v1.5 versus 49.54 for the older small English model. These third-party measurements and
the model's footprint/license motivated selection; they do not substitute for ctx's own corpus.

Sources:

- <https://qdrant.github.io/fastembed/examples/Supported_Models/>
- <https://huggingface.co/BAAI/bge-small-en-v1.5>
- <https://github.com/qdrant/fastembed>

Package and model licenses are independent. FastEmbed is Apache-2.0, ONNX Runtime is MIT, and the
model card reports MIT; review the exact artifact's terms before redistribution.

## Local evidence

The production tier in `docs/evaluation.md` uses the exact identity above and physically denies
socket connections during normal semantic and stdio-MCP operations. On the fixed hard corpus,
semantic-only Recall@1/Recall@3/MRR is 0.500/0.750/0.625 and hybrid is
0.500/1.000/0.750. A >3,000-token section's low-overlap relevant tail ranks first in the dedicated
production regression. The full production 20k-line index took 87.78 seconds on the dated host;
warm semantic median was 499.4 ms. These are observations, not guarantees.

`HashEmbedding` is deterministic test mechanics only. Its retrieval numbers and timings must not
be presented as production semantic quality.
