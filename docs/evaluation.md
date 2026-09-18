# V1 evaluation, performance, and adversarial evidence

Recorded **2026-09-18**. Raw machine-readable results are committed under `benchmarks/*-2026-09-18.json`; commands generate temporary local corpora and do not use network. Measurements are host observations, not promises.

Environment: Linux 6.18.33.2 WSL2 x86_64, Python 3.12.3, 16 reported CPUs;
ctx 1.0.0, FastEmbed 0.8.0, MCP 2.2.0, NumPy 2.5.3, ONNX Runtime 1.30.0,
Pydantic 2.13.5. `platform.processor()` reported `x86_64`; no more specific CPU model is
claimed.

> **Pre-use compatibility note.** The production numbers below are retained V1 evidence and were
> not recharacterized after semantic-window, span-diversity, completeness, artifact, and heading
> changes. The current deterministic mechanics rerun uses a 15,000-token pack: all 6/6 required
> sections were present, serialized use was 10,668 tokens, and exact returned source was 3,302 of
> 89,461 characters (96.31% reduction). The former strict 7,000 budget now reports only 4/6 and
> `PARTIAL` rather than silently claiming completeness. Current actual-BGE gates were not run
> because FastEmbed and the verified local model were unavailable; see `PRE_USE_HARDENING_REPORT.md`.

## Tier separation

- **Deterministic mechanics:** `HashEmbedding(64)` proves storage, dimensionality, filtering,
  caching, fusion and incrementality. It is not semantic and supports no production-quality claim.
- **Production semantic:** verified cached `BAAI/bge-small-en-v1.5`, exact FastEmbed/Qdrant
  artifact revision `52398278842ec682c6f32300af41344b1c0b0bb2`, aggregate manifest SHA-256
  `3d57fa27448b19e85e3ad1abb0fa5d4938dc01487f26fe0c531b52e049ab90c2`, 384 dimensions,
  FastEmbed 0.8.0/ONNX. The production suite patches socket connection APIs to fail and proves
  indexing, semantic search, context packing and an actual SDK-client stdio MCP subprocess remain
  offline.

Production commands never download:

```console
CTX_RUN_PRODUCTION_EMBEDDINGS=1 CTX_MODEL_DIR=/verified/cache \
  uv run pytest -q tests/test_production_embeddings.py
uv run python benchmarks/run_retrieval_evaluation.py \
  --production --model-dir /verified/cache
uv run python benchmarks/run_benchmarks.py --production --model-dir /verified/cache
```

## Hard retrieval and token efficiency

The generated corpus contains normative `spec.md` and `architecture.md`, reference
`research.md`, historical `old-spec.md`, informal `notes.md`, conflicting duplicate CP-14 text, a
lexical decoy, RunManifest, security, acceptance/verify evidence, and a one-section passage beyond
3,000 tokens. Four fixed cases cover exact checkpoint, low-overlap tail, lexical decoy and model
paraphrase. This is a small regression corpus, not a general benchmark.

| Channel / metric | Recall@1 | Recall@3 | MRR |
|---|---:|---:|---:|
| FTS-only | 0.500 | 0.750 | 0.583 |
| actual BGE semantic-only | 0.500 | 0.750 | 0.625 |
| actual BGE hybrid | 0.500 | **1.000** | **0.750** |
| Hash semantic-only (mechanics only) | 0.250 | 0.250 | 0.250 |
| Hash hybrid (mechanics only) | 0.500 | 0.500 | 0.500 |

Hybrid materially improves the hard set over FTS at Recall@3 and MRR, though not Recall@1. The
separate production regression proves the relevant long-tail chunk is semantic rank 1 under the
low-overlap query `What must happen before accepting rerun evidence?`. The lexical-decoy case is
why authority is only a staged tie-break rather than a global relevance boost.

In the retained V1 production run, checkpoint resolution accuracy was **1.000**. The 7,000-budget pack returned all **6/6** required
normative sections (checkpoint, dependency, RunManifest, security, acceptance, verify), 12 exact
selected sections, and a complete serialized estimate of **6,064** tokens. Exact returned source
was 1,074 characters from an 89,461-character corpus: **98.80% source-character reduction**. The
token estimate includes the compact MCP/JSON envelope and provenance; character reduction is a
separate descriptive measure and is not mislabeled as token compression.

Raw reports:

- `benchmarks/retrieval-production-2026-09-18.json`
- `benchmarks/retrieval-mechanics-2026-09-18.json`

## Performance

The performance corpus has a generated 3,000-line specification, 20,000-line research file and
one huge 20,000-sentence section (4,555 chunks total). Medians use the repeat counts in
`benchmarks/run_benchmarks.py`.

| Operation | Hash mechanics | Actual local BGE |
|---|---:|---:|
| parse 3k lines | 98.1 ms | 113.7 ms |
| parse 20k lines | 669.2 ms | 806.5 ms |
| parse huge one-section source | 217.0 ms | 237.0 ms |
| cold model construction | n/a | 809.4 ms |
| complete parse/chunk/embed/index | 2.02 s | 87.78 s |
| exact lookup warm | 6.42 ms | 8.07 ms |
| FTS warm | 7.30 ms | 9.09 ms |
| semantic warm | n/a | 499.4 ms |
| hybrid warm | 409.8 ms | 532.3 ms |
| 7k pack warm | 438.1 ms | 688.0 ms |
| unchanged sync | 3.87 ms | 7.50 ms |
| SQLite DB | 14.77 MB | 21.45 MB |
| process peak RSS reported by `ru_maxrss` | 1.30 GiB | 2.51 GiB |

Peak RSS is process-wide high-water memory, not isolated vector-matrix memory; Python/ONNX and
parsing allocations are included. Brute-force vectors are deliberately bounded in-process and
cached in at most four immutable matrix entries. These measured production latencies are
acceptable for personal local use but are not server-scale claims. A sparse 100 MiB source was
rejected by the default 25 MiB per-file bound in 0.29 ms; full 100 MiB indexing is not claimed.

Raw reports:

- `benchmarks/results-production-2026-09-18.json`
- `benchmarks/results-mechanics-2026-09-18.json`

## End-to-end and lifecycle evidence

`tests/test_final_success.py`, `tests/test_v1_audit.py`, `tests/test_mcp.py`, and the production
suite jointly prove:

- normative CP-14 remains primary; dependencies, RunManifest, applicable security,
  acceptance/verify, hashes/ranges, conflict authority and omissions are explicit;
- one section produces many bounded exact chunks and a relevant tail is returned, not the prefix;
- 20 concurrent search+pack readers, two sync calls and sync overlap have no thread/lock or mixed
  generation failure;
- the official MCP SDK 2 client lists/calls status, outline, search, exact, symbol, section, lines,
  refs, deps, checkpoint, checkpoint context and pack, then cancels an in-flight request and
  cleanly reaps the subprocess;
- MCP stdout remains protocol-only (the SDK client would fail framing otherwise), no network
  listener is configured, hostile Markdown remains inert, and socket-denied production MCP works;
- V1→V2 migration, stale sources, missing vectors, rename/delete/recreate, old-path reuse,
  duplicate checkpoints/references, equal ties, filters-before-cutoff and configured limits are
  permanent tests.

## Adversarial review and limitations

Covered inputs include huge headings/paragraphs/fences/tables, emoji and Unicode, random-like
long content through byte-safe splitting, duplicate headings/checkpoints/symbols, generated
3k/20k corpora, 100-document scale in the inherited suite, rename/delete/recreate, symlink
escape, invalid UTF-8, malformed Markdown/config/DB-version, missing/wrong model manifest,
concurrent calls, MCP cancellation and source size growth after open. Windows drive/root/case
normalization is deterministic and Windows CI is configured; junction behavior remains
best-effort under the host filesystem and is not claimed race-proof.

Known limitations:

1. Exact GitHub anchor compatibility is not claimed; ctx uses documented internal slugs and
   exposes ambiguity.
2. `synchronous=NORMAL` is a deliberate WAL performance/durability choice; sudden power loss has
   weaker durability than FULL, while readers still cannot see a partial transaction.
3. Global in-process read/write serialization favors correctness over maximum concurrent sync
   throughput. Multiple OS processes rely on SQLite busy handling and do not share the Python
   operation lock.
4. Huge fences/tables may be emitted as bounded exact lexical pieces rather than semantic vectors;
   no corrupted fragment is called authoritative.
5. Production evaluation is intentionally small, English-centric and model-specific. Re-run on a
   real project before relying on quality or latency figures.
6. A workspace writable by an untrusted local attacker is outside the source-read race boundary.

## Release verdict

**READY FOR PERSONAL DAILY USE.** Every P0 has a direct implementation and regression, locked
quality gates pass, actual-model and official-MCP evidence passes offline, and limitations above
are explicit. This is not a multi-user service or a substitute for reviewing exact source.
