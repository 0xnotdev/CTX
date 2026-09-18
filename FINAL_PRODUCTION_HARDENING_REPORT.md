# Final production hardening evidence

This report covers only FINAL-01 strict cross-document completeness and FINAL-02 fail-closed
required semantic retrieval, based on canonical pre-use commit
`931a1e0b76cf9fe34207f45772b3a820c4d6fb45`.

## Delivered contract

A coding agent may begin implementation only when:

1. the pack was requested with `strict_agent=true` (CLI `--strict-agent`);
2. `require_semantic=true` is effective and retrieval metadata reports
   `HYBRID_SEMANTIC` with `semantic` active;
3. `completeness_status == COMPLETE`; and
4. required omissions, ambiguities, and possible conflicts are all empty.

Otherwise the caller must retrieve more, increase/explicitly expand the budget, or inspect raw
Markdown. Exact Markdown ranges and hashes remain authoritative.

Strict discovery deterministically groups signals from task/checkpoint title, goal, requirements,
dependencies, acceptance, files/modules, interfaces, security, failures, and verification. It
searches all configured normative documents globally plus once per document, classifies material
architecture, decisions, current progress/state, security, testing, dependency, and other
normative evidence, and marks every classified non-optional range required. Required filter or
budget drops are exact actionable omissions. Optional background can be omitted harmlessly.

Required semantics now validates the initialized local provider, index identity/dimensions and
algorithm metadata, complete source/hash-compatible finite vectors for the current generation,
and query execution. Model/checksum/revision/provider, vector, or query failures serialize as
`SEMANTIC_RETRIEVAL_UNAVAILABLE`; strict operation never falls back. Explicit non-strict operation
remains labeled `LEXICAL_ONLY` with active channels.

## Regression and validation evidence

Passed commands:

- `uv run pytest -q tests/test_final_production_hardening.py`
- `uv run pytest -q tests/test_pre_use_semantic_windows.py tests/test_pre_use_span_diversity.py tests/test_pre_use_completeness.py tests/test_pre_use_checkpoint_namespacing.py tests/test_pre_use_heading_normalization.py tests/test_pre_use_validation_corpus.py`
- `uv run pytest -q` (96 passed, 3 production-model tests skipped)
- `uv run ruff format --check .`
- `uv run ruff check .`
- `uv run mypy src`
- `uv build --out-dir dist/final-hardening-build/package`
- offline wheel install and CLI workflow via
  `uv pip install --offline --python .smoke-final/bin/python dist/final-hardening-build/package/ctx_context-1.0.0-py3-none-any.whl`
- actual stdio MCP client coverage is in `tests/test_final_production_hardening.py` and
  `tests/test_mcp.py`, including tool schema, strict typed failure, and lexical-only metadata
- `uv run python benchmarks/run_retrieval_evaluation.py --output dist/final-hardening-build/retrieval-mechanics.json`
- `uv run python benchmarks/run_benchmarks.py --output dist/final-hardening-build/performance-mechanics.json`
- `uv run python benchmarks/cosine.py`

Final deterministic mechanics observations: required pack recall `6/6`, checkpoint accuracy
`1.0`, hybrid recall@3 `0.75`, 758-chunk index `5430.65 ms`, warm hybrid median `349.93 ms`, pack
median `951.89 ms`, and unchanged sync `4.21 ms`. Hash embeddings prove mechanics only and make no
production semantic-quality claim.

The realistic regression corpus contains SPEC, ARCHITECTURE, DECISIONS, PROGRESS, SECURITY,
TESTING, OPERATIONS, optional BACKGROUND, and irrelevant normative material. It proves low lexical
overlap discovery, exact decision/progress omissions, architecture/security/testing/other
normative coverage, optional omission, deterministic ordering, ambiguity/conflict visibility,
semantic provider/query/vector failures, CLI/config/MCP parity, socket-denied offline operation,
and exact provenance. The complete prior PRE-USE suite and full suite retain stale-source refusal
and atomic-generation coverage.

## Unavailable real-model gates

No verified cached FastEmbed/BGE model exists in this worktree/host cache; normal runtime did not
attempt a download. These gates were attempted and **did not pass**, so this report makes no new
real-model or production-BGE benchmark claim:

- `CTX_RUN_PRODUCTION_EMBEDDINGS=1 CTX_MODEL_DIR="$PWD/dist/final-hardening-build/missing-models" uv run pytest -q tests/test_production_embeddings.py`
  — failed all three gates because the exact BGE manifest/artifacts were absent.
- `uv run python benchmarks/run_retrieval_evaluation.py --production --model-dir "$PWD/dist/final-hardening-build/missing-models" --output dist/final-hardening-build/retrieval-production.json`
  — failed before evaluation with `invalid or missing model manifest`.

`tests/test_production_embeddings.py` now contains the cached-BGE strict multi-document regression
and remains the release gate when the verified cache is available.

## Remaining limitations

Discovery is deterministic and bounded, not an oracle: it depends on correctly configured
`NORMATIVE` authority, the installed embedding model's retrieval quality, role-bearing document
paths/headings, conservative semantic score gates, and top-six focused results per normative
document. Unusual terminology or more than six highly relevant sections behind one focused probe
may require targeted retrieval or raw-Markdown review. Conflict detection remains the documented
compact deterministic signal, not general natural-language theorem proving. These limits are why
the production contract requires semantic activation and directs fallback rather than treating any
navigation layer as authority.
