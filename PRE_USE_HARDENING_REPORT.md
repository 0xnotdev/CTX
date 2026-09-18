# CTX pre-use hardening report

## Summary

Implemented only PRE-USE-01 through PRE-USE-05 on top of exact V1 commit
`4993849704aecd15232cf02b538fb9b05bb79807`. The work preserves exact Markdown authority,
source/range SHA-256 provenance, stale-index refusal, one-transaction generations, offline normal
runtime, explicit ambiguity, and navigation-only derived data.

Checkpoint commits:

- PRE-USE-01: `71d5454`
- PRE-USE-02: `f71a8b2`
- PRE-USE-03: `6f855f2`
- PRE-USE-04: `981f6a4`
- PRE-USE-05: `5291155`
- integrated large-corpus release gate and strict required-section correction: `5b84122`

## PRE-USE-01 — semantic retrieval windows

- Replaced 384-byte, non-overlapping chunks with derived windows targeting 320 embedding-model
  tokens, 48-token overlap (15%), and a 448-token hard input ceiling below BGE's 512-token limit.
- Verified FastEmbed uses its local tokenizer's `token_count`. Without that runtime, window sizing
  uses the documented deterministic `ceil(UTF-8 bytes / 4) + 2` approximation. `HashEmbedding`
  exposes its exact fixture token count.
- Authoritative section boundaries and text are unchanged. Every window keeps stable identity,
  document/section ownership, exact offsets/lines, source hash, separate navigation-text hash,
  ordinal, chunker version, and committed generation through hydration.
- Heading-path prefixes occur only in embedding text. Returned evidence is exact source text sliced
  at stored offsets. Unicode and oversized fences/tables remain exact contiguous ranges.
- Parser/chunker/embedding-text fingerprints force stale derived data and vectors to rebuild.

Regression coverage includes an old-boundary-crossing concept, a no-lexical-hit synonym query,
overlap, exact hydration, unchanged section ordering, Unicode, and fenced code mapping.

## PRE-USE-02 — multiple distant evidence spans per section

- Removed section-ID-wide collapse from structural, FTS, semantic, and hybrid fusion.
- Fusion now merges overlapping ranges, not every range owned by a section.
- Deterministic diversity selects distinct sections/documents first, then permits at most three
  disjoint ranges per section. Overlapping neighboring windows, identical range hashes, and at
  least 92% cross-section token-set duplicates collapse.
- The three-span cap is intentionally small enough to stop one huge section flooding a top-k but
  large enough for the observed two/three independent-requirement use case. Deferred candidates
  fill remaining result slots.
- Context-pack candidate identity is range-aware, so two distant exact ranges from one section can
  both survive with distinct hashes.

Regression coverage includes two distant required ranges in one section, adjacent overlap
collapse, copied-range suppression, another section/document surviving, the three-span cap, and
exact section/range hashes.

## PRE-USE-03 — explicit context-pack completeness

- Context pack schema v3 reports top-level `COMPLETE`, `PARTIAL`, `AMBIGUOUS`, `CONFLICTING`, or
  `NOT_APPLICABLE` plus coverage for primary evidence, dependencies, architecture, security,
  acceptance, verification, and checkpoint descendants.
- Dropped required evidence carries category, reason, document ID/path, section, checkpoint and
  dependency where applicable, source lines, and range hash. Ambiguities retain candidate source
  refs; conflicts remain compact source refs without duplicated source text.
- Optional-neighbor omission does not make a pack partial. Required primary/checkpoint/dependency/
  architecture/security/acceptance/verification/descendant evidence does.
- Required checkpoint sections and graph targets must be complete source sections; a navigation
  window cannot make their category complete. Generic query packs can retain multiple exact
  query-covering ranges.
- Strict mode never expands the budget. It returns an honest partial pack or the existing
  `CONTEXT_BUDGET_TOO_SMALL` / `PRIMARY_REQUIREMENT_TOO_LARGE` machine error.
- API, CLI, checkpoint context, and MCP expose explicit required-budget expansion. When enabled,
  the effective budget may grow only to the configured maximum; requested/effective budget,
  `budget_expanded`, and serialized use are disclosed. The CLI/MCP default is now 15,000 tokens.
- Complete serialized accounting still includes the MCP/JSON-RPC envelope, all completeness
  metadata, safety margin, and fixed-point count fields.
- Checkpoint context resolves a scoped root while retaining cross-document dependency evidence.

The requested cases cover: all fits; optional neighbor omitted; dependency cannot fit; acceptance
partially omitted; security cannot fit; primary cannot fit; ambiguous dependency; conflicting
evidence; distant same-section spans; strict completeness/accounting. An additional regression
covers cross-document checkpoint dependencies. The current mechanics evaluation confirms why the
policy matters: the old 7k budget is honestly partial at 4/6 required sections, while 15k returns
6/6 in 10,668 serialized estimated tokens.

## PRE-USE-04 — document-aware checkpoint artifacts

- Generated artifacts are loaded only from
  `.ctx/checkpoints/<percent-encoded-document-id>/CP-N.json`; the namespace is reversible,
  filesystem-safe, persistent across same-content rename, and keyed exactly like durable
  checkpoints: `(document_id, checkpoint_id)`.
- Artifact models include document ID and checkpoint ID. Conflicting embedded provenance is
  rejected. Global CP-label fallback was removed.
- Deleting a document cleans only its known namespace; other documents with the same checkpoint
  label remain intact. Rebuild and no-op sync preserve namespaced artifacts.
- Legacy `.ctx/checkpoints/CP-N.json` migrates atomically only when embedded document ID (preferred)
  or document path uniquely identifies one active document. Unprovenanced/ambiguous legacy files
  remain ignored for explicit rebuild/remediation and are never reinterpreted.
- Artifact hashes remain in behavior fingerprints, so a valid artifact change advances exactly
  one atomic generation.

Regression coverage uses the same checkpoint label in two and three documents, deletion, rename,
rebuild, legacy migration/invalidation, ambiguous unscoped lookup, and document path/ID-scoped
lookup.

## PRE-USE-05 — canonical heading normalization

- Added one shared CTX internal canonical heading function and removed parser/graph slug
  implementations.
- Scheme: Unicode NFKC, Unicode case fold, retain Unicode alphanumeric characters and combining
  marks, map every run of whitespace/punctuation/symbols/controls (including periods,
  underscores, hyphens, parentheses, slashes, and emoji) to one ASCII hyphen, trim, and use
  `section` for an empty key.
- Parser section identity and canonical duplicate occurrence, graph/local-anchor resolution,
  checkpoint heading/field recognition, FTS heading indexing, structural matching, heading-prefix
  filtering, and ranking boosts use the shared scheme. Exact source headings remain unchanged.
- Duplicate canonical headings receive deterministic occurrence IDs and ambiguous reference
  candidates in source order.
- This is explicitly not GitHub/CommonMark/renderer anchor compatibility. Local `#anchor` links
  use CTX canonical identity; external `path#anchor` links remain explicitly unresolved.
- Heading normalization is behavior-fingerprinted, so existing indexes rebuild rather than mixing
  identities.

Regression coverage specifies NFKC, case folding, repeated spaces, periods, underscores, hyphens,
punctuation, parentheses, slashes, emoji, Greek/Cyrillic/CJK text, empty keys, deterministic
canonical duplicates, local ambiguity, external-anchor uncertainty, indexed search, and heading
prefix filters.

## Tests and validation

New focused suites:

- `tests/test_pre_use_semantic_windows.py`
- `tests/test_pre_use_span_diversity.py`
- `tests/test_pre_use_completeness.py`
- `tests/test_pre_use_checkpoint_namespacing.py`
- `tests/test_pre_use_heading_normalization.py`
- `tests/test_pre_use_validation_corpus.py`

The integrated validation corpus creates three substantial Markdown documents with a huge
checkpoint descendant, two distant requirements, repeated CP-17 labels, a cross-document CP-2 and
RunManifest dependency, Unicode/punctuation headings, security/acceptance/verification,
Python/Unicode code, and a table. It verifies exact hydration/hashes, distant spans, honest partial
and expanded complete packs, namespaced artifacts, canonical resolution, stale-source refusal,
no-op and changed atomic generations, and socket-denied offline runtime.

Executed gates:

```console
uv sync --frozen
uv run --frozen ruff format --check .
uv run --frozen ruff check .
uv run --frozen mypy src
uv run --frozen pytest -q
uv build
uv run --frozen python benchmarks/cosine.py
uv run --frozen python benchmarks/run_benchmarks.py
uv run --frozen python benchmarks/run_retrieval_evaluation.py
```

Final full collection: 90 tests; 88 passed and 2 production-embedding tests skipped by their
explicit environment gate. Ruff formatting/lint and strict mypy passed. Wheel and sdist built. A
clean local venv installed the wheel and completed installed `ctx` init/add/index/pack/status;
status was `CLEAN`, pack status was `COMPLETE`, and serialized use stayed within budget. Current
mechanics benchmark completed; cosine measured 7.99 ms/query on 20k x 384 vectors in this run.
The official MCP SDK subprocess/concurrency/cancellation tests are part of the passing full suite.

Not run:

- `CTX_RUN_PRODUCTION_EMBEDDINGS=1 ... tests/test_production_embeddings.py`
- production FastEmbed retrieval/performance benchmark modes

Reason: FastEmbed/ONNX and a verified local BGE artifact were not installed (`CTX_MODEL_DIR` was
unset). No model was downloaded because model acquisition is an explicit network operation and
normal validation must remain offline. Historical V1 production measurements remain documented
but are not represented as current-window results.

## Compatibility and migration impact

- No database replacement or broad migration was added; SQLite schema remains V2.
- Parser, chunker, embedding text, graph, retrieval, checkpoint artifact, and heading behavior
  versions changed. A normal sync detects this and atomically rebuilds derived sections/windows,
  FTS, graph/checkpoints, and model vectors while preserving opaque document IDs.
- Canonical heading changes can change section IDs. Clients must re-run sync and fetch current IDs;
  stale generations are refused.
- Search chunk IDs/ranges and embeddings change because windows are larger and overlap.
- Context pack serialization is schema v3 and adds required fields. Consumers must inspect status
  and category coverage. Defaults are 15k; strict requested budgets are still honored.
- Generated artifact producers must write document-scoped paths. Provenanced legacy files migrate;
  ambiguous legacy files require explicit regeneration.

## Remaining limitations

1. `COMPLETE` means complete for CTX's deterministic evidence plan and explicit structure/query,
   not proof that an unstated requirement does not exist elsewhere. Raw authoritative Markdown
   remains the fallback for uncertainty.
2. The no-model window estimator is deterministic, not a model-tokenizer guarantee. Verified
   FastEmbed uses exact local token counting and the hard 448-token ceiling.
3. At most three disjoint ranges per section survive one retrieval result. Additional relevant
   ranges require another targeted query or raw section hydration.
4. External renderer anchor syntax is not guessed; external anchors remain unresolved.
5. Current production-BGE quality/performance was not rerun in this environment.
6. Existing WAL `synchronous=NORMAL`, local-writer trust boundary, brute-force vectors, and
   process-local operation lock limitations are unchanged.

## Dogfood-readiness verdict

**READY FOR CONTROLLED CTX-FIRST DOGFOODING, with authoritative Markdown fallback required.**

The five requested release gates pass in deterministic/offline, CLI, MCP, installed-wheel, stale
source, atomic-generation, and realistic large-corpus coverage. Agents can now distinguish a
small complete pack from a deceptively incomplete one, and same-label checkpoints/windows cannot
cross authority boundaries. Deployment must reject non-`COMPLETE` packs for implementation,
inspect ambiguities/conflicts/omissions, and hydrate exact Markdown when wording or coverage is
uncertain. This is not a claim of CTX-only corpus completeness. Before treating the semantic
channel as production-qualified on a target corpus, rerun the skipped verified local BGE gates.
