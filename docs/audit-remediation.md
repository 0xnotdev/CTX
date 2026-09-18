# V1 audit remediation record

Audit date: **2026-09-18**. Baseline: complete V0 implementation
`2e80f1dded1544bc8b8fa2c1849564bda2878218`. Original Markdown remains authoritative.
This record distinguishes deterministic CI evidence from production-model evidence.

## Failure analysis method

The supplied failures were reproduced before implementation. The initiating triggers were a
64-token pack carrying a 1,000-character task, a long single section, and rename plus old-path
reuse. They produced respectively `estimated_tokens=274 > token_budget=64`, one unbounded chunk
and a 7,019-character search response, and `UNIQUE constraint failed: documents.id`. Installing
the official MCP SDK 2.2.0 additionally reproduced V2 worker-thread access to the V0 shared
SQLite connection (`ProgrammingError: SQLite objects created in a thread can only be used in that
same thread`).

The common masking conditions were normal-sized tasks/sections, no path reuse, sequential CLI
calls, SDK v1 running synchronous handlers on the creator thread, and tests that counted section
text rather than the serialized response. Earliest divergences were respectively pack-envelope
construction, parser emission of exactly one chunk, derivation of document identity from path,
and creation of one process-global connection. Disconfirming evidence ruled out FTS itself,
Markdown source corruption, UUID collision, and WAL configuration: the wrong values existed
before those layers. Permanent reproductions are in `tests/test_v1_audit.py` and the migrated MCP
suite.

## Findings and repairs

`pytest` references below mean `uv run pytest -q`; all implementation models are frozen and
`extra="forbid"` unless a library boundary requires otherwise.

| ID | Sev | Symptom and root cause | Regression / implementation / files | Verification | Status |
|---|---:|---|---|---|---|
| P0-01 | P0 | Long task could exceed its advertised pack budget because base/task overhead was never rejected. | Boundary and long-task tests; machine-readable `ContextBudgetTooSmall` with requested/minimum/task/metadata values. `context_pack.py`, `test_v1_audit.py`. | `pytest tests/test_v1_audit.py -k budget` | FIXED |
| P0-02 | P0 | `bytes/4` was mislabeled conservative and only source fragments were counted. | `TokenCounter` protocol; strict-byte, explicit generic, model-specific contract; complete compact MCP envelope, keys, escaping, omissions/conflicts and safety margin are counted. | same | FIXED |
| P0-03 | P0 | Conflicts duplicated complete sources. | `PossibleConflict.sources` is two compact `SourceRef`s with no text; full pack is rebudgeted after conflict construction. | `pytest tests/test_context_pack.py -k conflict` | FIXED |
| P0-04 | P0 | One section always made one unbounded chunk. | AST-aware, byte-safe multi-chunk parser with deterministic paragraph/sentence/line splits and exact offsets. `parser.py`, `models.py`. | `pytest tests/test_parser.py` | FIXED |
| P0-05 | P0 | Huge blocks could reach model-side truncation. | Embedding text and source are separate; safe 448-byte upper bound stays below a 512-token byte fallback; fences/tables remain exact chunks or bounded exact line/lexical chunks. | `pytest tests/test_v1_audit.py -k multichunk` | FIXED |
| P0-06 | P0 | Search and packs returned section prefixes/full sections rather than match-centered evidence. | Default `SourceExcerpt` comes from matched chunk; full sections are opt-in; packs retain chunk match ranges and range hashes. `store.py`, `service.py`. | same | FIXED |
| P0-07 | P0 | Path-derived IDs collide after rename and old-path reuse. | Opaque UUID-style IDs; unique-hash rename preservation; ambiguous same-hash changes never guessed; recreate gets a new ID. | `pytest tests/test_v1_audit.py -k opaque` | FIXED |
| P0-08 | P0 | Checkpoints were globally keyed by checkpoint text and silently selected one. | Durable key `(document_id, checkpoint_id)`; document filter then authority then priority; equal ties raise `AMBIGUOUS_CHECKPOINT`. | `pytest tests/test_v1_audit.py -k checkpoint` | FIXED |
| P0-09 | P0 | Cross-reference duplicates fanned out as certainty. | Same-document/authority/priority resolution; one RESOLVED target or explicit UNRESOLVED/AMBIGUOUS candidates, reason, evidence and origin. | `pytest tests/test_v1_audit.py -k reference` | FIXED |
| P0-10 | P0 | CLI and MCP assembled retrieval engines independently. | Both call `create_context_engine`; verified local model enables hybrid, otherwise active structural+lexical fallback is reported. | `pytest tests/test_cli.py tests/test_mcp.py` | FIXED |
| P0-11 | P0 | MCP v1 was deliberately pinned despite stable v2. | Official docs/PyPI researched; pin `mcp>=2.2,<3`; `MCPServer`, snake_case client fields and actual stdio SDK client tests. `docs/dependency-research.md`. | `uv run pytest -q tests/test_mcp.py tests/test_security.py` | FIXED |
| P0-12 | P0 | One creator-thread SQLite connection failed under MCP V2 workers. | Checked thread-local connections, WAL, FK, 5s busy timeout, NORMAL synchronous and path-scoped explicit write serialization. | `pytest tests/test_v1_audit.py -k concurrent` | FIXED |
| P0-13 | P0 | Documents, FTS, graph, checkpoints and vectors committed as separate generations. | Parse/graph/vector preparation precedes one `BEGIN IMMEDIATE`; one transaction replaces all derived state and advances generation once. | same | FIXED |
| P0-14 | P0 | Behavioral metadata changes did not advance generation. | Fingerprint includes authority, priority, paths, all algorithms, artifact fingerprint, embedding identity and dimensions. | `pytest tests/test_v1_audit.py -k algorithm` | FIXED |
| P0-15 | P0 | Status only had stale/missing booleans. | CLEAN/SOURCE/METADATA/PARSER/EMBEDDINGS/GRAPH/SCHEMA/MISSING categories with reasons and vector checks. | `pytest tests/test_v1_audit.py -k algorithm` | FIXED |
| P0-16 | P0 | Filters were applied after candidate cutoffs or only in packs. | Shared `FilterSet`; document, exclusion, authority, heading and scope predicates enter structural/FTS/vector/symbol SQL before ranking. | `pytest tests/test_v1_audit.py -k filters` | FIXED |
| P1-01/P1-18 | P1 | Full sections and excerpts shared an ambiguous type/hash. | `SourceSection`, `SourceExcerpt`, `SearchHit`, `ContextPackItem`, `SourceRange`; every partial result has section and range hashes. | `pytest tests/test_v1_audit.py -k multichunk` | FIXED |
| P1-02/P1-03 | P1 | Hash embeddings were presented too close to production quality evidence. | Deterministic mechanics and opt-in cached BGE tiers are separated; lexical/semantic/hybrid metrics are separate and the production fixture has a >3k-token relevant tail. | `pytest tests/test_evaluation.py`; production command in `docs/evaluation.md` | FIXED |
| P1-04/P1-05/P1-06/P1-28 | P1 | Workspace-local implicit model cache and weak identity allowed downloads/drift. | platformdirs global cache plus overrides; explicit status/download/install/verify; manifest checksums/revision/runtime/dimensions/chunking identity; normal factory verifies before loading. | `pytest tests/test_v1_audit.py -k network`; model commands | FIXED |
| P1-07/P1-08 | P1 | Fixed temp name, incomplete fsync/permissions and raw parser errors. | Unique 0600 same-dir temp, file+parent fsync, atomic replace, private `.ctx`; strict version/path/authority/limit/embedding validation and actionable `ConfigError`. | `pytest tests/test_v1_audit.py -k config` | FIXED |
| P1-09 | P1 | Size was checked only before an ordinary path open. | Containment plus `O_NOFOLLOW` where available, descriptor identity/size validation before/after, bounded bytes and strict UTF-8. Threat boundary documented in `SECURITY.md`. | `pytest tests/test_security.py` | FIXED |
| P1-10 | P1 | MCP limits and workspace limits could disagree silently. | Absolute MCP annotations are ceilings; services consistently apply lower configured limits and report them through status/doctor errors. | `pytest tests/test_security.py` | FIXED |
| P1-11/P1-12/P1-13 | P1 | Checkpoint fields lost multiline lists; errors/security were generic searches. | Structural multiline/list/heading fields; error suffix/code evidence; security applicability from checkpoint/graph/same-doc normative evidence before semantic fallback. | `pytest tests/test_checkpoints.py` | FIXED |
| P1-14 | P1 | Graph could not distinguish declarations from examples. | PROSE/HEADING/CODE/INLINE_CODE/LINK/CHECKPOINT_FIELD origin, evidence and confidence; declaration owners outrank incidental uses. | `pytest tests/test_graph.py` | FIXED |
| P1-15 | P1 | Filters were not first-class across APIs. | Shared service signatures and CLI/MCP schema expose coherent pre-cutoff filters. | `pytest tests/test_v1_audit.py -k filters` | FIXED |
| P1-16 | P1 | Semantic/query caches were absent or risked unbounded growth. | Immutable generation-keyed vectors and a 128-entry identity/query/generation LRU; sync invalidates query cache. | benchmark command in `docs/evaluation.md` | FIXED |
| P1-17 | P1 | Outline loaded section text only to discard it. | Dedicated metadata-only SQL and `OutlineEntry`. | `pytest tests/test_cli.py` | FIXED |
| P1-19 | P1 | Durable/external Pydantic models accepted extras and were mutable. | Shared frozen `StrictModel`, extra forbidden, range validators, schema versions at durable top-level boundaries. | `uv run mypy src && pytest` | FIXED |
| P1-20/P1-21 | P1 | Schema creation was one-shot V1 and generation omitted algorithm identity. | V1→V2 migration preserves document config and rebuilds derived data; versions/fingerprint are persisted and status-checked. | `pytest tests/test_v1_audit.py -k migrates` | FIXED |
| P1-22 | P1 | Old benchmark date/environment could be mistaken for current evidence. | Reports carry 2026-09-18, OS/Python/CPU/dependency and exact identity; unavailable production results are explicitly not fabricated. | `docs/evaluation.md` | FIXED |
| P1-23 | P1 | No reproducible resolver lock. | Committed `uv.lock`; all gates use `uv sync --frozen`/`uv run`. | `uv sync --frozen` | FIXED |
| P1-24/P1-25 | P1 | CI/release and security posture were incomplete. | Python matrix, format/lint/mypy/test/build/wheel/MCP/migration jobs; separate production tier; `SECURITY.md`, `CHANGELOG.md`. | `.github/workflows/ci.yml` | FIXED |
| P1-26 | P1 | README lacked tested client examples/workflow. | Codex, Claude Code, Pi and generic stdio examples; no port; pack→inspect exact source→code workflow. | README smoke commands | FIXED |
| P1-27 | P1 | No read-only diagnostic. | `ctx doctor [--offline]` reports config/schema/FTS/source/model/MCP/SQLite/Python/algorithms/readiness without network or writes. | `uv run ctx doctor --help` | FIXED |
| P1-29 | P1 | Checkpoint/graph/line services lacked CLI parity. | `checkpoint show/context`, `refs`, `deps`, `lines`, all direct shared-service adapters. | `uv run ctx --help` | FIXED |
| P1-30/P1-31 | P1 | Generic auxiliary searches polluted packs and could displace the requested checkpoint. | Evidence-first category order; generic searches are conditional fallback; exact primary retained or explicit `PRIMARY_REQUIREMENT_TOO_LARGE`. | `pytest tests/test_context_pack.py tests/test_v1_audit.py -k budget` | FIXED |
| P1-32/P1-33 | P1 | Authority globally boosted irrelevant content and duplicates crowded results. | Relevance buckets then decisive authority/priority tie-break; hash/parent/document diversity after exact resolution. | `pytest tests/test_hybrid.py` | FIXED |
| P1-34/P1-35 | P1 | Evaluation lacked hard multi-document conflicts and token efficiency. | V1 fixture/report covers normative/reference/historical/informal sources, duplicate IDs, ambiguity, conflicts, tail retrieval, serialized size and compression. | evaluation commands | FIXED |
| P1-36 | P1 | MCP lifecycle/concurrency/disconnect behavior was unproved. | SDK-v2 subprocess tests plus 20 concurrent reads/sync generation regression; stdio-only run/finally close. | `pytest tests/test_mcp.py tests/test_v1_audit.py -k concurrent` | FIXED |
| P1-37 | P1 | Windows path/case/drive/junction behavior was underspecified. | Drive/root/backslash normalization, case-fold duplicate checks, resolve containment, no-follow where available; CI includes Windows. | Python matrix CI | FIXED |
| P2-01 | P2 | Deterministic syntax classes omitted several modern identifier forms. | Added kebab, namespace, dotted Python, file:symbol, flags, digit errors, section refs and anchor classes. | `pytest tests/test_hybrid.py tests/test_graph.py` | FIXED |
| P2-02 | P2 | Internal slug matching implied exact GitHub compatibility. | Parser/graph explicitly label internal deterministic slugs; duplicate anchors become ambiguity, not guessed GitHub semantics. | architecture docs | FIXED |
| P2-03 | P2 | Multi-query responses could mix generations. | Every source item and pack item carries generation; read/sync lock and final invariant assertions reject mixtures. | concurrency regression | FIXED |
| P2-04 | P2 | Query caching could retain stale text. | Only bounded query vectors are cached by identity/query/generation; source result caching is intentionally omitted pending evidence. | benchmark notes | FIXED |
| P2-05 | P2 | Hash timings could be read as production performance. | Separate mechanics and FastEmbed report sections; production claims require verified model identity. | `docs/evaluation.md` | FIXED |

## Root-cause trace summary

- **Budget:** trigger was long/escaped input; masking was short ASCII; symptom was a successful
  over-budget response; divergence began before candidate selection in a constant `+24` base;
  full serialized counting disproved source-text size as an adequate proxy.
- **Chunking/search:** trigger was one huge AST section with a tail match; normal short sections
  masked it; divergence was parser emission of one `Chunk`; exact source hashes disproved source
  corruption and identified candidate granularity as the failure.
- **Identity:** trigger was rename followed by old-path reuse; no reuse masked it; divergence was
  `sha256(path)` at registration; the preserved renamed row disproved random SQLite corruption.
- **Resolution:** trigger was duplicate checkpoint/reference labels; unique corpora masked it;
  divergence was `.fetchone()`/fan-out extraction; equal authority/priority is now positive
  disconfirming evidence against a unique answer.
- **Concurrency/generation:** trigger was MCP v2 worker threads and sync overlap; sequential CLI
  masked it; divergence was connection construction and per-subsystem commits; WAL alone did not
  fix Python thread ownership or logical atomicity.

## Migration notes

Opening a V1 database migrates it to SQLite schema V2. Document path/authority/priority
configuration is retained under newly generated opaque IDs. V1 sections, chunks, FTS, vectors,
graph and checkpoint rows are disposable and intentionally rebuilt by the next `ctx sync`;
V1 path-derived identity cannot safely survive old-path reuse. The migration is transactional,
sets `PRAGMA user_version=2`, and records a migration generation. Back up `.ctx/index.sqlite3`
before downgrade experiments; V1 binaries must not open V2 databases.
