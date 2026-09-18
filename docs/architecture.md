# V1 architecture and offline contract

## Invariant

Original Markdown is authoritative. A `SourceSection` is one complete heading-delimited AST
unit; a `SourceExcerpt` is an exact contiguous subset. Both carry document/section/range hashes,
line/column/offset provenance, authority, and one `index_generation`. Search chunks, embedding
text, vectors, graph/checkpoint records, rankings, omissions, conflicts, and generated artifacts
are disposable navigation data and cannot replace source text.

## Components

```text
 Codex / Claude Code / generic MCP       CLI / Pi bash workflow
                 | stdio                           |
                 v                                 v
        MCP SDK 2 MCPServer  ───────────> create_context_engine(...)
                                             |
                                      ContextEngine services
                   ┌─────────────────────────┼────────────────────────┐
                   v                         v                        v
          Markdown AST/chunker       retrieval/context plan    graph/checkpoints
                   \                         |                       /
                    \                        v                      /
                     └──────── SQLite V2 + FTS5 + local vectors ───┘
                                      WAL / atomic generation
                                             |
                                      exact Markdown files
```

CLI and MCP contain input/output adaptation only. Both use `create_context_engine`; hybrid is
active only when a configured model manifest and every artifact checksum verify. Ordinary
non-strict operation otherwise reports explicit `LEXICAL_ONLY` structural+lexical mode. A
`require_semantic` request (implied by `strict_agent`) fails closed through the same factory and
engine with typed `SEMANTIC_RETRIEVAL_UNAVAILABLE`; no adapter has a private retrieval path.

## Parse and retrieval

`markdown-it-py` source maps define sections. Stable section identity derives from opaque
document ID, the canonical full heading path, and canonical duplicate occurrence. One shared CTX
internal function is used by parser IDs, graph/local-anchor resolution, FTS heading indexing,
structural search, heading-prefix filters, checkpoint heading recognition, and retrieval boosts:
NFKC then Unicode case-fold; retain Unicode alphanumeric characters and combining marks; map each
run of spaces, periods, underscores, hyphens, punctuation, parentheses, slashes, symbols/emoji, or
controls to one ASCII hyphen; trim hyphens; use `section` if empty. Thus canonically equivalent
headings are disambiguated deterministically as occurrences 1, 2, ... while exact source headings
remain untouched. This is **not claimed to implement GitHub/CommonMark or renderer anchor
semantics**. Only local `#anchor` links use the CTX scheme; external `path#anchor` links remain
explicitly unresolved, and duplicate canonical candidates remain ambiguous.

One section emits multiple overlapping semantic `search_chunks`. Each row stores section/window
ID, ordinal, exact source range/text/hash, separate heading-prefixed embedding text/hash, model
input estimate, and chunker version. Authoritative section boundaries never change. Windows target
320 embedding-model tokens with 48 tokens (15%) overlap and a hard 448-token input ceiling, leaving
64 tokens below the default BGE model's 512-token truncation limit. Verified FastEmbed uses its
local tokenizer's exact `token_count`; structural/lexical-only operation uses the documented
deterministic `ceil(UTF-8 bytes / 4) + 2` approximation. Line/sentence/word and fence boundaries
are preferred where possible; oversized fences/tables become exact contiguous windows. Optional
heading-path context exists only in embedding text. Hydrated evidence is always exact source text
at the stored offsets and hashes; overlapping windows are disposable navigation, not authority.

Structural, FTS and vector SQL receives document/authority/exclusion/heading/scope filters before
its cutoff. RRF fuses overlapping ranges rather than globally collapsing by section. Deterministic
selection first reserves room for distinct sections/documents, then admits at most three disjoint
ranges from one section; overlapping adjacent windows, identical hashes, and >=92% cross-section
token-set duplicates collapse. This bound was chosen to retain two or three distant requirements
in a huge section without allowing one section to consume a small top-k. Authority and priority
decide materially comparable candidates. Search defaults to matched exact excerpts and retains
each range location/hash. Complete sections require explicit opt-in.

Context planning orders exact checkpoint/requirement, explicit refs, dependencies, checkpoint
fields, interfaces/models, global normative constraints, task-specific error/security evidence,
and semantic fallback. Generic auxiliary searches are conditional. In `STRICT_AGENT` mode it
also derives at most sixteen deterministic discovery signals, in stable order, from task text,
checkpoint title, goal/why/scope, dependencies, files/modules, interfaces/models, CLI behavior,
acceptance, failures, security, verification, and normative requirement lines. The signals are
deterministically grouped into at most four global hybrid probes across configured normative
sources; one focused probe per normative document prevents a large document from hiding another
behind the global cutoff. Results are deterministically fused by section and
classified using source path/heading role, direct term/identifier overlap, and conservative
semantic-score gates (with lower gates only for repeated query evidence). A role-named document
is not required merely because vector search returned a top result. Material architecture,
decision, current-state/progress, security, acceptance, verification,
testing, dependency, and other normative constraints become required; bounded optional
background/history/research neighbors remain optional. Conservative classification prevents
generic normative hits from being indiscriminately declared relevant. No LLM participates.

Ambiguous graph records have no target and cannot be treated as certainty. Every selected item
reports category, reason, relevance, confidence and authority. A requested checkpoint root is
first and cannot be displaced; insufficient room produces `PRIMARY_REQUIREMENT_TOO_LARGE`.
Checkpoint context can resolve its root in one document while retaining required cross-document
graph targets.

Every pack reports top-level `COMPLETE`, `PARTIAL`, `AMBIGUOUS`, `CONFLICTING`, or
`NOT_APPLICABLE` and explicit coverage for primary evidence, dependencies, architecture,
security, acceptance, verification, and checkpoint descendants. Strict packs additionally report
decisions, current state, testing, and other normative coverage when applicable. Required evidence
is derived from checkpoint structure/resolved graph edges, direct query-term coverage, or strict
materiality classification; optional neighbors do not make a pack partial. Each dropped required
range has an actionable category/reason/confidence, document, section, checkpoint/dependency where
applicable, line range, and range hash. Ambiguities retain candidate source refs and conflicts
remain compact refs. Status precedence is conflict, ambiguity, required omission, complete
coverage, then not-applicable; callers must inspect the records rather than interpreting the enum
as a substitute for source.

A production coding agent may begin only when retrieval metadata reports `STRICT_AGENT`,
`HYBRID_SEMANTIC`, and an active `semantic` channel; status is `COMPLETE`; and required omissions,
ambiguities, and conflicts are empty. Otherwise CTX directs more retrieval, a larger budget, or
raw-Markdown fallback.

## Serialized token budgets

A counter implements `identity`, `count_text`, and `count_serialized`. Supported semantics are:

- `STRICT_BYTE_UPPER_BOUND`: one token per UTF-8 byte, no safety margin;
- `APPROXIMATE_GENERIC` (default): UTF-8 bytes/3 with an explicit 15% serialized margin;
- `MODEL_SPECIFIC`: injectable exact tokenizer implementations.

The budget includes task, JSON keys/escaping, provenance, category coverage, required omissions,
ambiguities, compact conflict refs, structured MCP result and JSON-RPC envelope. A fixed-point
count includes the count fields themselves. Results expose requested/effective budget,
method/identity, safety margin, content, metadata and serialized token counts. Strict mode never
silently expands: required evidence that cannot fit makes the result honestly `PARTIAL`, while an
unfit primary keeps the machine-readable error. The explicit CLI/MCP/API
`allow_required_budget_expansion` option may raise the effective budget up to the configured
maximum to return all required evidence, and discloses `budget_expanded=true` plus actual use. If
even the empty envelope cannot fit, `ContextBudgetTooSmall` gives requested, minimum, task
estimate and metadata overhead. `PossibleConflict` contains compact `SourceRef`s, never another
source copy.

## SQLite V2 and generations

Required durable objects are `documents`, `document_versions`, `sections`, `search_chunks`,
`embeddings`, `references`, `symbols`, `checkpoints`, `index_generations`, `index_metadata`, and
`search_chunks_fts`. Compatibility read views retain old V0 table names where harmless.

Document IDs are random opaque UUID-style values. Unique same-content rename detection can retain
an ID; an ambiguous rename is never guessed. Delete/recreate and old-path reuse obtain new IDs.
Checkpoints key `(document_id, checkpoint_id)`. Generated navigation artifacts use the same
identity on disk as `.ctx/checkpoints/<percent-encoded-document-id>/CP-N.json`; the namespace is
reversible and Windows-safe, survives a same-content rename because the opaque ID survives, and
is cleaned/ignored independently when that document is deleted. Scoped artifact provenance that
claims another document is rejected. Legacy unscoped `.ctx/checkpoints/CP-N.json` files migrate
atomically only when embedded `document_id` (preferred) or `document_path` identifies exactly one
active document; otherwise they remain explicitly ignored for rebuild/remediation and are never
attached by global CP label. References retain RESOLVED/UNRESOLVED/AMBIGUOUS, candidate IDs,
reason, evidence and syntax origin.

Connections are checked thread-local connections, not one `check_same_thread=False` connection.
Every connection enables foreign keys, WAL, 5000 ms busy timeout and `synchronous=NORMAL`.
Writes take a process-local path lock and `BEGIN IMMEDIATE`. NORMAL+WAL protects committed state
from process crashes but does not claim power-loss durability equivalent to FULL.

Sync reads and parses all immutable source inputs, extracts graph/checkpoints, reuses valid vectors,
and computes missing vectors before the write. One transaction replaces structure, FTS, vectors,
graph, checkpoint and behavior metadata and increments generation exactly once. Multi-query reads
hold the same logical operation lock; source-bearing responses assert one generation. Readers see
old or new complete state, not a mixture.

Generation behavior fingerprints include document path/authority/priority/hash; schema, parser,
chunker, embedding-text, graph, checkpoint and retrieval versions; generated-artifact fingerprint;
and full embedding identity/dimensions. Status distinguishes CLEAN, SOURCE_STALE,
METADATA_STALE, PARSER_STALE, EMBEDDINGS_STALE, GRAPH_STALE, SCHEMA_STALE and MISSING_SOURCE.

### V1 migration

Opening schema V1 transactionally preserves configured path, authority and priority, assigns new
opaque IDs, drops disposable derived rows, creates V2 and records `V1_TO_V2_MIGRATION`. The next
sync rebuilds exact derived state. This intentional identity reset is required because a V1 ID was
a path hash and could not safely represent old-path reuse.

## Local model lifecycle and caches

The platformdirs cache defaults to `~/.cache/ctx/models` on Linux and has environment/config/CLI
overrides. Only `ctx model download` can pass `local_files_only=False`. `model install` copies
already-local files. Verification checks every artifact against a manifest before ONNX startup.
Normal index/sync/search/pack/MCP/checkpoint/doctor operations neither download nor update-check
nor emit telemetry.

Embedding identity includes provider, model, exact configured revision, aggregate artifact hash,
dimensions and runtime version; algorithm metadata separately binds chunker and embedding-text
versions. A mismatch invalidates vectors. Before a required-semantic pack, CTX validates provider
initialization, indexed identity/dimensions/algorithm metadata, one compatible source- and
embedding-hash-bound finite vector per current chunk, and query execution. Missing/corrupt model,
checksum/revision mismatch, provider failure, missing/stale/incompatible vectors, or semantic
query failure becomes a typed machine-readable error rather than lexical fallback. Normal runtime
remains local-only and never installs/downloads a model.

The semantic matrix cache is immutable, bounded to four generation+identity+filter entries. Query
vectors use a 128-entry identity+query+generation LRU. Generation changes make old entries
unreachable; source response text is not cached.

## Security boundary and dependencies

Markdown/HTML/links/fences are inert strings. Source containment rejects absolute, drive-rooted,
lexical and resolved escapes. Linux uses `O_NOFOLLOW`; all platforms compare descriptor identity
and recheck actual read size. No portable userspace design is perfectly race-proof against an
untrusted writer controlling the whole directory tree; such a workspace is outside the boundary.
See [`../SECURITY.md`](../SECURITY.md).

Dependency research dated 2026-09-18 is in [`dependency-research.md`](dependency-research.md).
The official stable MCP SDK is 2.2.x (`MCPServer`); ctx exposes only its stdio transport. Pydantic,
Typer/Rich, markdown-it-py, SQLite, NumPy, platformdirs and optional FastEmbed/ONNX are the only
major runtime layers. There is no LangChain/LlamaIndex, remote API, hosted vector service or
external database. Package licenses do not grant model redistribution rights.
