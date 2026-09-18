# Checkpoint evidence ledger

Each checkpoint is implemented and accepted before the next starts. A checkpoint's row names
its clean commit; later rows record concrete hashes because a commit cannot contain its own ID.

| Checkpoint | Commit | Focused / acceptance commands | Observed result |
|---|---|---|---|
| CP-00 | `b776774` | `pytest`; `ctx --help`; `ctx --version`; `ruff check .`; `ruff format --check .`; `mypy src` | 2 tests passed; CLI exits 0; lint, format, and strict types pass |
| CP-01 | `18d879e` | `pytest tests/test_parser.py`; full `pytest`; lint/format/type | 6 parser and 8 total tests passed; AST sections reconstruct exact source |
| CP-02 | `7e3d3d7` | `pytest tests/test_store.py`; full `pytest`; lint/format/type | 4 store and 12 total tests passed; provenance, schema, path defenses verified |
| CP-03 | `6d70311` | `pytest tests/test_incremental.py`; full `pytest`; lint/format/type | 3 incremental and 15 total tests passed; zero-work sync, retained embedding, rename, delete, and stale guard verified |
| CP-04 | `8a18739` | `pytest tests/test_lexical.py`; full `pytest`; lint/format/type | 7 lexical and 22 total tests passed; all specified exact identifiers rank owning section first |
| CP-05 | `3093403` | `pytest tests/test_embeddings.py`; full `pytest`; `python benchmarks/cosine.py`; lint/format/type | 3 embedding and 25 total tests passed; 20k×384 cosine measured 9.31 ms/query; incremental vectors verified |
| CP-06 | `c469480` | `pytest tests/test_hybrid.py`; full `pytest`; lint/format/type | 3 hybrid and 28 total tests passed; classifier and stable RRF verified; CP-14 structural result first |
| CP-07 | `e9ed4ea` | `pytest tests/test_graph.py`; full `pytest`; lint/format/type | 2 graph and 30 total tests passed; six edge kinds, symbols, traversal, unresolved labels, and no self-edges verified |
| CP-08 | `aa39949` | `pytest tests/test_context_pack.py`; full `pytest`; lint/format/type | 3 pack and 33 total tests passed; deliberate categories, budget, AST-safe reduction/omission, provenance, and conservative conflicts verified |
| CP-09 | `a86e3fb` | `pytest tests/test_checkpoints.py`; full `pytest`; lint/format/type | 2 checkpoint and 35 total tests passed; structured fields, exact source, graph context, and GENERATED artifact precedence verified |
| CP-10 | `fd64b78` | `pytest tests/test_mcp.py`; full `pytest`; lint/format/type | stdio client listed 15 tools and exercised exact section, hybrid search, and pack; 36 total tests passed; hostile fence stayed inert |
| CP-11 | `85205a7` | `pytest tests/test_cli.py`; full `pytest`; `ctx --help`; lint/format/type; AGENTS word count | 4 CLI and 38 total tests passed; full init/add/index/search/section/pack workflow; template is 167 words |
| CP-12 | `d27efee` | focused eval/large/security/final tests; full `pytest`; Ruff; mypy; `python -m build`; wheel install/CLI smoke; MCP integration; benchmarks | 44 tests passed; quality metrics meet thresholds; package smoke and three-document installed CLI workflow passed; measured results in `docs/evaluation.md` |

## V1 audit phases

V1 repairs were executed in the required FIX-00 through FIX-11 order, with permanent
reproductions recorded in `docs/audit-remediation.md`. The final commit history groups phases where
cross-cutting schema/model/service changes could not safely be separated while retaining a
runnable tree:

| Phase | Delivered evidence |
|---|---|
| FIX-00 | lock, MCP 2.2 decision/migration, dependency record, security/changelog |
| FIX-01 | opaque document IDs, V1→V2 migration, behavior generations |
| FIX-02 | bounded AST multi-chunk source/embedding units and long-tail regression |
| FIX-03 | exact excerpts, match ranges, SQL pre-cutoff filters, metadata outline |
| FIX-04 | complete serialized budgeting, machine errors, compact conflicts, primary retention |
| FIX-05 | document-aware checkpoint/reference ambiguity, multiline fields, origin/error/security evidence |
| FIX-06 | thread-local SQLite, explicit write/read serialization and atomic sync/concurrency tests |
| FIX-07 | global verified model lifecycle, strong identity, strict offline and bounded caches |
| FIX-08 | shared engine factory, SDK-v2 MCP and complete CLI/service parity |
| FIX-09 | deterministic versus actual-BGE quality/performance tiers and dated raw reports |
| FIX-10 | private config, no-follow/post-read checks, Windows/path/security adversarial tests |
| FIX-11 | README/architecture/evaluation, CI, clean wheel gates and final review |
