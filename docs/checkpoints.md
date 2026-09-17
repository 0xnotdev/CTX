# Checkpoint evidence ledger

Each checkpoint is implemented and accepted before the next starts. A checkpoint's row names
its clean commit; later rows record concrete hashes because a commit cannot contain its own ID.

| Checkpoint | Commit | Focused / acceptance commands | Observed result |
|---|---|---|---|
| CP-00 | `b776774` | `pytest`; `ctx --help`; `ctx --version`; `ruff check .`; `ruff format --check .`; `mypy src` | 2 tests passed; CLI exits 0; lint, format, and strict types pass |
| CP-01 | `18d879e` | `pytest tests/test_parser.py`; full `pytest`; lint/format/type | 6 parser and 8 total tests passed; AST sections reconstruct exact source |
| CP-02 | `7e3d3d7` | `pytest tests/test_store.py`; full `pytest`; lint/format/type | 4 store and 12 total tests passed; provenance, schema, path defenses verified |
| CP-03 | `6d70311` | `pytest tests/test_incremental.py`; full `pytest`; lint/format/type | 3 incremental and 15 total tests passed; zero-work sync, retained embedding, rename, delete, and stale guard verified |
| CP-04 | `CP-04 lexical` | `pytest tests/test_lexical.py`; full `pytest`; lint/format/type | 7 lexical and 22 total tests passed; all specified exact identifiers rank owning section first |
