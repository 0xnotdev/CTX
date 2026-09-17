# Checkpoint evidence ledger

Each checkpoint is implemented and accepted before the next starts. A checkpoint's row names
its clean commit; later rows record concrete hashes because a commit cannot contain its own ID.

| Checkpoint | Commit | Focused / acceptance commands | Observed result |
|---|---|---|---|
| CP-00 | `b776774` | `pytest`; `ctx --help`; `ctx --version`; `ruff check .`; `ruff format --check .`; `mypy src` | 2 tests passed; CLI exits 0; lint, format, and strict types pass |
| CP-01 | `CP-01 parser` | `pytest tests/test_parser.py`; full `pytest`; lint/format/type | 6 parser and 8 total tests passed; AST sections reconstruct exact source |
