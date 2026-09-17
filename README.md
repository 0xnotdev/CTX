# ctx

`ctx` is local context infrastructure for coding agents. It indexes large Markdown knowledge
bases once, then returns exact, bounded source sections with provenance through a CLI or a
local stdio MCP server.

**The original Markdown is always authoritative.** Indexes, chunks, embeddings, graph edges,
metadata, and generated artifacts are navigation aids; they never replace or rewrite source.

## V0 scope

One user and one workspace, Markdown, SQLite/FTS5, local embeddings, CLI, and MCP over stdio.
There is no cloud service, account system, web UI, chat layer, agent runtime, or non-Markdown
ingestion.

## Quick start

```console
python -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,mcp,embeddings]'
ctx --help
```

The complete workflow and agent guidance are documented as later checkpoints land.

## Development

```console
pytest
ruff check .
ruff format --check .
mypy src
```

Apache-2.0 licensed. Third-party and embedding-model license considerations are recorded in
[`docs/architecture.md`](docs/architecture.md).
