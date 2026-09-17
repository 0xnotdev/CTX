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
python3.12 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev,embeddings]'
ctx init
ctx add spec.md --authority normative
ctx add research.md --authority reference
ctx add architecture.md --authority normative
ctx index --download-model   # explicit first download; later runs are offline
ctx pack "Implement CP-14" --token-budget 7000 --json
ctx mcp                      # local stdio only
```

If the optional embedding package/model is unavailable, indexing and retrieval continue with
structure and FTS5 and print a warning; use `--no-embeddings` to make that choice explicit.
`ctx sync` does zero parse/index work for unchanged documents. Other commands are `status`,
`docs`, `outline`, `section`, `search`, `find`, `remove`, and `pack`; each retrieval operation
uses the same application service as MCP. `AGENTS.md` is a tiny recommended agent template.

## Development

```console
pytest
ruff check .
ruff format --check .
mypy src
```

Apache-2.0 licensed. Third-party and embedding-model license considerations are recorded in
[`docs/architecture.md`](docs/architecture.md); measured V0 quality, scale, and security results
are in [`docs/evaluation.md`](docs/evaluation.md).
