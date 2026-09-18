# ctx

`ctx` is a fully local context layer for large Markdown-first, specification-driven projects. It
indexes exact source into SQLite/FTS5, optional verified local embeddings, a deterministic graph,
and document-aware checkpoint metadata, then serves compact source excerpts through a CLI or
**stdio-only** MCP server.

> **Original Markdown is authoritative.** Sections and excerpts are exact contiguous source
> ranges with SHA-256 provenance and one committed index generation. Chunks, vectors, graph
> edges, checkpoint fields, rankings, conflicts, and generated artifacts are navigation aids.

There is no SaaS, account, cloud database, vector server, remote inference, remote embedding,
telemetry, web UI, network listener, or agent runtime. Normal operation is offline-capable.

## Install and first workspace

Python 3.12+ is supported. For development, use the committed lock:

```console
uv sync --frozen
uv run ctx --version
```

For an installed release:

```console
python -m pip install 'ctx-context[embeddings]'
ctx init
ctx add spec.md --authority normative --priority 10
ctx add architecture.md --authority normative
ctx add research.md --authority reference
ctx index --no-embeddings
# Explicit non-production lexical-only pack; output labels LEXICAL_ONLY:
ctx pack "Implement CP-14" --no-embeddings --token-budget 15000 --json
# Production/agent pack (after installing and indexing with the verified local model):
ctx pack "Implement CP-14" --strict-agent --require-semantic --token-budget 15000 --json
# Explicit policy opt-in; output discloses requested and expanded effective budget:
ctx pack "Implement CP-14" --strict-agent --require-semantic --token-budget 15000 \
  --allow-required-budget-expansion --json
```

The default search response is a match-centered exact `SourceExcerpt`; request
`ctx search ... --include-full-section` only when a response limit permits the complete section.
`ctx outline` is metadata-only.

## Optional local semantic model

Normal commands never download or check for a model. The only network-capable operation is the
explicit command below:

```console
ctx model status --root /repo --json
ctx model download --model BAAI/bge-small-en-v1.5       # explicit network opt-in
ctx model verify --model BAAI/bge-small-en-v1.5
ctx index --root /repo                                  # verified local model only
```

The global cache follows platform conventions (`~/.cache/ctx/models` on Linux). Override it with
`CTX_MODEL_DIR`, `[embedding].model_dir`, or `--model-dir`. `ctx model install PATH` installs
already-local artifacts without network. A manifest binds provider/runtime, model, revision,
artifact checksums, dimensions, runtime, chunker, and embedding-text behavior. In ordinary
non-strict operation, missing or invalid artifacts produce an explicit `LEXICAL_ONLY`
structural+lexical mode shown in pack metadata. `--require-semantic` and `--strict-agent` instead
fail closed with machine-readable `SEMANTIC_RETRIEVAL_UNAVAILABLE`; no strict pack is emitted.
Set `[embedding].require_semantic = true` to make that fail-closed policy the workspace default.

## Agent workflow

1. Call `get_context_pack` (or `ctx pack`) with `strict_agent=true`/`--strict-agent` and
   `require_semantic=true`/`--require-semantic` for the task/checkpoint.
2. Begin implementation **only if** retrieval metadata reports `context_mode=STRICT_AGENT`,
   `retrieval_mode=HYBRID_SEMANTIC`, `active_channels` contains `semantic`,
   `completeness_status=COMPLETE`, and required omissions, ambiguities, and conflicts are empty.
3. Inspect category coverage and exact source, range hash, section hash, authority, and generation.
   Strict discovery uses bounded deterministic signals from checkpoint title/goal, requirements,
   dependencies, acceptance, files/modules, interfaces, security, and verification to classify
   materially relevant normative evidence across specification, architecture, decisions,
   progress/current state, security, testing, and other configured normative documents.
4. Re-read decisive IDs with `get_section`/`get_lines` if complete wording is needed.
5. Code only from authoritative source; never treat graph, embeddings, rankings, or generated
   prose as a requirement.
6. Otherwise stop and request more retrieval, a larger budget, explicit required-budget
   expansion, or raw-Markdown inspection. This includes `PARTIAL`, `AMBIGUOUS`, `CONFLICTING`,
   semantic unavailability, stale source, and budget errors. Optional background/neighbors may be
   omitted without making a strict pack partial.

Useful commands:

```console
ctx status --json
ctx doctor --offline --json
ctx search "RunManifest" --document spec.md --authority-floor normative --json
ctx lines spec.md 120 180 --json
ctx refs SECTION_ID --json
ctx deps SECTION_ID --json
ctx checkpoint show CP-14 --document spec.md --json
ctx checkpoint context CP-14 --document spec.md --strict-agent --require-semantic \
  --budget 15000 --json
ctx sync
```

## Local stdio MCP

The server is `ctx mcp --root /absolute/repo`. It writes protocol frames only to stdout and opens
no listener. Diagnostics belong on stderr. Examples use an absolute workspace path. MCP
`get_context_pack` and `get_checkpoint_context` expose the same `strict_agent` and
`require_semantic` booleans as the CLI; typed semantic failures are returned as tool errors with
structured `code`, `reason`, identity/generation details, and active channels.

### Codex

`~/.codex/config.toml`:

```toml
[mcp_servers.ctx]
command = "ctx"
args = ["mcp", "--root", "/absolute/repo"]
startup_timeout_sec = 30
```

### Claude Code

```console
claude mcp add --transport stdio --scope user ctx -- ctx mcp --root /absolute/repo
claude mcp get ctx
```

### Generic MCP client

```json
{
  "mcpServers": {
    "ctx": {
      "command": "ctx",
      "args": ["mcp", "--root", "/absolute/repo"]
    }
  }
}
```

The integration suite launches this command with the official MCP Python SDK 2 client and lists
and calls the status, outline, search, exact, symbol, section, lines, refs, deps, checkpoint,
checkpoint-context, and context-pack services.

### Pi

Current Pi intentionally has **no built-in MCP client**. Do not invent an MCP settings entry. The
tested local integration is Pi's built-in `bash` tool calling the same shared CLI services:

```text
Use `ctx pack "<task>" --strict-agent --require-semantic --token-budget 15000 --json`, enforce
the production contract above, inspect exact sources with `ctx section <id> --json` or
`ctx lines ...`, then implement from the Markdown.
```

This repository's `AGENTS.md` supplies that workflow automatically when Pi starts in the project.
An audited third-party Pi MCP extension may launch the same generic stdio command above, but such
an extension runs arbitrary code and is outside ctx's trust boundary.

## Limits, trust, and platforms

Workspace limits and MCP absolute ceilings are both enforced. Source paths must be relative
Markdown paths contained by the workspace. Reads use no-follow opening where available and
verify descriptor identity/size after reading. See [`SECURITY.md`](SECURITY.md) for the exact
local-race boundary.

Linux is the primary measured platform. CI covers Linux, macOS, and Windows path/SQLite/stdio
behavior on Python 3.12 and 3.13. Windows drive/case/backslash paths and junction/symlink escape
are rejected or contained; ONNX startup requires a compatible `onnxruntime` wheel.

## Development and evidence

```console
uv sync --frozen
uv run ruff format --check .
uv run ruff check .
uv run mypy src
uv run pytest -q
uv build
```

Production BGE tests are intentionally separate and never download:

```console
CTX_RUN_PRODUCTION_EMBEDDINGS=1 CTX_MODEL_DIR=/verified/cache \
  uv run pytest -q tests/test_production_embeddings.py
```

Architecture, migrations, measured retrieval/performance results, audit repairs, and final gate
evidence are in [`docs/architecture.md`](docs/architecture.md),
[`docs/evaluation.md`](docs/evaluation.md), [`docs/audit-remediation.md`](docs/audit-remediation.md),
and [`FINAL_PRODUCTION_HARDENING_REPORT.md`](FINAL_PRODUCTION_HARDENING_REPORT.md). Apache-2.0
licensed; model licensing is separate and must be reviewed before redistribution.
