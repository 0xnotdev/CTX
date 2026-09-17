# Architecture and dependency record

## Invariants and boundaries

`ctx` is layered local infrastructure: CLI/MCP adapters call shared application services,
which call a Markdown parser, retrieval engine, and SQLite repository. Source files are opened
read-only and remain authoritative. Every derived object is disposable navigation data.
V0 deliberately excludes web UI, auth/accounts/teams, SaaS or sync services, hosted APIs,
agent execution, general document chat, non-Markdown formats, and external databases.

## Dependency decisions (researched 2025-09-17)

Versions were checked against each project's PyPI release metadata (the authoritative package
metadata consumed by installers) and linked upstream documentation. Bounds deliberately allow
compatible patch/minor updates; the quality suite records the actually tested environment.

| Package | Selected line | Purpose / decision | License |
|---|---:|---|---|
| Pydantic | 2.x | typed boundary and domain models | MIT |
| Typer | 0.x | typed CLI | MIT |
| Rich | 14.x | readable terminal output | MIT |
| markdown-it-py | 4.x | CommonMark token maps/AST, never ad-hoc token slicing | MIT |
| NumPy | 2.x | measured in-process cosine brute force | BSD-3-Clause |
| FastEmbed | 0.x, optional | local ONNX embedding backend | Apache-2.0 |
| MCP Python SDK | 1.x/2.x, optional | official FastMCP stdio adapter | MIT |
| pytest | 8/9.x, dev | tests | MIT |
| Ruff | 0.x, dev | format/lint | MIT |
| mypy | 1.x, dev | strict type checking | MIT |

Metadata/docs: [Pydantic](https://pypi.org/project/pydantic/),
[Typer](https://pypi.org/project/typer/), [Rich](https://pypi.org/project/rich/),
[markdown-it-py](https://pypi.org/project/markdown-it-py/),
[NumPy](https://pypi.org/project/numpy/), [FastEmbed](https://pypi.org/project/fastembed/),
[MCP](https://pypi.org/project/mcp/), and [pytest](https://pypi.org/project/pytest/).
No LangChain, LlamaIndex, vector server, cloud API, or paid dependency is used.

## Local embedding decision

The production default is `BAAI/bge-small-en-v1.5` (384 dimensions) through FastEmbed's local
ONNX CPU runtime. Model download is explicit; cached operation sets `local_files_only`, and no
external embedding API or API key exists. Model identity and dimensions are persisted beside
each vector. The model is MIT licensed, independently of FastEmbed's Apache-2.0 package license.
Selection evidence, sources, measured quality figures, and our NumPy brute-force measurement
are in [`benchmarks/embedding_selection.md`](../benchmarks/embedding_selection.md). A
stable hash fixture keeps ordinary tests network-free.

## Hybrid retrieval

Queries are deterministically classified for checkpoint/section IDs, identifier shapes, paths,
and quoted phrases. Structural, FTS5/BM25, and local cosine rankings are combined with
reciprocal-rank fusion using `k=60`, then exact-heading/direct-identifier, authority, and
priority boosts plus stable provenance tie-breaking. RRF is based on Cormack, Clarke, and
Büttcher, *Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods*,
SIGIR 2009 (<https://doi.org/10.1145/1571941.1572114>). `k=60` is the paper's commonly reported
setting; it is recorded and evaluation-tested rather than claimed universally optimal.
Generated candidates sort behind original sources.

Context packs prioritize direct requirements, references, dependencies, interfaces/models,
global/error/security constraints, acceptance/verification material, then small structural
neighbors. The default token count is explicitly an estimate (`ceil(UTF-8 bytes / 4)`), and
AST block maps permit only valid whole-block reductions. Fences and tables are never sliced.
Checkpoint recognition is deterministic and extensible: recognized headings and canonical
fields are stored as navigation metadata while exact root/child sections remain the response
source. Optional `.ctx/checkpoints/CP-N.json` artifacts are bounded, hashed, labeled GENERATED,
and kept in a separate `generated_artifact` field; source fields always win and artifacts are
never interpreted as normative text.

`POSSIBLE_CONFLICT` is intentionally conservative: it requires an overlapping exact technical
identifier and one of a small deterministic contradictory phrase pairs (`must`/`must not`,
enabled/disabled, allowed/prohibited, required/forbidden). It labels both exact sources and
authorities and is **not** general natural-language contradiction detection or reconciliation.

The MCP adapter uses the official Python SDK's FastMCP 1.x API over local stdio. Version 2
renamed FastMCP and is intentionally excluded until a deliberate adapter migration; this is why
the dependency is bounded `<2`. Every tool delegates to `ContextEngine`, applies typed input and
configured response bounds, and returns Markdown/HTML/links/scripts only as inert JSON string
data. There is no execution path in the server.

Dependency package licensing does **not** imply a model license; users must review model
metadata before explicit download or redistribution.
