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

Dependency package licensing does **not** imply a model license; users must review model
metadata before explicit download or redistribution.
