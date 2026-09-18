# Dependency research record

Research date: **2026-09-18**. Sources were read from official project documentation and PyPI
release metadata; the lock file records the exact tested resolution.

## MCP Python SDK

The official SDK documentation and repository identify v2 as the current stable release line.
PyPI published `mcp 2.2.0` on 2026-09-07, requiring Python 3.10+. The official v2 migration guide
states that `FastMCP` was renamed to `MCPServer`, the supported high-level import is
`from mcp.server import MCPServer`, synchronous handlers run in worker threads, Python-facing
wire fields use snake_case, and `run()` defaults to stdio. It also says v1 remains a maintained
fallback only for projects not yet migrated.

Decision: use `mcp>=2.2,<3`, migrate to `MCPServer`, test with the official v2 client over a real
stdio subprocess, and retain no HTTP transport in ctx. There is no verified blocker requiring a
v1 pin.

Primary sources:

- <https://py.sdk.modelcontextprotocol.io/>
- <https://py.sdk.modelcontextprotocol.io/migration/>
- <https://github.com/modelcontextprotocol/python-sdk/tree/v2.2.0>
- <https://pypi.org/project/mcp/>

## Other dependencies

The implementation remains deliberately small: Pydantic for strict boundaries, Typer/Rich for
CLI presentation, markdown-it-py for source-mapped Markdown structure, SQLite/FTS5 from Python,
NumPy for bounded in-process cosine scoring, platformdirs for the global model cache, and
optional FastEmbed for a verified local ONNX model. No framework, vector server, hosted model,
telemetry SDK, remote database, or agent runtime is added.
