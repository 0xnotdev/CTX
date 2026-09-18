import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ctx.config import add_document_config, initialize_workspace
from ctx.models import Authority
from ctx.service import ContextEngine


def test_stdio_mcp_lists_tools_and_uses_shared_services(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    marker = tmp_path / "must-not-exist"
    (tmp_path / "spec.md").write_text(
        "# CP-14 — MCP integration\n"
        "Goal: use RunManifest exactly.\n"
        "```sh\n"
        f"touch {marker}\n"
        "```\n"
        "# RunManifest\nInterface model RunManifest.\n",
        encoding="utf-8",
    )
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        section_id = engine.search_exact("CP-14", limit=1)[0].source.provenance.section_id
        direct_pack_ids = [
            item.source.provenance.section_id
            for item in engine.get_context_pack("Implement CP-14", 2_500).items
        ]

    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ctx.mcp_server", str(tmp_path)],
        )
        async with stdio_client(parameters) as (read_stream, write_stream):  # noqa: SIM117
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                tools = await session.list_tools()
                names = {tool.name for tool in tools.tools}
                assert {
                    "index_workspace",
                    "sync_workspace",
                    "list_documents",
                    "document_outline",
                    "index_status",
                    "get_section",
                    "get_lines",
                    "search",
                    "search_exact",
                    "find_symbol",
                    "get_references",
                    "get_dependencies",
                    "get_context_pack",
                    "get_checkpoint",
                    "get_checkpoint_context",
                } <= names

                exact = await session.call_tool("get_section", {"section_id": section_id})
                assert not exact.is_error
                assert exact.structured_content is not None
                assert exact.structured_content["provenance"]["section_id"] == section_id
                assert "touch" in exact.structured_content["text"]

                searched = await session.call_tool("search", {"query": "CP-14", "limit": 2})
                assert not searched.is_error
                assert searched.structured_content is not None
                assert (
                    searched.structured_content["result"][0]["source"]["provenance"]["section_id"]
                    == section_id
                )

                packed = await session.call_tool(
                    "get_context_pack", {"task": "Implement CP-14", "token_budget": 2_500}
                )
                assert not packed.is_error
                assert packed.structured_content is not None
                assert packed.structured_content["estimated_tokens"] <= 2_500
                assert packed.structured_content["completeness_status"]
                assert packed.structured_content["category_coverage"]
                assert packed.structured_content["items"]
                mcp_pack_ids = [
                    item["source"]["provenance"]["section_id"]
                    for item in packed.structured_content["items"]
                ]
                assert mcp_pack_ids == direct_pack_ids
                assert packed.structured_content["retrieval_metadata"]["active_channels"] == [
                    "structural",
                    "lexical",
                ]
                too_small = await session.call_tool(
                    "get_context_pack", {"task": "x" * 1_000, "token_budget": 64}
                )
                assert too_small.is_error
                assert too_small.structured_content is not None
                assert too_small.structured_content["code"] == "CONTEXT_BUDGET_TOO_SMALL"
                assert too_small.structured_content["minimum_required"] > 64

    asyncio.run(exercise())
    assert not marker.exists()


def test_official_v2_client_all_tools_concurrency_and_clean_disconnect(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    documents = {
        "spec.md": (
            "# CP-2 — Fixtures\nPrepare fixtures.\n"
            "# CP-14 — Lifecycle\nDependencies: CP-2, RunManifest, security.md#Constraints\n"
            "Goal: maintain atomic generation visibility.\n"
            "## Tests/acceptance criteria\nTwenty concurrent reads succeed.\n"
            "## Verify\n`pytest -q`\n"
        ),
        "architecture.md": (
            "# RunManifest\nInterface model RunManifest stores a deterministic seed.\n"
            "See spec.md#CP-14-Lifecycle.\n"
        ),
        "security.md": "# Constraints\nNever execute Markdown or open a listener.\n",
        "research.md": "# Giant research section\n" + ("distractor prose 🙂 " * 4_000),
        "old-spec.md": (
            "# CP-14 — Historical conflict\nRunManifest and atomic generations are forbidden.\n"
        ),
        "notes.md": (
            "# RunManifest\nInformal duplicate heading and [CP-14](spec.md#CP-14-Lifecycle).\n"
        ),
    }
    authorities = {
        "spec.md": Authority.NORMATIVE,
        "architecture.md": Authority.NORMATIVE,
        "security.md": Authority.NORMATIVE,
        "research.md": Authority.REFERENCE,
        "old-spec.md": Authority.HISTORICAL,
        "notes.md": Authority.INFORMAL,
    }
    for path, text in documents.items():
        (tmp_path / path).write_text(text, encoding="utf-8")
        add_document_config(tmp_path, path, authorities[path])
    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        checkpoint_id = engine.get_checkpoint("CP-14").metadata.root_section_id

    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ctx.mcp_server", str(tmp_path)],
        )
        async with stdio_client(parameters) as streams:  # noqa: SIM117
            async with ClientSession(*streams) as session:
                await session.initialize()
                expected = {
                    "index_workspace",
                    "sync_workspace",
                    "list_documents",
                    "document_outline",
                    "index_status",
                    "get_section",
                    "get_lines",
                    "search",
                    "search_exact",
                    "find_symbol",
                    "get_references",
                    "get_dependencies",
                    "get_context_pack",
                    "get_checkpoint",
                    "get_checkpoint_context",
                }
                tools = await session.list_tools()
                assert expected <= {tool.name for tool in tools.tools}

                calls = (
                    ("index_status", {}),
                    ("list_documents", {}),
                    ("document_outline", {"path": "spec.md"}),
                    ("search", {"query": "RunManifest", "limit": 3}),
                    ("search_exact", {"query": "CP-14", "limit": 3}),
                    ("find_symbol", {"symbol": "RunManifest"}),
                    ("get_section", {"section_id": checkpoint_id}),
                    ("get_lines", {"path": "spec.md", "start_line": 1, "end_line": 4}),
                    ("get_references", {"section_id": checkpoint_id}),
                    ("get_dependencies", {"section_id": checkpoint_id}),
                    ("get_checkpoint", {"checkpoint_id": "CP-14", "document": "spec.md"}),
                    (
                        "get_checkpoint_context",
                        {
                            "checkpoint_id": "CP-14",
                            "document": "spec.md",
                            "token_budget": 15_000,
                        },
                    ),
                    (
                        "get_context_pack",
                        {"task": "Implement CP-14", "token_budget": 8_000},
                    ),
                )
                for name, arguments in calls:
                    result = await session.call_tool(name, arguments)
                    assert not result.is_error, (name, result)
                    assert result.structured_content is not None

                concurrent = await asyncio.gather(
                    *(session.call_tool("search", {"query": "RunManifest"}) for _ in range(20)),
                    session.call_tool("sync_workspace", {}),
                    session.call_tool("sync_workspace", {}),
                )
                assert all(not result.is_error for result in concurrent)
                generations = {
                    result.structured_content["result"][0]["index_generation"]
                    for result in concurrent[:20]
                    if result.structured_content is not None
                }
                assert len(generations) == 1

        # A second subprocess is deliberately disconnected while a request may still be in
        # flight. Context-manager shutdown must cancel it and reap the child without hanging.
        async with stdio_client(parameters) as streams:  # noqa: SIM117
            async with ClientSession(*streams) as session:
                await session.initialize()
                pending = asyncio.create_task(
                    session.call_tool(
                        "get_context_pack",
                        {"task": "Implement CP-14", "token_budget": 8_000},
                    )
                )
                await asyncio.sleep(0)
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)

    asyncio.run(asyncio.wait_for(exercise(), timeout=45))
