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
                assert not exact.isError
                assert exact.structuredContent is not None
                assert exact.structuredContent["provenance"]["section_id"] == section_id
                assert "touch" in exact.structuredContent["text"]

                searched = await session.call_tool("search", {"query": "CP-14", "limit": 2})
                assert not searched.isError
                assert searched.structuredContent is not None
                assert (
                    searched.structuredContent["result"][0]["source"]["provenance"]["section_id"]
                    == section_id
                )

                packed = await session.call_tool(
                    "get_context_pack", {"task": "Implement CP-14", "token_budget": 1_500}
                )
                assert not packed.isError
                assert packed.structuredContent is not None
                assert packed.structuredContent["estimated_tokens"] <= 1_500
                assert packed.structuredContent["items"]

    asyncio.run(exercise())
    assert not marker.exists()
