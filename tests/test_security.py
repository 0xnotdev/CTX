import asyncio
import os
import stat
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ctx.config import (
    ConfigError,
    LimitsConfig,
    add_document_config,
    initialize_workspace,
    load_config,
    read_source,
    save_config,
)
from ctx.models import Authority
from ctx.service import ContextEngine, StaleIndexError


def test_query_result_line_and_read_only_source_bounds(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    source = tmp_path / "safe.md"
    original = "# Safe\n" + "line\n" * 100
    source.write_text(original, encoding="utf-8")
    source.chmod(0o444)
    add_document_config(tmp_path, "safe.md", Authority.NORMATIVE)
    try:
        with ContextEngine(tmp_path) as engine:
            engine.index_workspace()
            assert source.read_text(encoding="utf-8") == original
            with pytest.raises(ValueError, match="max_query_chars"):
                engine.search("x" * 4_097)
            assert len(engine.search("line", limit=10_000)) <= engine.config.limits.max_results
            with pytest.raises(ValueError, match="2000"):
                engine.get_lines("safe.md", 1, 2_001)
            source.chmod(0o644)
            source.write_text(original.replace("# Safe", "# Changed"), encoding="utf-8")
            with pytest.raises(StaleIndexError, match="STALE_INDEX"):
                engine.search("line")
    finally:
        source.chmod(0o644)


def test_config_write_without_posix_fchmod(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows has no os.fchmod; mkstemp must remain a valid secure write path."""
    monkeypatch.delattr(os, "fchmod", raising=False)

    initialize_workspace(tmp_path)

    assert load_config(tmp_path).documents == ()
    config = tmp_path / ".ctx" / "config.toml"
    if os.name == "posix":
        assert stat.S_IMODE(config.stat().st_mode) == 0o600
    assert not list((tmp_path / ".ctx").glob(".config.*.tmp"))


def test_hostile_and_malformed_markdown_remains_inert_data(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    marker = tmp_path / "executed"
    hostile = (
        "---\nbroken: [\n---\n# Hostile\n"
        "<script>alert('x')</script>\n"
        f"[click](file://{marker})\n"
        "```sh\n"
        f"touch {marker}\n"
        ""  # deliberately unclosed fence
    )
    (tmp_path / "hostile.md").write_text(hostile, encoding="utf-8")
    add_document_config(tmp_path, "hostile.md", Authority.NORMATIVE)
    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        result = engine.search_exact("Hostile", limit=1)[0].source
        assert "<script>" in result.text
        assert "touch" in result.text
    assert not marker.exists()


def test_mcp_input_and_response_limits_are_enforced(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "large.md").write_text("# Large\n" + "x" * 8_000, encoding="utf-8")
    add_document_config(tmp_path, "large.md", Authority.NORMATIVE)
    config = load_config(tmp_path)
    config = config.model_copy(
        update={
            "limits": LimitsConfig(
                max_file_bytes=config.limits.max_file_bytes,
                max_query_chars=config.limits.max_query_chars,
                max_results=config.limits.max_results,
                max_response_chars=1_000,
            )
        }
    )
    save_config(tmp_path, config)
    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        section_id = engine.store.section_ids()[0]

    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ctx.mcp_server", str(tmp_path)],
        )
        async with stdio_client(parameters) as (read_stream, write_stream):  # noqa: SIM117
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                oversized_response = await session.call_tool(
                    "get_section", {"section_id": section_id}
                )
                assert oversized_response.is_error
                oversized_query = await session.call_tool("search", {"query": "q" * 4_097})
                assert oversized_query.is_error

    asyncio.run(exercise())


def test_source_growth_after_open_is_rechecked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    initialize_workspace(tmp_path)
    source = tmp_path / "growing.md"
    source.write_text("small", encoding="utf-8")
    config = add_document_config(tmp_path, "growing.md", Authority.NORMATIVE)
    limits = LimitsConfig(max_file_bytes=10)
    original_read = os.read
    changed = False

    def growing_read(descriptor: int, amount: int) -> bytes:
        nonlocal changed
        if not changed:
            changed = True
            source.write_text("x" * 20, encoding="utf-8")
        return original_read(descriptor, amount)

    monkeypatch.setattr(os, "read", growing_read)
    with pytest.raises(ConfigError, match="exceeds max_file_bytes after read"):
        read_source(tmp_path, config.documents[0], limits)
