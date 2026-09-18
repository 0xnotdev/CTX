"""Opt-in cached production-BGE tier; never downloads a model."""

from __future__ import annotations

import asyncio
import os
import socket
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ctx.config import add_document_config, initialize_workspace
from ctx.embeddings import FastEmbedProvider
from ctx.evaluation import RetrievalCase, _channel_metrics
from ctx.models import Authority
from ctx.service import ContextEngine

pytestmark = pytest.mark.production_embedding


def _provider() -> FastEmbedProvider:
    if os.environ.get("CTX_RUN_PRODUCTION_EMBEDDINGS") != "1":
        pytest.skip("set CTX_RUN_PRODUCTION_EMBEDDINGS=1 with a verified cached model")
    model_dir = Path(os.environ["CTX_MODEL_DIR"])
    return FastEmbedProvider(cache_dir=model_dir)


def test_real_bge_low_overlap_tail_beats_lexical_decoy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("normal operation attempted network access")

    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    provider = _provider()
    initialize_workspace(tmp_path)
    filler = "Routine bookkeeping records are archived after processing. " * 1_500
    relevant = (
        "Each experimental repetition reconstructs identical evidence from a fixed pseudorandom "
        "initialization and compares byte-level manifests before accepting the run."
    )
    (tmp_path / "spec.md").write_text(
        f"# Repeated experiment contract\n{filler}{relevant}\n", encoding="utf-8"
    )
    (tmp_path / "notes.md").write_text(
        "# Reproducible seeded trials\n"
        "This lexical glossary mentions reproducible seeded trials but defines no behavior.\n",
        encoding="utf-8",
    )
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    add_document_config(tmp_path, "notes.md", Authority.INFORMAL)
    with ContextEngine(tmp_path, embedder=provider) as engine:
        engine.sync_workspace()
        target = engine.store.document_sections("spec.md")[0].provenance.section_id
        query = "What must happen before accepting rerun evidence?"
        semantic = engine.search_semantic(query, limit=3)
        assert semantic[0].source.provenance.section_id == target
        assert relevant in semantic[0].source.text
        case = [RetrievalCase("low-overlap-tail", query, target)]
        metrics = _channel_metrics(engine, case, "semantic")
        assert metrics.recall_at_1 == 1.0
        pack = engine.get_context_pack(query, 2_000)
        assert pack.items
        assert pack.serialized_estimated_tokens <= pack.token_budget


def test_real_model_stdio_mcp_remains_offline_under_socket_denial(tmp_path: Path) -> None:
    provider = _provider()
    initialize_workspace(tmp_path)
    (tmp_path / "spec.md").write_text(
        "# CP-14 — Offline MCP\nGoal: local retrieval only.\n", encoding="utf-8"
    )
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    with ContextEngine(tmp_path, embedder=provider) as engine:
        engine.sync_workspace()

    denial = tmp_path / "deny-network"
    denial.mkdir()
    (denial / "sitecustomize.py").write_text(
        "import socket\n"
        "def denied(*args, **kwargs): raise AssertionError('network attempted')\n"
        "socket.create_connection = denied\n"
        "socket.socket.connect = denied\n",
        encoding="utf-8",
    )

    async def exercise() -> None:
        environment = dict(os.environ)
        environment["CTX_MODEL_DIR"] = str(Path(os.environ["CTX_MODEL_DIR"]).resolve())
        environment["PYTHONPATH"] = str(denial) + os.pathsep + environment.get("PYTHONPATH", "")
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ctx.mcp_server", str(tmp_path)],
            env=environment,
        )
        async with stdio_client(parameters) as streams:  # noqa: SIM117
            async with ClientSession(*streams) as session:
                await session.initialize()
                status = await session.call_tool("index_status", {})
                assert not status.is_error
                assert status.structured_content is not None
                assert "semantic" in status.structured_content["active_channels"]
                search = await session.call_tool(
                    "search", {"query": "local offline retrieval", "limit": 3}
                )
                assert not search.is_error
                pack = await session.call_tool(
                    "get_context_pack",
                    {"task": "Implement CP-14", "token_budget": 2_000},
                )
                assert not pack.is_error

    asyncio.run(asyncio.wait_for(exercise(), timeout=45))
