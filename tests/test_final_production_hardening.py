from __future__ import annotations

import asyncio
import hashlib
import json
import socket
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from typer.testing import CliRunner

from ctx.cli import app
from ctx.config import add_document_config, initialize_workspace, load_config, save_config
from ctx.models import Authority, CompletenessStatus, CoverageCategory, CoverageStatus
from ctx.service import ContextEngine, SemanticRetrievalError, create_context_engine


class ConceptEmbedding:
    """Small deterministic fixture that gives the corpus intentionally low lexical overlap."""

    dimensions = 8
    identity = "ctx/final-concept-fixture:1:8"
    metadata = {
        "provider": "ctx",
        "model_name": "final-concept-fixture",
        "revision": "1",
        "artifact_sha256": "deterministic-test-only",
        "runtime_version": "1",
    }

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def _one(self, text: str) -> np.ndarray[Any, np.dtype[np.float32]]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        vector[1 if "unrelated archive" in text.casefold() else 0] = 1.0
        return vector

    def embed_documents(self, texts: list[str]) -> np.ndarray[Any, np.dtype[np.float32]]:
        return np.stack([self._one(text) for text in texts])

    def embed_query(self, query: str) -> np.ndarray[Any, np.dtype[np.float32]]:
        return self._one(query)


class QueryFailureEmbedding(ConceptEmbedding):
    identity = "ctx/final-query-failure:1:8"

    def embed_query(self, query: str) -> np.ndarray[Any, np.dtype[np.float32]]:
        raise RuntimeError("fixture query failure")


def _configure(root: Path, documents: dict[str, str]) -> None:
    initialize_workspace(root)
    for path, text in documents.items():
        (root / path).write_text(text, encoding="utf-8")
        add_document_config(root, path, Authority.NORMATIVE)


def _strict_corpus() -> dict[str, str]:
    return {
        "SPEC.md": (
            "# CP-17 — Seamless policy transition\n"
            "## Goal\nMove request authorization without interrupting service.\n"
            "## Files/modules\n`src/router.py`, `PolicyEpoch`, and `activate_candidate`.\n"
            "## Acceptance criteria\nRequests remain available throughout activation.\n"
            "## Verify\nRun the handoff integration suite.\n"
        ),
        "ARCHITECTURE.md": (
            "# Control-plane topology\n"
            "Two evaluators coexist behind an epoch pointer; only the elected generation may "
            "serve decisions.\n"
        ),
        "DECISIONS.md": (
            "# ADR-42 — Shadow handoff\n"
            "The successor remains dark beside the incumbent until a quorum receipt makes the "
            "ownership flip irreversible.\n" + "Decision detail. " * 500
        ),
        "PROGRESS.md": (
            "# Current delivery state\n"
            "The incumbent evaluator owns production traffic and no shadow implementation exists "
            "yet.\n" + "Current-state detail. " * 500
        ),
        "SECURITY.md": (
            "# Threat boundary\n"
            "Treat candidate rule bundles as hostile data and never execute their embedded "
            "payloads.\n"
        ),
        "TESTING.md": (
            "# Release qualification\n"
            "A canary matrix must prove uninterrupted reads across the ownership flip "
            "and rollback.\n"
        ),
        "OPERATIONS.md": (
            "# Rollout constraint\n"
            "Operators must retain the prior epoch receipt until the successor is durable.\n"
        ),
        "BACKGROUND.md": (
            "# Background history\n"
            "Optional historical context about earlier authorization systems.\n"
            + "Background only. "
            * 5_000
        ),
        "ARCHIVE.md": "# Unrelated archive\nOld bibliography with no implementation effect.\n",
        "UNRELATED_ARCHITECTURE.md": (
            "# Platform topology\nUnrelated archive bibliography with no implementation effect.\n"
        ),
    }


def _coverage(pack: Any, category: CoverageCategory) -> Any:
    return next(item for item in pack.category_coverage if item.category is category)


def test_strict_agent_cross_document_completeness_and_exact_omissions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def denied(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("strict CTX runtime attempted network access")

    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    _configure(tmp_path, _strict_corpus())
    with ContextEngine(tmp_path, embedder=ConceptEmbedding()) as engine:
        engine.sync_workspace()
        complete = engine.get_context_pack(
            "Implement CP-17",
            60_000,
            strict_agent=True,
            require_semantic=True,
        )
        paths = {item.source.provenance.document_path for item in complete.items}
        assert {
            "SPEC.md",
            "ARCHITECTURE.md",
            "DECISIONS.md",
            "PROGRESS.md",
            "SECURITY.md",
            "TESTING.md",
            "OPERATIONS.md",
        } <= paths
        assert "ARCHIVE.md" not in paths
        assert "UNRELATED_ARCHITECTURE.md" not in paths
        assert complete.completeness_status is CompletenessStatus.COMPLETE
        assert not complete.omitted_required_evidence
        for excluded_path, category in (
            ("DECISIONS.md", CoverageCategory.DECISIONS),
            ("PROGRESS.md", CoverageCategory.CURRENT_STATE),
        ):
            excluded = engine.get_context_pack(
                "Implement CP-17",
                60_000,
                strict_agent=True,
                require_semantic=True,
                exclude_documents={excluded_path},
            )
            assert excluded.completeness_status is CompletenessStatus.PARTIAL
            omission = next(
                item for item in excluded.omitted_required_evidence if item.category is category
            )
            assert omission.document_path == excluded_path
            assert omission.section_id and omission.range_sha256
            assert "excluded by context-pack filters" in omission.reason
        assert complete.retrieval_metadata["context_mode"] == "STRICT_AGENT"
        assert complete.retrieval_metadata["retrieval_mode"] == "HYBRID_SEMANTIC"
        assert complete.retrieval_metadata["require_semantic"] is True
        assert complete.retrieval_metadata["active_channels"] == [
            "structural",
            "lexical",
            "semantic",
        ]
        for category in (
            CoverageCategory.ARCHITECTURE,
            CoverageCategory.DECISIONS,
            CoverageCategory.CURRENT_STATE,
            CoverageCategory.SECURITY,
            CoverageCategory.TESTING,
            CoverageCategory.NORMATIVE,
        ):
            coverage = _coverage(complete, category)
            assert coverage.required
            assert coverage.status is CoverageStatus.COVERED

        repeated = engine.get_context_pack(
            "Implement CP-17", 60_000, strict_agent=True, require_semantic=True
        )
        assert [item.source.ref for item in repeated.items] == [
            item.source.ref for item in complete.items
        ]

        required_only = engine.get_context_pack(
            "Implement CP-17",
            60_000,
            strict_agent=True,
            require_semantic=True,
            exclude_documents={"BACKGROUND.md"},
        )
        optional_omitted = engine.get_context_pack(
            "Implement CP-17",
            required_only.serialized_estimated_tokens + 500,
            strict_agent=True,
            require_semantic=True,
        )
        optional_paths = {item.source.provenance.document_path for item in optional_omitted.items}
        assert "BACKGROUND.md" not in optional_paths
        assert optional_omitted.completeness_status is CompletenessStatus.COMPLETE
        assert optional_omitted.omitted_relevant_sections
        assert not optional_omitted.omitted_required_evidence

        decision_item = next(
            item for item in required_only.items if item.category == "decision_constraint"
        )
        partial = engine.get_context_pack(
            "Implement CP-17",
            required_only.serialized_estimated_tokens - decision_item.estimated_tokens,
            strict_agent=True,
            require_semantic=True,
        )
        assert partial.completeness_status is CompletenessStatus.PARTIAL
        assert {item.category for item in partial.omitted_required_evidence} & {
            CoverageCategory.DECISIONS,
            CoverageCategory.CURRENT_STATE,
        }
        for omission in partial.omitted_required_evidence:
            if omission.category not in {
                CoverageCategory.DECISIONS,
                CoverageCategory.CURRENT_STATE,
            }:
                continue
            assert omission.document_id
            assert omission.document_path in {"DECISIONS.md", "PROGRESS.md"}
            assert omission.section_id
            assert omission.start_line and omission.end_line
            assert omission.range_sha256
            source = (tmp_path / omission.document_path).read_text(encoding="utf-8")
            assert hashlib.sha256(source.encode()).hexdigest() == omission.range_sha256
            assert "strict cross-document discovery" in omission.reason


def test_strict_agent_preserves_ambiguity_and_conflict_visibility(tmp_path: Path) -> None:
    _configure(
        tmp_path,
        {
            "SPEC.md": "# CP-17 — Network switch\nDependencies: CP-2\nNetworkPolicy rollout.\n",
            "ONE.md": "# CP-2 — First\nFirst equal dependency.\n",
            "TWO.md": "# CP-2 — Second\nSecond equal dependency.\n",
            "DECISIONS.md": "# NetworkPolicy decision\nNetworkPolicy must allow ingress.\n",
            "SECURITY.md": "# NetworkPolicy security\nNetworkPolicy must not allow ingress.\n",
        },
    )
    with ContextEngine(tmp_path, embedder=ConceptEmbedding()) as engine:
        engine.sync_workspace()
        pack = engine.get_context_pack(
            "Implement CP-17", 20_000, strict_agent=True, require_semantic=True
        )
        assert pack.ambiguous_evidence
        assert pack.possible_conflicts
        assert pack.completeness_status is CompletenessStatus.CONFLICTING


def test_configured_require_semantic_is_enforced_by_shared_factory(tmp_path: Path) -> None:
    _configure(tmp_path, {"SPEC.md": "# CP-17 — Configured\nImplement the contract.\n"})
    config = load_config(tmp_path)
    save_config(
        tmp_path,
        config.model_copy(
            update={"embedding": config.embedding.model_copy(update={"require_semantic": True})}
        ),
    )
    reloaded = load_config(tmp_path)
    assert reloaded.embedding.require_semantic is True
    with pytest.raises(SemanticRetrievalError) as captured:
        create_context_engine(tmp_path, model_dir=tmp_path / "missing")
    assert captured.value.reason == "MODEL_MISSING"


def test_semantic_strict_mode_fails_closed_but_lexical_mode_is_explicit(tmp_path: Path) -> None:
    _configure(tmp_path, {"SPEC.md": "# CP-17 — Local\nImplement the local contract.\n"})
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        status = engine.status()
        assert status.retrieval_mode == "LEXICAL_ONLY"
        assert status.active_channels == ("structural", "lexical")
        lexical = engine.get_context_pack("Implement CP-17", 5_000)
        assert lexical.retrieval_metadata["retrieval_mode"] == "LEXICAL_ONLY"
        assert lexical.retrieval_metadata["context_mode"] == "STANDARD"
        assert lexical.retrieval_metadata["require_semantic"] is False
        assert lexical.retrieval_metadata["active_channels"] == ["structural", "lexical"]
        with pytest.raises(SemanticRetrievalError) as captured:
            engine.get_context_pack(
                "Implement CP-17", 5_000, strict_agent=True, require_semantic=True
            )
        assert captured.value.as_dict()["code"] == "SEMANTIC_RETRIEVAL_UNAVAILABLE"
        assert captured.value.as_dict()["reason"] == "PROVIDER_UNAVAILABLE"


def test_semantic_strict_mode_rejects_missing_incompatible_and_corrupt_vectors(
    tmp_path: Path,
) -> None:
    _configure(tmp_path, {"SPEC.md": "# CP-17 — Local\nImplement the local contract.\n"})
    provider = ConceptEmbedding()
    with ContextEngine(tmp_path, embedder=provider) as engine:
        engine.sync_workspace()
        engine.store.connection.execute(
            "DELETE FROM embeddings WHERE chunk_id=(SELECT chunk_id FROM embeddings LIMIT 1)"
        )
        engine.store.connection.commit()
        with pytest.raises(SemanticRetrievalError) as missing:
            engine.get_context_pack(
                "Implement CP-17", 5_000, strict_agent=True, require_semantic=True
            )
        assert missing.value.reason == "VECTOR_INDEX_MISSING"
        assert missing.value.active_channels == ("structural", "lexical")
        degraded = engine.get_context_pack("Implement CP-17", 5_000)
        assert degraded.retrieval_metadata["retrieval_mode"] == "LEXICAL_ONLY"
        assert degraded.retrieval_metadata["active_channels"] == [
            "structural",
            "lexical",
        ]
        assert all("semantic" not in item.reason for item in degraded.items)

        engine.sync_workspace()
        engine.store.connection.execute(
            "UPDATE index_metadata SET value='wrong-identity' WHERE key='embedding_identity'"
        )
        engine.store.connection.commit()
        with pytest.raises(SemanticRetrievalError) as incompatible:
            engine.get_context_pack(
                "Implement CP-17", 5_000, strict_agent=True, require_semantic=True
            )
        assert incompatible.value.reason == "IDENTITY_MISMATCH"

        engine.store.connection.execute(
            "UPDATE index_metadata SET value=? WHERE key='embedding_identity'", (provider.identity,)
        )
        engine.store.connection.execute(
            "UPDATE embeddings SET chunk_source_sha256='stale' "
            "WHERE chunk_id=(SELECT chunk_id FROM embeddings LIMIT 1)"
        )
        engine.store.connection.commit()
        with pytest.raises(SemanticRetrievalError) as stale:
            engine.get_context_pack(
                "Implement CP-17", 5_000, strict_agent=True, require_semantic=True
            )
        assert stale.value.reason == "VECTOR_INDEX_STALE"

        engine.store.connection.execute(
            "UPDATE embeddings SET chunk_source_sha256=(SELECT source_sha256 "
            "FROM search_chunks WHERE search_chunks.id=embeddings.chunk_id)"
        )
        engine.store.connection.execute(
            "UPDATE embeddings SET vector=x'00' "
            "WHERE chunk_id=(SELECT chunk_id FROM embeddings LIMIT 1)"
        )
        engine.store.connection.commit()
        with pytest.raises(SemanticRetrievalError) as corrupt:
            engine.get_context_pack(
                "Implement CP-17", 5_000, strict_agent=True, require_semantic=True
            )
        assert corrupt.value.reason == "VECTOR_INDEX_CORRUPT"
        corrupt_status = engine.status()
        assert corrupt_status.category == "EMBEDDINGS_STALE"
        assert corrupt_status.retrieval_mode == "LEXICAL_ONLY"
        assert corrupt_status.active_channels == ("structural", "lexical")


def test_semantic_query_failure_and_factory_initialization_failure_are_typed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _configure(tmp_path, {"SPEC.md": "# CP-17 — Local\nImplement the local contract.\n"})
    with ContextEngine(tmp_path, embedder=QueryFailureEmbedding()) as engine:
        engine.sync_workspace()
        with pytest.raises(SemanticRetrievalError) as query_failure:
            engine.get_context_pack(
                "Implement CP-17", 5_000, strict_agent=True, require_semantic=True
            )
        assert query_failure.value.reason == "QUERY_FAILED"
        assert "fixture query failure" in query_failure.value.detail

    def broken_provider(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("model artifact checksum mismatch: fixture")

    monkeypatch.setattr("ctx.service.FastEmbedProvider", broken_provider)
    with pytest.raises(SemanticRetrievalError) as initialization:
        create_context_engine(tmp_path, require_semantic=True)
    assert initialization.value.reason == "MODEL_CHECKSUM_MISMATCH"

    def wrong_revision(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("installed model revision does not match requested revision")

    monkeypatch.setattr("ctx.service.FastEmbedProvider", wrong_revision)
    with pytest.raises(SemanticRetrievalError) as revision:
        create_context_engine(tmp_path, require_semantic=True)
    assert revision.value.reason == "MODEL_IDENTITY_MISMATCH"

    def initialization_failure(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("verified local provider failed to initialize")

    monkeypatch.setattr("ctx.service.FastEmbedProvider", initialization_failure)
    with pytest.raises(SemanticRetrievalError) as provider_failure:
        create_context_engine(tmp_path, require_semantic=True)
    assert provider_failure.value.reason == "PROVIDER_INITIALIZATION_FAILED"


def test_cli_strict_option_and_explicit_lexical_metadata(tmp_path: Path) -> None:
    _configure(tmp_path, {"SPEC.md": "# CP-17 — CLI\nImplement the contract.\n"})
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
    runner = CliRunner()
    lexical = runner.invoke(
        app,
        [
            "pack",
            "Implement CP-17",
            "--root",
            str(tmp_path),
            "--no-embeddings",
            "--json",
        ],
    )
    assert lexical.exit_code == 0, lexical.output
    lexical_data = json.loads(lexical.stdout)
    assert lexical_data["retrieval_metadata"]["retrieval_mode"] == "LEXICAL_ONLY"
    assert lexical_data["retrieval_metadata"]["active_channels"] == [
        "structural",
        "lexical",
    ]

    strict = runner.invoke(
        app,
        [
            "pack",
            "Implement CP-17",
            "--root",
            str(tmp_path),
            "--no-embeddings",
            "--strict-agent",
            "--require-semantic",
            "--json",
        ],
    )
    assert strict.exit_code == 2
    strict_data = json.loads(strict.stdout)
    assert strict_data["code"] == "SEMANTIC_RETRIEVAL_UNAVAILABLE"
    assert strict_data["reason"] == "DISABLED"


def test_actual_stdio_mcp_strict_schema_error_and_lexical_metadata(tmp_path: Path) -> None:
    _configure(tmp_path, {"SPEC.md": "# CP-17 — MCP\nImplement the contract.\n"})
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()

    async def exercise() -> None:
        parameters = StdioServerParameters(
            command=sys.executable,
            args=["-m", "ctx.cli", "mcp", "--root", str(tmp_path), "--no-embeddings"],
        )
        async with stdio_client(parameters) as streams:  # noqa: SIM117
            async with ClientSession(*streams) as session:
                await session.initialize()
                tools = await session.list_tools()
                tool = next(item for item in tools.tools if item.name == "get_context_pack")
                assert "strict_agent" in tool.input_schema["properties"]
                assert "require_semantic" in tool.input_schema["properties"]

                lexical = await session.call_tool(
                    "get_context_pack",
                    {"task": "Implement CP-17", "token_budget": 5_000},
                )
                assert not lexical.is_error
                assert lexical.structured_content is not None
                assert (
                    lexical.structured_content["retrieval_metadata"]["retrieval_mode"]
                    == "LEXICAL_ONLY"
                )
                assert lexical.structured_content["retrieval_metadata"]["active_channels"] == [
                    "structural",
                    "lexical",
                ]

                strict = await session.call_tool(
                    "get_context_pack",
                    {
                        "task": "Implement CP-17",
                        "token_budget": 5_000,
                        "strict_agent": True,
                        "require_semantic": True,
                    },
                )
                assert strict.is_error
                assert strict.structured_content is not None
                assert strict.structured_content["code"] == "SEMANTIC_RETRIEVAL_UNAVAILABLE"
                assert strict.structured_content["reason"] == "DISABLED"

    asyncio.run(asyncio.wait_for(exercise(), timeout=30))
