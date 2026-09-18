from __future__ import annotations

import hashlib
import json
import socket
from pathlib import Path

import pytest

from ctx.checkpoints import checkpoint_artifact_path
from ctx.config import add_document_config, initialize_workspace
from ctx.models import Authority, CompletenessStatus
from ctx.service import ContextEngine, StaleIndexError
from tests.source_fixtures import read_exact_source, write_exact_source


def _large_spec() -> str:
    operations = [
        "## Unified operational requirements",
        "DISTANT_ALPHA requires a signed request before dispatch.",
    ]
    for index in range(700):
        operations.append(
            f"Operational ledger paragraph {index}: preserve deterministic local state, "
            "source provenance, rollback boundaries, and exact audit records."
        )
        if index == 350:
            operations.extend(
                [
                    "```python",
                    "def verify_generation(expected: int, actual: int) -> bool:",
                    "    return expected == actual  # 検証 🙂",
                    "```",
                    "| generation | state | action |",
                    "| --- | --- | --- |",
                    "| current | clean | retain |",
                    "| stale | blocked | rebuild |",
                ]
            )
    operations.append("DISTANT_OMEGA requires durable receipt verification after commit.")
    return (
        "# CP-17 — Primary release hardening\n"
        "## Goal\nShip exact local evidence without silent omission.\n"
        "## Dependencies\nCP-2 and RunManifest\n"
        "## Architecture\nUse the local generation ledger and immutable source ranges.\n"
        "## Security\nNever execute Markdown, HTML, links, or fenced code.\n"
        "## Acceptance\nBoth distant requirements and cross-document dependencies are present.\n"
        "## Verify\nRun `pytest -q` and compare every range SHA-256.\n"
        + "\n".join(operations)
        + "\n# Unicode／Punctuation (Résumé) 🙂\nCanonical heading evidence.\n"
    )


def _write_corpus(root: Path) -> None:
    documents = {
        "spec.md": _large_spec(),
        "architecture.md": (
            "# CP-2 — Foundation\nProvide the cross-document dependency contract.\n"
            "# RunManifest\nInterface model RunManifest stores generation and source hashes.\n"
            "# CP-17 — Architecture duplicate\nDocument-scoped architecture checkpoint.\n"
        ),
        "security.md": (
            "# CP-17 — Security duplicate\nDocument-scoped security checkpoint.\n"
            "# Security controls\nHostile Markdown stays inert and runtime remains offline.\n"
        ),
    }
    initialize_workspace(root)
    priorities = {"spec.md": 10, "architecture.md": 5, "security.md": 5}
    for path, text in documents.items():
        write_exact_source(root / path, text)
        add_document_config(root, path, Authority.NORMATIVE, priorities[path])


def test_realistic_pre_use_large_corpus_release_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_corpus(tmp_path)

    def offline(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("normal CTX runtime attempted network access")

    monkeypatch.setattr(socket, "create_connection", offline)
    monkeypatch.setattr(socket.socket, "connect", offline)
    with ContextEngine(tmp_path) as engine:
        first = engine.sync_workspace()
        assert first.documents_added == 3
        assert engine.sync_workspace().index_generation == first.index_generation

        records = {record.path: record.id for record in engine.store.list_documents()}
        for path, document_id in records.items():
            artifact = checkpoint_artifact_path(tmp_path, document_id, "CP-17")
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text(
                json.dumps(
                    {
                        "provenance": {"document_id": document_id},
                        "owner": path,
                    }
                ),
                encoding="utf-8",
            )
        artifact_sync = engine.sync_workspace()
        assert artifact_sync.index_generation == first.index_generation + 1
        for path in records:
            checkpoint = engine.get_checkpoint("CP-17", document=path)
            assert checkpoint.metadata.generated_artifact is not None
            assert checkpoint.metadata.generated_artifact.data["owner"] == path

        distant = engine.search_lexical("DISTANT_ALPHA DISTANT_OMEGA", limit=8)
        spans = [
            hit
            for hit in distant
            if hit.source.provenance.heading_path[-1] == "Unified operational requirements"
        ]
        assert len(spans) == 2
        assert len({hit.source.provenance.range_sha256 for hit in spans}) == 2
        spans.sort(key=lambda hit: hit.source.provenance.start_offset)
        assert spans[0].source.provenance.end_offset <= spans[1].source.provenance.start_offset

        partial = engine.get_checkpoint_context(
            "CP-17", document="spec.md", token_budget=10_000
        ).context_pack
        assert partial.completeness_status is CompletenessStatus.PARTIAL
        assert partial.omitted_required_evidence
        assert all(
            item.document_id and item.section_id for item in partial.omitted_required_evidence
        )

        complete = engine.get_checkpoint_context(
            "CP-17",
            document="spec.md",
            token_budget=15_000,
            allow_required_budget_expansion=True,
        ).context_pack
        assert complete.completeness_status is CompletenessStatus.COMPLETE
        assert complete.budget_expanded
        assert not complete.omitted_required_evidence
        assert any(
            item.source.provenance.document_path == "architecture.md"
            and item.category == "explicit_dependency"
            for item in complete.items
        )
        assert {item.index_generation for item in complete.items} == {complete.index_generation}
        for item in complete.items:
            provenance = item.source.provenance
            source = read_exact_source(tmp_path / provenance.document_path)
            assert source[provenance.start_offset : provenance.end_offset] == item.source.text
            assert hashlib.sha256(item.source.text.encode()).hexdigest() == provenance.range_sha256

        canonical = engine.search_exact("unicode punctuation résumé", limit=2)
        assert canonical[0].source.provenance.heading_path[-1] == "Unicode／Punctuation (Résumé) 🙂"

        write_exact_source(tmp_path / "spec.md", _large_spec() + "\n<!-- stale -->\n")
        with pytest.raises(StaleIndexError):
            engine.search("DISTANT_ALPHA", limit=2)
        changed = engine.sync_workspace()
        assert changed.index_generation == artifact_sync.index_generation + 1
        refreshed = engine.search("DISTANT_ALPHA", limit=2)
        assert refreshed and all(
            hit.index_generation == changed.index_generation for hit in refreshed
        )
