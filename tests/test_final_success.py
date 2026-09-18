from pathlib import Path

from ctx.config import add_document_config, initialize_workspace
from ctx.embeddings import HashEmbedding
from ctx.models import Authority
from ctx.service import ContextEngine


def test_final_cp14_context_and_one_section_incremental_success(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    spec = """# Global constraints
Original Markdown is authoritative; generated summaries are navigation-only.
# RunManifest
Interface model RunManifest records seed, trial ID, and evidence hash.
# CP-2 — Fixture setup
Prepare deterministic fixtures and a RunManifest.
# CP-14 — Repeated trials and reproducibility
## Goal
Run three deterministic repeated trials.
## Dependencies
CP-2 and RunManifest
## Interfaces/models
RunManifest
## Failure conditions
Return INVALID_EVIDENCE when evidence hashes differ.
## Security constraints
Never execute source Markdown, HTML, links, or scripts.
## Tests/acceptance criteria
Three seeded runs produce byte-identical RunManifest records.
## Verify
`pytest tests/test_reproducibility.py -q`
## Out of scope
Cloud execution and distributed orchestration.
"""
    architecture = """# Design
SQLite and local stdio MCP share ContextEngine.
# Storage
Sections are authoritative and chunks are search-only.
"""
    research = "# Research\n" + "Generated trial observation.\n" * 500
    (tmp_path / "spec.md").write_text(spec, encoding="utf-8")
    (tmp_path / "architecture.md").write_text(architecture, encoding="utf-8")
    (tmp_path / "research.md").write_text(research, encoding="utf-8")
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE, 10)
    add_document_config(tmp_path, "architecture.md", Authority.NORMATIVE, 5)
    add_document_config(tmp_path, "research.md", Authority.REFERENCE)

    embedder = HashEmbedding(64)
    with ContextEngine(tmp_path, embedder=embedder) as engine:
        indexed = engine.index_workspace()
        assert indexed.documents_added == 3
        pack = engine.get_context_pack(
            "Implement CP-14 with RunManifest and INVALID_EVIDENCE", 15_000
        )
        assert pack.estimated_tokens <= 15_000
        headings = [item.source.provenance.heading_path[-1] for item in pack.items]
        assert any(heading.startswith("CP-14") for heading in headings)
        assert any(heading.startswith("CP-2") for heading in headings)
        assert "RunManifest" in headings
        assert "Tests/acceptance criteria" in headings
        assert "Security constraints" in headings
        assert all(item.source.provenance.document_sha256 for item in pack.items)
        assert all(item.source.provenance.section_sha256 for item in pack.items)
        assert all(item.source.provenance.heading_path for item in pack.items)
        assert all(
            item.source.provenance.start_line <= item.source.provenance.end_line
            for item in pack.items
        )
        assert sum(len(item.source.text) for item in pack.items) < sum(
            len(path.read_text(encoding="utf-8"))
            for path in (
                tmp_path / "spec.md",
                tmp_path / "architecture.md",
                tmp_path / "research.md",
            )
        )

        checkpoint = engine.get_checkpoint_context("CP-14", token_budget=7_000)
        assert checkpoint.checkpoint.sources[0].text.startswith("# CP-14")
        assert any(edge.edge.label == "CP-2" for edge in checkpoint.dependencies)
        assert any(
            item.provenance.heading_path[-1] == "RunManifest"
            for item in checkpoint.interfaces_models
        )
        assert "INVALID_EVIDENCE" in checkpoint.named_errors
        assert checkpoint.acceptance_criteria
        assert checkpoint.verification_commands
        assert checkpoint.out_of_scope

        before = {
            row["chunk_id"]: bytes(row["vector"])
            for row in engine.store.connection.execute(
                "SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id"
            )
        }
        changed_architecture = architecture.replace(
            "SQLite and local stdio MCP share ContextEngine.",
            "SQLite, CLI, and local stdio MCP share ContextEngine.",
        )
        (tmp_path / "architecture.md").write_text(changed_architecture, encoding="utf-8")
        synced = engine.sync_workspace()
        assert synced.documents_changed == 1
        assert synced.documents_unchanged == 2
        assert synced.sections_changed == 1
        assert synced.sections_unchanged == 1
        assert synced.embeddings_created == 1
        assert synced.embeddings_retained == 1
        after = {
            row["chunk_id"]: bytes(row["vector"])
            for row in engine.store.connection.execute(
                "SELECT chunk_id, vector FROM embeddings ORDER BY chunk_id"
            )
        }
        retained = set(before) & set(after)
        assert retained
        assert all(before[chunk_id] == after[chunk_id] for chunk_id in retained)
        assert len(set(before) - set(after)) == 1
        assert len(set(after) - set(before)) == 1
