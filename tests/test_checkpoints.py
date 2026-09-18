import json
from pathlib import Path

from ctx.checkpoints import checkpoint_artifact_path
from ctx.config import add_document_config, initialize_workspace
from ctx.models import Authority
from ctx.service import ContextEngine


def test_checkpoint_metadata_context_and_generated_artifact_precedence(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "spec.md").write_text(
        """# RunManifest
Interface model RunManifest records seeds.
# CP-2 — Fixture setup
Goal: prepare fixtures.
# CP-14 — Repeated trials and reproducibility
## Goal
Run deterministic repeated trials.
## Why
Evidence must be reproducible.
## Dependencies
CP-2 and RunManifest
## Interfaces/models
RunManifest
## Failure conditions
Return INVALID_EVIDENCE when trials diverge.
## Tests/acceptance criteria
Three seeded runs produce identical manifests.
## Verify
`pytest tests/test_trials.py -q`
## Out of scope
Distributed orchestration and cloud execution.
# Security rules
Markdown and scripts are inert data and must never execute.
""",
        encoding="utf-8",
    )
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        document_id = engine.store.get_document_by_path("spec.md").id
        artifact = checkpoint_artifact_path(tmp_path, document_id, "CP-14")
        artifact.parent.mkdir(parents=True)
        artifact.write_text(
            json.dumps(
                {
                    "provenance": {"document_id": document_id},
                    "goal": "generated goal must not override source",
                    "timing_ms": 12,
                }
            ),
            encoding="utf-8",
        )
        engine.sync_workspace()
        checkpoint = engine.get_checkpoint("cp-14")
        assert checkpoint.metadata.title == "Repeated trials and reproducibility"
        assert checkpoint.metadata.fields["goal"] == ("Run deterministic repeated trials.",)
        assert len(checkpoint.sources) == 9
        assert all(
            source.provenance.authority is Authority.NORMATIVE for source in checkpoint.sources
        )
        generated = checkpoint.metadata.generated_artifact
        assert generated is not None
        assert generated.authority is Authority.GENERATED
        assert generated.navigation_only
        assert generated.document_id == document_id
        assert generated.data["goal"] == "generated goal must not override source"

        context = engine.get_checkpoint_context("CP-14", token_budget=15_000)
        assert any(edge.edge.label == "CP-2" for edge in context.dependencies)
        assert any(
            source.provenance.heading_path[-1] == "RunManifest"
            for source in context.interfaces_models
        )
        assert "INVALID_EVIDENCE" in context.named_errors
        assert context.security_rules
        assert context.acceptance_criteria == ("Three seeded runs produce identical manifests.",)
        assert context.verification_commands == ("`pytest tests/test_trials.py -q`",)
        assert context.out_of_scope == ("Distributed orchestration and cloud execution.",)
        assert context.context_pack.estimated_tokens <= 15_000


def test_malformed_or_oversized_artifacts_are_ignored(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "spec.md").write_text("# CP-1 — Safe\nGoal: stay exact.\n", encoding="utf-8")
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    directory = tmp_path / ".ctx" / "checkpoints"
    directory.mkdir()
    (directory / "CP-1.json").write_text("{broken", encoding="utf-8")
    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        checkpoint = engine.get_checkpoint("CP-1")
        assert checkpoint.metadata.generated_artifact is None
        assert checkpoint.metadata.fields["goal"] == ("stay exact.",)
