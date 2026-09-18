from pathlib import Path

import pytest

from ctx.config import add_document_config, initialize_workspace
from ctx.context_pack import PrimaryRequirementTooLarge
from ctx.models import Authority
from ctx.service import ContextEngine


def test_pack_prioritizes_requirement_dependencies_models_and_acceptance(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "spec.md").write_text(
        """# RunManifest
Interface model RunManifest records trial seeds.
# CP-2 — Setup
Create fixtures for RunManifest.
# CP-14 — Repeated trials and reproducibility
Dependencies: CP-2, RunManifest
Goal: repeated trials produce deterministic evidence.
# Security constraints
Never execute Markdown or embedded scripts.
# Tests and Verify
Acceptance: run `pytest tests/test_trials.py`. Verify: `ctx status`.
""",
        encoding="utf-8",
    )
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        pack = engine.get_context_pack("Implement CP-14 using RunManifest", 5_000)
        headings = [item.source.provenance.heading_path[-1] for item in pack.items]
        categories = {item.category for item in pack.items}
        assert headings[0].startswith("CP-14")
        assert any(heading.startswith("CP-2") for heading in headings)
        assert "RunManifest" in headings
        reasons = " ".join(item.reason for item in pack.items)
        assert "declared dependency" in reasons
        assert "referenced interface/model" in reasons
        assert "security constraint" in reasons
        assert "acceptance_or_verify" in categories
        assert pack.estimated_tokens <= pack.token_budget
        assert all(item.source.provenance.document_sha256 for item in pack.items)
        assert all(item.range_sha256 for item in pack.items)
        assert pack.token_count_method == "ctx/generic-utf8-div3:1"
        assert pack.serialized_estimated_tokens <= pack.token_budget
        assert (
            pack.content_tokens + pack.metadata_tokens + pack.budget_safety_margin
            >= pack.estimated_tokens
        )


def test_pack_truncation_uses_ast_safe_ranges_and_reports_omissions(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    fence = "```python\n" + "print('large')\n" * 300 + "```\n"
    table = "| h | v |\n|---|---|\n" + "| a | b |\n" * 200
    (tmp_path / "large.md").write_text(
        "# CP-14 — Bounded\nShort exact requirement.\n" + fence + "# Table\n" + table,
        encoding="utf-8",
    )
    add_document_config(tmp_path, "large.md", Authority.NORMATIVE)
    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        with pytest.raises(PrimaryRequirementTooLarge):
            engine.get_context_pack("CP-14 Table", 1_500)
        pack = engine.get_context_pack("CP-14 Table", 1_500, allow_required_budget_expansion=True)
        assert pack.budget_expanded
        assert pack.items
        direct = pack.items[0]
        assert direct.source.text.startswith("# CP-14")
        assert direct.source.text.count("```") == 2
        assert direct.source.source_type == "section"
        assert (
            direct.source.provenance.end_line
            == engine.get_section(direct.source.provenance.section_id).provenance.end_line
        )


def test_possible_conflict_is_conservative_and_source_labeled(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "norm.md").write_text(
        "# NetworkPolicy rule\nNetworkPolicy must allow ingress.\n", encoding="utf-8"
    )
    (tmp_path / "history.md").write_text(
        "# Old NetworkPolicy\nNetworkPolicy must not allow ingress.\n", encoding="utf-8"
    )
    add_document_config(tmp_path, "norm.md", Authority.NORMATIVE)
    add_document_config(tmp_path, "history.md", Authority.HISTORICAL)
    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        pack = engine.get_context_pack("NetworkPolicy ingress", 3_000)
        assert pack.possible_conflicts
        conflict = pack.possible_conflicts[0]
        assert conflict.label == "POSSIBLE_CONFLICT"
        assert conflict.identifier == "NetworkPolicy"
        assert {item.authority for item in conflict.sources} == {
            Authority.NORMATIVE,
            Authority.HISTORICAL,
        }
        assert all(not hasattr(item, "text") for item in conflict.sources)
