from pathlib import Path

from ctx.config import add_document_config, initialize_workspace
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
        pack = engine.get_context_pack("Implement CP-14 using RunManifest", 4_000)
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
        assert pack.token_count_method == "utf8_bytes_div4_ceiling"


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
        pack = engine.get_context_pack("CP-14 Table", 350)
        assert pack.estimated_tokens <= 350
        assert pack.items
        direct = pack.items[0]
        assert direct.source.text.startswith("# CP-14")
        assert direct.source.text.count("```") in {0, 2}
        if "| h | v |" in direct.source.text:
            assert "| a | b |" in direct.source.text
        assert (
            direct.source.provenance.end_line
            <= engine.get_section(direct.source.provenance.section_id).provenance.end_line
        )
        assert pack.omitted_relevant_sections


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
        pack = engine.get_context_pack("NetworkPolicy ingress", 2_000)
        assert pack.possible_conflicts
        conflict = pack.possible_conflicts[0]
        assert conflict.label == "POSSIBLE_CONFLICT"
        assert conflict.identifier == "NetworkPolicy"
        assert {item.provenance.authority for item in conflict.sources} == {
            Authority.NORMATIVE,
            Authority.HISTORICAL,
        }
