from pathlib import Path
from typing import Any

import numpy as np
import pytest

from ctx.config import add_document_config, initialize_workspace
from ctx.context_pack import PrimaryRequirementTooLarge
from ctx.models import Authority, CompletenessStatus
from ctx.service import ContextEngine


class _FlatEmbedding:
    dimensions = 4
    identity = "ctx/test-conflict-scope:1:4"
    metadata = {
        "provider": "ctx",
        "model_name": "test-conflict-scope",
        "revision": "1",
        "artifact_sha256": "deterministic-test-only",
        "runtime_version": "1",
    }

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def embed_documents(self, texts: list[str]) -> np.ndarray[Any, np.dtype[np.float32]]:
        return np.ones((len(texts), self.dimensions), dtype=np.float32)

    def embed_query(self, query: str) -> np.ndarray[Any, np.dtype[np.float32]]:
        return np.ones(self.dimensions, dtype=np.float32)


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
        assert all(item.start_line == item.end_line for item in conflict.sources)


def test_conflict_detection_ignores_broad_protocol_mentions_with_unrelated_modal(
    tmp_path: Path,
) -> None:
    symbols = (
        "AlphaExecutor",
        "BetaSink",
        "GammaAdapter",
        "DeltaPolicyEngine",
        "EpsilonOperator",
        "ZetaCoverageTracker",
        "EtaRegressionStore",
        "ThetaResourceState",
    )
    initialize_workspace(tmp_path)
    protocol_types = "\n".join(f"class {symbol}(Protocol): ..." for symbol in symbols)
    (tmp_path / "protocol.md").write_text(
        "# Protocol catalogue\n"
        "The support contracts below are normative. Implementations MUST NOT replace "
        "support payloads with untyped dictionaries.\n"
        "```python\n"
        f"{protocol_types}\n"
        "```\n",
        encoding="utf-8",
    )
    add_document_config(tmp_path, "protocol.md", Authority.NORMATIVE)
    for index, symbol in enumerate(symbols, start=1):
        path = f"component-{index}.md"
        (tmp_path / path).write_text(
            f"# {symbol} delivery\n{symbol} must process its assigned request.\n",
            encoding="utf-8",
        )
        add_document_config(tmp_path, path, Authority.NORMATIVE)

    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        pack = engine.get_context_pack(" ".join(symbols), 30_000)
        selected = "\n".join(item.source.text for item in pack.items)
        assert all(symbol in selected for symbol in symbols)
        assert pack.completeness_status is CompletenessStatus.COMPLETE
        assert not pack.possible_conflicts


def test_conflict_detection_attributes_enabled_disabled_to_governed_object(
    tmp_path: Path,
) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "catalog.md").write_text(
        "# SharedHarness catalog\n"
        "SharedHarness integration is in scope. Every enabled operator has positive and "
        "negative applicability coverage.\n",
        encoding="utf-8",
    )
    (tmp_path / "verifier.md").write_text(
        "# SharedHarness verifier\n"
        "SharedHarness integration is in scope. Verifier fixtures produce survivors when "
        "checks disabled.\n",
        encoding="utf-8",
    )
    add_document_config(tmp_path, "catalog.md", Authority.NORMATIVE)
    add_document_config(tmp_path, "verifier.md", Authority.NORMATIVE)

    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        pack = engine.get_context_pack("SharedHarness enabled disabled", 8_000)
        selected = "\n".join(item.source.text for item in pack.items)
        assert "enabled operator" in selected
        assert "checks disabled" in selected
        assert pack.completeness_status is CompletenessStatus.COMPLETE
        assert not pack.possible_conflicts


@pytest.mark.parametrize(
    ("positive", "negative", "identifier", "task"),
    (
        (
            "# NetworkPolicy rule\nNetworkPolicy must allow ingress.\n",
            "# Old NetworkPolicy rule\nNetworkPolicy must not allow ingress.\n",
            "NetworkPolicy",
            "NetworkPolicy ingress",
        ),
        (
            "# FeatureSwitch state\nFeatureSwitch is enabled for default routing.\n",
            "# Old FeatureSwitch state\nFeatureSwitch is disabled for default routing.\n",
            "FeatureSwitch",
            "FeatureSwitch default routing",
        ),
        (
            "# DataExport permission\nDataExport is allowed for guest access.\n",
            "# Old DataExport permission\nDataExport is prohibited for guest access.\n",
            "DataExport",
            "DataExport guest access",
        ),
        (
            "# AuditTrail requirement\nAuditTrail is required for release.\n",
            "# Old AuditTrail requirement\nAuditTrail is forbidden for release.\n",
            "AuditTrail",
            "AuditTrail release",
        ),
    ),
)
def test_strict_conflict_detection_still_blocks_genuine_contradictions(
    tmp_path: Path, positive: str, negative: str, identifier: str, task: str
) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "current.md").write_text(positive, encoding="utf-8")
    (tmp_path / "conflict.md").write_text(negative, encoding="utf-8")
    add_document_config(tmp_path, "current.md", Authority.NORMATIVE)
    add_document_config(tmp_path, "conflict.md", Authority.HISTORICAL)

    with ContextEngine(tmp_path, embedder=_FlatEmbedding()) as engine:
        engine.sync_workspace()
        pack = engine.get_context_pack(task, 8_000, strict_agent=True, require_semantic=True)
        assert pack.completeness_status is CompletenessStatus.CONFLICTING
        assert any(conflict.identifier == identifier for conflict in pack.possible_conflicts)
