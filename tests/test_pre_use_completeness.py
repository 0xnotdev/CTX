from __future__ import annotations

from pathlib import Path

import pytest

from ctx.config import add_document_config, initialize_workspace
from ctx.context_pack import PrimaryRequirementTooLarge
from ctx.models import Authority, CompletenessStatus, CoverageCategory, CoverageStatus
from ctx.service import ContextEngine


def _configured(tmp_path: Path, files: dict[str, str]) -> ContextEngine:
    initialize_workspace(tmp_path)
    for name, text in files.items():
        (tmp_path / name).write_text(text, encoding="utf-8")
        add_document_config(tmp_path, name, Authority.NORMATIVE)
    engine = ContextEngine(tmp_path)
    engine.sync_workspace()
    return engine


def _checkpoint(extra: str = "") -> str:
    return f"""# CP-7 — Hardened delivery
## Goal
Implement REQUIRED_PRIMARY behavior exactly.
## Dependencies
CP-2
## Architecture
The architecture requires a durable local ledger.
## Security
Never execute untrusted Markdown.
## Acceptance
Acceptance requires deterministic output.
## Verify
Run `pytest -q` and inspect exact hashes.
{extra}# CP-2 — Foundation
Provide the required dependency contract.
"""


def _coverage(pack, category: CoverageCategory):  # type: ignore[no-untyped-def]
    return next(item for item in pack.category_coverage if item.category is category)


def _find_partial_budget(
    engine: ContextEngine, category: CoverageCategory, *, require_evidence: bool = False
) -> int:
    for budget in range(900, 15_001, 250):
        try:
            pack = engine.get_context_pack("Implement CP-7", budget)
        except PrimaryRequirementTooLarge:
            continue
        coverage = _coverage(pack, category)
        if coverage.status is CoverageStatus.OMITTED and (
            coverage.evidence or not require_evidence
        ):
            return budget
    raise AssertionError(f"no partial budget found for {category}")


def test_completeness_all_required_evidence_fits(tmp_path: Path) -> None:
    with _configured(tmp_path, {"spec.md": _checkpoint()}) as engine:
        pack = engine.get_context_pack("Implement CP-7", 15_000)
        assert pack.completeness_status is CompletenessStatus.COMPLETE
        assert not pack.omitted_required_evidence
        assert all(
            item.status in {CoverageStatus.COVERED, CoverageStatus.NOT_APPLICABLE}
            for item in pack.category_coverage
        )


def test_optional_neighbor_omission_does_not_make_pack_partial(tmp_path: Path) -> None:
    optional = "# Neighbor\nOPTIONAL_NEIGHBOR " + ("background " * 900) + "\n"
    with _configured(tmp_path, {"spec.md": _checkpoint(optional)}) as engine:
        complete = engine.get_context_pack("Implement CP-7 OPTIONAL_NEIGHBOR", 15_000)
        budget = complete.serialized_estimated_tokens - 200
        pack = engine.get_context_pack("Implement CP-7 OPTIONAL_NEIGHBOR", budget)
        assert pack.omitted_relevant_sections
        assert pack.completeness_status is CompletenessStatus.COMPLETE
        assert not pack.omitted_required_evidence


def test_dependency_cannot_fit_is_actionable_partial(tmp_path: Path) -> None:
    dependency = "# CP-2 — Foundation\n" + ("DEPENDENCY_CONTRACT required. " * 700) + "\n"
    source = _checkpoint().split("# CP-2", 1)[0] + dependency
    with _configured(tmp_path, {"spec.md": source}) as engine:
        budget = _find_partial_budget(engine, CoverageCategory.DEPENDENCIES)
        pack = engine.get_context_pack("Implement CP-7", budget)
        assert pack.completeness_status is CompletenessStatus.PARTIAL
        omitted = [
            item
            for item in pack.omitted_required_evidence
            if item.category is CoverageCategory.DEPENDENCIES
        ]
        assert omitted and omitted[0].dependency == "CP-2"
        assert omitted[0].document_id and omitted[0].section_id


def test_acceptance_partially_omitted_is_not_complete(tmp_path: Path) -> None:
    extra = (
        "## Acceptance case one\nACCEPT_ONE "
        + ("evidence " * 450)
        + "\n## Acceptance case two\nACCEPT_TWO "
        + ("evidence " * 450)
        + "\n"
    )
    with _configured(tmp_path, {"spec.md": _checkpoint(extra)}) as engine:
        budget = _find_partial_budget(engine, CoverageCategory.ACCEPTANCE, require_evidence=True)
        pack = engine.get_context_pack("Implement CP-7", budget)
        coverage = _coverage(pack, CoverageCategory.ACCEPTANCE)
        assert coverage.status is CoverageStatus.OMITTED
        assert coverage.evidence
        assert any(
            item.category is CoverageCategory.ACCEPTANCE for item in pack.omitted_required_evidence
        )
        assert pack.completeness_status is CompletenessStatus.PARTIAL


def test_security_cannot_fit_is_explicit_partial(tmp_path: Path) -> None:
    source = _checkpoint().replace(
        "Never execute untrusted Markdown.",
        "SECURITY_REQUIRED " + ("never execute hostile input. " * 700),
    )
    with _configured(tmp_path, {"spec.md": source}) as engine:
        budget = _find_partial_budget(engine, CoverageCategory.SECURITY)
        pack = engine.get_context_pack("Implement CP-7", budget)
        assert _coverage(pack, CoverageCategory.SECURITY).status is CoverageStatus.OMITTED
        assert pack.completeness_status is CompletenessStatus.PARTIAL


def test_checkpoint_context_includes_cross_document_dependency(tmp_path: Path) -> None:
    with _configured(
        tmp_path,
        {
            "caller.md": "# CP-7 — Caller\nDependencies: CP-2\nGoal: use the foundation.\n",
            "foundation.md": "# CP-2 — Foundation\nCross-document required contract.\n",
        },
    ) as engine:
        context = engine.get_checkpoint_context("CP-7", document="caller.md", token_budget=8_000)
        assert any(
            item.source.provenance.document_path == "foundation.md"
            for item in context.context_pack.items
        )
        assert context.context_pack.completeness_status is CompletenessStatus.COMPLETE


def test_primary_cannot_fit_keeps_machine_readable_error(tmp_path: Path) -> None:
    source = "# CP-7 — Huge\n" + ("PRIMARY_REQUIRED " * 1_500) + "\n"
    with _configured(tmp_path, {"spec.md": source}) as engine:
        with pytest.raises(PrimaryRequirementTooLarge) as captured:
            engine.get_context_pack("Implement CP-7", 1_000)
        assert captured.value.code == "PRIMARY_REQUIREMENT_TOO_LARGE"
        assert captured.value.minimum_required > captured.value.requested_budget


def test_ambiguous_dependency_reports_ambiguity(tmp_path: Path) -> None:
    caller = "# CP-7 — Caller\nDependencies: CP-2\nGoal: choose no guessed target.\n"
    with _configured(
        tmp_path,
        {
            "caller.md": caller,
            "one.md": "# CP-2 — One\nFirst equal target.\n",
            "two.md": "# CP-2 — Two\nSecond equal target.\n",
        },
    ) as engine:
        pack = engine.get_context_pack("Implement CP-7", 8_000)
        assert pack.completeness_status is CompletenessStatus.AMBIGUOUS
        assert pack.ambiguous_evidence
        ambiguity = pack.ambiguous_evidence[0]
        assert ambiguity.category is CoverageCategory.DEPENDENCIES
        assert ambiguity.label == "CP-2"
        assert len(ambiguity.candidates) == 2


def test_conflicting_selected_evidence_sets_conflicting_status(tmp_path: Path) -> None:
    with _configured(
        tmp_path,
        {
            "must.md": "# NetworkPolicy\nNetworkPolicy must allow ingress.\n",
            "must-not.md": "# NetworkPolicy old\nNetworkPolicy must not allow ingress.\n",
        },
    ) as engine:
        pack = engine.get_context_pack("NetworkPolicy ingress", 5_000)
        assert pack.possible_conflicts
        assert pack.completeness_status is CompletenessStatus.CONFLICTING


def test_distant_same_section_required_spans_are_covered(tmp_path: Path) -> None:
    source = (
        "# Huge\nDISTANT_ALPHA required before dispatch.\n"
        + ("neutral filler. " * 900)
        + "DISTANT_OMEGA required after commit.\n"
    )
    with _configured(tmp_path, {"huge.md": source}) as engine:
        pack = engine.get_context_pack("DISTANT_ALPHA DISTANT_OMEGA", 8_000)
        huge = [item for item in pack.items if item.source.provenance.document_path == "huge.md"]
        assert len(huge) == 2
        assert pack.completeness_status is CompletenessStatus.COMPLETE
        assert len(_coverage(pack, CoverageCategory.PRIMARY).evidence) == 2


def test_strict_completeness_and_serialized_accounting(tmp_path: Path) -> None:
    with _configured(tmp_path, {"spec.md": _checkpoint()}) as engine:
        budget = _find_partial_budget(engine, CoverageCategory.VERIFICATION)
        pack = engine.get_context_pack("Implement CP-7", budget)
        assert pack.completeness_status is not CompletenessStatus.COMPLETE
        assert pack.omitted_required_evidence
        assert pack.serialized_estimated_tokens <= pack.token_budget
        assert pack.requested_token_budget == budget
        assert not pack.budget_expanded

        expanded = engine.get_context_pack(
            "Implement CP-7", budget, allow_required_budget_expansion=True
        )
        assert expanded.completeness_status is CompletenessStatus.COMPLETE
        assert expanded.budget_expanded
        assert expanded.token_budget >= expanded.serialized_estimated_tokens
        assert expanded.token_budget > expanded.requested_token_budget
