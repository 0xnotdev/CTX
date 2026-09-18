"""Exact-source context planning with complete serialized-response budgeting."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ctx.heading import canonical_heading
from ctx.models import (
    AmbiguousEvidence,
    Authority,
    BudgetMethod,
    CategoryCoverage,
    CompletenessStatus,
    ContextPack,
    ContextPackItem,
    CoverageCategory,
    CoverageStatus,
    EdgeType,
    FilterSet,
    OmittedRequiredEvidence,
    PossibleConflict,
    ResolutionStatus,
    SourceItem,
)
from ctx.retrieval import classify_query

if TYPE_CHECKING:
    from ctx.service import ContextEngine


class TokenCounter(Protocol):
    """Counter contract for source text and the complete serialized agent response."""

    @property
    def identity(self) -> str: ...

    @property
    def method(self) -> BudgetMethod: ...

    @property
    def safety_margin_ratio(self) -> float: ...

    def count_text(self, text: str) -> int: ...

    def count_serialized(self, value: object) -> int: ...


class StrictByteUpperBound:
    """Tokenizer-independent upper bound: each UTF-8 byte is charged as one token."""

    identity = "ctx/utf8-byte-upper-bound:1"
    method = BudgetMethod.STRICT_BYTE_UPPER_BOUND
    safety_margin_ratio = 0.0

    def count_text(self, text: str) -> int:
        return len(text.encode("utf-8"))

    def count_serialized(self, value: object) -> int:
        text = value if isinstance(value, str) else _json(value)
        return len(text.encode("utf-8"))


class ApproximateGenericCounter:
    """Explicit generic estimate with a 15% serialized safety margin.

    This is not advertised as a tokenizer upper bound. Deployments may inject a model-specific
    counter; strict callers can select :class:`StrictByteUpperBound`.
    """

    identity = "ctx/generic-utf8-div3:1"
    method = BudgetMethod.APPROXIMATE_GENERIC
    safety_margin_ratio = 0.15

    def count_text(self, text: str) -> int:
        return max(0, math.ceil(len(text.encode("utf-8")) / 3))

    def count_serialized(self, value: object) -> int:
        text = value if isinstance(value, str) else _json(value)
        return max(0, math.ceil(len(text.encode("utf-8")) / 3))


# V0 import compatibility. Its name and behavior now state approximation honestly.
class ApproximateTokenCounter(ApproximateGenericCounter):
    @property
    def name(self) -> str:
        return self.identity

    def count(self, text: str) -> int:
        return self.count_text(text)


class ContextBudgetTooSmall(ValueError):
    code = "CONTEXT_BUDGET_TOO_SMALL"

    def __init__(
        self,
        *,
        requested_budget: int,
        minimum_required: int,
        task_token_estimate: int,
        metadata_overhead: int,
    ):
        self.requested_budget = requested_budget
        self.minimum_required = minimum_required
        self.task_token_estimate = task_token_estimate
        self.metadata_overhead = metadata_overhead
        super().__init__(
            f"{self.code}: requested={requested_budget}, minimum_required={minimum_required}, "
            f"task_token_estimate={task_token_estimate}, metadata_overhead={metadata_overhead}"
        )

    def as_dict(self) -> dict[str, int | str]:
        return {
            "code": self.code,
            "requested_budget": self.requested_budget,
            "minimum_required": self.minimum_required,
            "task_token_estimate": self.task_token_estimate,
            "metadata_overhead": self.metadata_overhead,
        }


class PrimaryRequirementTooLarge(ContextBudgetTooSmall):
    code = "PRIMARY_REQUIREMENT_TOO_LARGE"


@dataclass
class _Candidate:
    source: SourceItem
    category_rank: int
    category: str
    reason: str
    relevance: float
    confidence: float
    sequence: int
    primary: bool = False
    required: bool = False
    coverage_categories: tuple[CoverageCategory, ...] = ()
    checkpoint_id: str | None = None
    dependency: str | None = None


_CATEGORIES = {
    "direct_requirement": 0,
    "direct_reference": 1,
    "explicit_dependency": 2,
    "checkpoint_field": 3,
    "interface_or_model": 4,
    "global_constraint": 5,
    "error_or_failure": 6,
    "security_constraint": 7,
    "acceptance_or_verify": 8,
    "semantic_fallback": 9,
    "neighbor_context": 10,
}


def _json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def serialized_agent_response(pack: ContextPack) -> dict[str, object]:
    """Exact ctx MCP response shape: structured data once plus a compact inert text notice."""
    data = pack.model_dump(mode="json")
    return {
        "jsonrpc": "2.0",
        "id": 2_147_483_647,
        "result": {
            "content": [{"type": "text", "text": "ctx context pack; use structuredContent"}],
            "structuredContent": data,
            "isError": False,
        },
    }


def _identifiers(text: str) -> set[str]:
    patterns = (
        r"\bCP-\d+\b",
        r"\b(?:[A-Z][a-z0-9]+){2,}\b",
        r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b",
        r"\b[A-Za-z_][\w-]*(?:\.[\w-]+)+(?:@\d+)?\b",
    )
    return {match.group(0) for pattern in patterns for match in re.finditer(pattern, text)}


def _opposition(left: str, right: str) -> str | None:
    pairs = (
        (r"\bmust\s+not\b", r"\bmust\b(?!\s+not)"),
        (r"\benabled\b", r"\bdisabled\b"),
        (r"\ballowed\b", r"\bprohibited\b"),
        (r"\brequired\b", r"\bforbidden\b"),
    )
    for positive, negative in pairs:
        if (re.search(positive, left, re.I) and re.search(negative, right, re.I)) or (
            re.search(negative, left, re.I) and re.search(positive, right, re.I)
        ):
            return f"deterministic contradictory phrase pair: {positive} / {negative}"
    return None


def _possible_conflicts(items: Sequence[ContextPackItem]) -> tuple[PossibleConflict, ...]:
    conflicts: list[PossibleConflict] = []
    for index, left in enumerate(items):
        left_ids = _identifiers(left.source.text)
        for right in items[index + 1 :]:
            overlap = sorted(left_ids & _identifiers(right.source.text))
            reason = _opposition(left.source.text, right.source.text)
            if overlap and reason:
                conflicts.append(
                    PossibleConflict(
                        identifier=overlap[0],
                        reason=reason,
                        sources=(left.source.ref, right.source.ref),
                    )
                )
                if len(conflicts) >= 10:
                    return tuple(conflicts)
    return tuple(conflicts)


def _counter_name(counter: TokenCounter) -> str:
    return counter.identity


def _candidate_range_key(candidate: _Candidate) -> tuple[str, int, int]:
    provenance = candidate.source.provenance
    return (provenance.section_id, provenance.start_offset, provenance.end_offset)


def _coverage_state(
    candidates: Sequence[_Candidate],
    selected: dict[tuple[str, int, int], ContextPackItem],
    unresolved: Sequence[OmittedRequiredEvidence],
    ambiguous: Sequence[AmbiguousEvidence],
    conflicts: tuple[PossibleConflict, ...],
) -> tuple[
    CompletenessStatus,
    tuple[CategoryCoverage, ...],
    tuple[OmittedRequiredEvidence, ...],
]:
    omissions = list(unresolved)
    missing = [
        candidate
        for candidate in candidates
        if candidate.required and _candidate_range_key(candidate) not in selected
    ]
    for candidate in missing:
        provenance = candidate.source.provenance
        # One actionable record identifies one missing range. Category coverage below still marks
        # every applicable category (including checkpoint_descendants) without duplicating this
        # provenance-heavy record.
        category = candidate.coverage_categories[-1]
        omissions.append(
            OmittedRequiredEvidence(
                category=category,
                reason="required evidence did not fit the context-pack budget",
                document_id=provenance.document_id,
                document_path=provenance.document_path,
                section_id=provenance.section_id,
                checkpoint_id=candidate.checkpoint_id,
                dependency=candidate.dependency,
                start_line=provenance.start_line,
                end_line=provenance.end_line,
                range_sha256=provenance.range_sha256,
            )
        )

    conflict_hashes = {source.range_sha256 for conflict in conflicts for source in conflict.sources}
    coverage_records: list[CategoryCoverage] = []
    for category in CoverageCategory:
        applicable = [
            candidate for candidate in candidates if category in candidate.coverage_categories
        ]
        evidence = tuple(
            dict.fromkeys(
                selected[_candidate_range_key(candidate)].source.ref
                for candidate in applicable
                if _candidate_range_key(candidate) in selected
            )
        )
        category_missing = [
            candidate for candidate in missing if category in candidate.coverage_categories
        ]
        category_omissions = [item for item in unresolved if item.category is category]
        category_ambiguous = [item for item in ambiguous if item.category is category]
        category_conflicting = any(item.range_sha256 in conflict_hashes for item in evidence)
        required = any(candidate.required for candidate in applicable) or bool(
            category_omissions or category_ambiguous
        )
        notes: list[str] = []
        if category_ambiguous:
            status = CoverageStatus.AMBIGUOUS
            notes.extend(item.reason for item in category_ambiguous)
        elif category_missing or category_omissions:
            status = CoverageStatus.OMITTED
            notes.append("one or more required evidence ranges are absent")
        elif category_conflicting:
            status = CoverageStatus.CONFLICTING
            notes.append("selected evidence contains a compact possible-conflict record")
        elif applicable:
            status = CoverageStatus.COVERED
        else:
            status = CoverageStatus.NOT_APPLICABLE
        coverage_records.append(
            CategoryCoverage(
                category=category,
                status=status,
                required=required,
                evidence=evidence,
                omitted_count=len(category_missing) + len(category_omissions),
                notes=tuple(dict.fromkeys(notes)),
            )
        )

    if conflicts:
        top = CompletenessStatus.CONFLICTING
    elif ambiguous:
        top = CompletenessStatus.AMBIGUOUS
    elif omissions:
        top = CompletenessStatus.PARTIAL
    elif any(item.status is CoverageStatus.COVERED for item in coverage_records):
        top = CompletenessStatus.COMPLETE
    else:
        top = CompletenessStatus.NOT_APPLICABLE
    return top, tuple(coverage_records), tuple(omissions)


def _pack(
    *,
    task: str,
    requested_budget: int,
    budget: int,
    budget_expanded: bool,
    counter: TokenCounter,
    items: Sequence[ContextPackItem],
    completeness_status: CompletenessStatus,
    category_coverage: Sequence[CategoryCoverage],
    omitted_required: Sequence[OmittedRequiredEvidence],
    ambiguous: Sequence[AmbiguousEvidence],
    omitted: Sequence[str],
    conflicts: tuple[PossibleConflict, ...],
    generation: int,
    retrieval_metadata: dict[str, str | int | bool | list[str]],
) -> ContextPack:
    content_tokens = sum(counter.count_text(item.source.text) for item in items)
    # Iterate because count fields and safety-margin digits are themselves serialized metadata.
    serialized = 0
    safety = 0
    metadata_tokens = 0
    result: ContextPack | None = None
    for _ in range(8):
        result = ContextPack(
            task=task,
            requested_token_budget=requested_budget,
            token_budget=budget,
            budget_expanded=budget_expanded,
            budget_method=counter.method,
            budget_counter_identity=counter.identity,
            budget_safety_margin=safety,
            content_tokens=content_tokens,
            metadata_tokens=metadata_tokens,
            serialized_estimated_tokens=serialized + safety,
            estimated_tokens=serialized + safety,
            token_count_method=counter.identity,
            items=tuple(items),
            completeness_status=completeness_status,
            category_coverage=tuple(category_coverage),
            omitted_required_evidence=tuple(omitted_required),
            ambiguous_evidence=tuple(ambiguous),
            omitted_relevant_sections=tuple(dict.fromkeys(omitted)),
            possible_conflicts=conflicts,
            index_generation=generation,
            retrieval_metadata=retrieval_metadata,
        )
        # stdio MCP is one compact JSON frame terminated by LF.
        raw = counter.count_serialized(_json(serialized_agent_response(result)) + "\n")
        next_safety = math.ceil(raw * counter.safety_margin_ratio)
        next_metadata = max(0, raw - content_tokens)
        if raw == serialized and next_safety == safety and next_metadata == metadata_tokens:
            break
        serialized, safety, metadata_tokens = raw, next_safety, next_metadata
    assert result is not None
    # One final object carries the converged values.
    return result.model_copy(
        update={
            "budget_safety_margin": safety,
            "metadata_tokens": metadata_tokens,
            "serialized_estimated_tokens": serialized + safety,
            "estimated_tokens": serialized + safety,
        }
    )


def _empty_coverage() -> tuple[CategoryCoverage, ...]:
    return tuple(
        CategoryCoverage(
            category=category,
            status=CoverageStatus.NOT_APPLICABLE,
            required=False,
        )
        for category in CoverageCategory
    )


def _minimum_base(
    *,
    task: str,
    counter: TokenCounter,
    generation: int,
    metadata: dict[str, str | int | bool | list[str]],
) -> tuple[int, int]:
    def required(field_budget: int) -> tuple[int, int]:
        pack = _pack(
            task=task,
            requested_budget=field_budget,
            budget=field_budget,
            budget_expanded=False,
            counter=counter,
            items=(),
            completeness_status=CompletenessStatus.NOT_APPLICABLE,
            category_coverage=_empty_coverage(),
            omitted_required=(),
            ambiguous=(),
            omitted=(),
            conflicts=(),
            generation=generation,
            retrieval_metadata=metadata,
        )
        return pack.serialized_estimated_tokens, pack.metadata_tokens

    candidate = 64
    for _ in range(16):
        value, _ = required(candidate)
        if value <= candidate:
            break
        candidate = value
    # Find the exact local boundary (the budget field changes only logarithmically).
    lower = max(1, candidate - 256)
    minimum = candidate
    overhead = 0
    for value in range(lower, candidate + 1):
        needed, metadata_tokens = required(value)
        if needed <= value:
            minimum, overhead = value, metadata_tokens
            break
    return minimum, overhead


def _filter_kwargs(filters: FilterSet) -> dict[str, Any]:
    return {
        "documents": set(filters.documents) if filters.documents else None,
        "authority_floor": filters.authority_floor,
        "authorities": set(filters.authorities) if filters.authorities else None,
        "exclude_documents": set(filters.exclude_documents),
        "heading_prefix": filters.heading_prefix,
        "scope": filters.scope,
    }


def _match_excerpt(
    engine: ContextEngine, source: SourceItem, task: str, filters: FilterSet
) -> SourceItem | None:
    if source.source_type == "excerpt":
        return source
    kwargs = _filter_kwargs(filters)
    kwargs["documents"] = {source.provenance.document_id}
    for query in (task, source.provenance.heading_path[-1]):
        for hit in engine.search_exact(query, limit=20, **kwargs):
            if hit.source.provenance.section_id == source.provenance.section_id:
                return hit.source
    return None


def build_context_pack(
    engine: ContextEngine,
    task: str,
    token_budget: int,
    *,
    filters: FilterSet | None = None,
    counter: TokenCounter | None = None,
    allow_required_budget_expansion: bool = False,
    checkpoint_document: str | None = None,
) -> ContextPack:
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    if token_budget > engine.config.limits.max_token_budget:
        raise ValueError("token_budget exceeds configured maximum")
    token_counter = counter or ApproximateGenericCounter()
    filters = filters or FilterSet()
    generation = engine.store.index_generation()
    classification = classify_query(task)
    retrieval_metadata: dict[str, str | int | bool | list[str]] = {
        "strategy": "evidence-first+graph+staged-authority+match-centered",
        "active_channels": list(engine.active_channels),
        "has_structural_query": bool(classification.structural_terms),
        "filtering": "pre-cutoff",
        "candidate_count": 0,
        "selected_count": 0,
        "primary_retained": True,
        "required_budget_expansion_allowed": allow_required_budget_expansion,
    }
    minimum, base_metadata = _minimum_base(
        task=task,
        counter=token_counter,
        generation=generation,
        metadata=retrieval_metadata,
    )
    if token_budget < minimum:
        raise ContextBudgetTooSmall(
            requested_budget=token_budget,
            minimum_required=minimum,
            task_token_estimate=token_counter.count_text(task),
            metadata_overhead=base_metadata,
        )

    candidates: dict[tuple[str, int, int], _Candidate] = {}
    ambiguities: list[AmbiguousEvidence] = []
    unresolved_required: list[OmittedRequiredEvidence] = []
    sequence = 0

    def candidate_key(source: SourceItem) -> tuple[str, int, int]:
        provenance = source.provenance
        for key, existing in candidates.items():
            other = existing.source.provenance
            if (
                provenance.section_id == other.section_id
                and provenance.start_offset < other.end_offset
                and other.start_offset < provenance.end_offset
            ):
                return key
        return (provenance.section_id, provenance.start_offset, provenance.end_offset)

    def add(
        source: SourceItem,
        category: str,
        reason: str,
        *,
        relevance: float,
        confidence: float,
        primary: bool = False,
        required: bool = False,
        coverage_categories: tuple[CoverageCategory, ...] = (),
        checkpoint_id: str | None = None,
        dependency: str | None = None,
    ) -> bool:
        nonlocal sequence
        p = source.provenance
        if p.index_generation != generation:
            raise RuntimeError("candidate generation changed while planning context")
        if (
            filters.documents
            and p.document_path not in filters.documents
            and p.document_id not in filters.documents
        ):
            return False
        if p.document_path in filters.exclude_documents:
            return False
        if filters.authority_floor is not None and p.authority < filters.authority_floor:
            return False
        if filters.authorities and p.authority not in filters.authorities:
            return False
        key = candidate_key(source)
        candidate = _Candidate(
            source,
            _CATEGORIES[category],
            category,
            reason,
            relevance,
            confidence,
            sequence,
            primary,
            required,
            coverage_categories,
            checkpoint_id,
            dependency,
        )
        sequence += 1
        existing = candidates.get(key)
        if existing is None:
            candidates[key] = candidate
            return True
        merged_categories = tuple(
            dict.fromkeys((*existing.coverage_categories, *candidate.coverage_categories))
        )
        replace = (candidate.primary and not existing.primary) or (
            candidate.category_rank < existing.category_rank
        )
        winner, other = (candidate, existing) if replace else (existing, candidate)
        winner.required = winner.required or other.required
        winner.primary = winner.primary or other.primary
        winner.coverage_categories = merged_categories
        winner.checkpoint_id = winner.checkpoint_id or other.checkpoint_id
        winner.dependency = winner.dependency or other.dependency
        if other.reason not in winner.reason:
            winner.reason += f"; {other.reason}"
        candidates[key] = winner
        return True

    filter_kwargs = _filter_kwargs(filters)
    primary_ids = classification.checkpoint_ids
    primary_section_ids: set[str] = set()
    for checkpoint_id in primary_ids:
        document_filter = checkpoint_document
        if document_filter is None and filters.documents and len(filters.documents) == 1:
            document_filter = next(iter(filters.documents))
        checkpoint = engine.get_checkpoint(checkpoint_id, document=document_filter)
        root = checkpoint.sources[0]
        primary_section_ids.add(root.provenance.section_id)
        add(
            root,
            "direct_requirement",
            f"exact requested checkpoint {checkpoint_id.upper()}",
            relevance=100.0,
            confidence=1.0,
            primary=True,
            required=True,
            coverage_categories=(CoverageCategory.PRIMARY,),
            checkpoint_id=checkpoint_id.upper(),
        )
        for child in checkpoint.sources[1:]:
            heading = canonical_heading(child.provenance.heading_path[-1])
            coverage = CoverageCategory.CHECKPOINT_DESCENDANTS
            if "depend" in heading:
                category = "explicit_dependency"
                coverage = CoverageCategory.DEPENDENCIES
            elif any(value in heading for value in ("interface", "model", "architect")):
                category = "interface_or_model"
                coverage = CoverageCategory.ARCHITECTURE
            elif "security" in heading:
                category = "security_constraint"
                coverage = CoverageCategory.SECURITY
            elif "verify" in heading:
                category = "acceptance_or_verify"
                coverage = CoverageCategory.VERIFICATION
            elif "test" in heading or "accept" in heading:
                category = "acceptance_or_verify"
                coverage = CoverageCategory.ACCEPTANCE
            elif "failure" in heading or "error" in heading:
                category = "error_or_failure"
            else:
                category = "checkpoint_field"
            add(
                child,
                category,
                f"structured field of requested {checkpoint_id.upper()}",
                relevance=50.0,
                confidence=1.0,
                required=True,
                coverage_categories=tuple(
                    dict.fromkeys((CoverageCategory.CHECKPOINT_DESCENDANTS, coverage))
                ),
                checkpoint_id=checkpoint_id.upper(),
            )

    direct = engine.search(task, limit=min(12, engine.config.limits.max_results), **filter_kwargs)
    uncovered_terms = {
        term.casefold()
        for term in re.findall(r"[\w.@/:-]+", task, flags=re.UNICODE)
        if len(term) > 2 and term.casefold() not in {"implement", "using", "with", "from"}
    }
    for rank, hit in enumerate(direct, start=1):
        source_folded = hit.source.text.casefold()
        covered_terms = {term for term in uncovered_terms if term in source_folded}
        direct_required = bool(not primary_ids and (rank == 1 or covered_terms))
        if direct_required:
            uncovered_terms -= covered_terms
        add(
            hit.source,
            "direct_requirement" if not primary_ids else "semantic_fallback",
            f"direct retrieval rank {rank}; channels={','.join(hit.channels)}",
            relevance=hit.score,
            confidence=max(0.5, 1.0 - rank * 0.04),
            required=direct_required,
            coverage_categories=((CoverageCategory.PRIMARY,) if not primary_ids else ()),
        )

    edge_categories = {
        EdgeType.REFERENCES: ("direct_reference", "explicit source reference"),
        EdgeType.DEPENDS_ON: ("explicit_dependency", "declared dependency"),
        EdgeType.USES_TYPE: ("interface_or_model", "referenced interface/model"),
        EdgeType.RELATED_SECTION: ("direct_reference", "linked related section"),
    }
    seeds = list(primary_section_ids) or [hit.source.provenance.section_id for hit in direct[:4]]
    checkpoint_children: list[str] = []

    def add_reference(reference: Any, relevance: float) -> None:
        edge = reference.edge
        if edge.edge_type not in edge_categories:
            return
        reference_coverage: CoverageCategory | None = (
            CoverageCategory.DEPENDENCIES
            if edge.edge_type is EdgeType.DEPENDS_ON
            else CoverageCategory.ARCHITECTURE
            if edge.edge_type is EdgeType.USES_TYPE
            else None
        )
        required = bool(primary_ids and reference_coverage is not None)
        if reference.target is None:
            if not required:
                return
            assert reference_coverage is not None
            if edge.status is ResolutionStatus.AMBIGUOUS:
                ambiguities.append(
                    AmbiguousEvidence(
                        category=reference_coverage,
                        label=edge.label,
                        reason=edge.reason,
                        source=reference.source.ref,
                        candidates=reference.candidates,
                    )
                )
            else:
                provenance = reference.source.provenance
                unresolved_required.append(
                    OmittedRequiredEvidence(
                        category=reference_coverage,
                        reason=f"unresolved required graph evidence: {edge.reason}",
                        document_id=provenance.document_id,
                        document_path=provenance.document_path,
                        section_id=provenance.section_id,
                        checkpoint_id=primary_ids[0].upper() if primary_ids else None,
                        dependency=edge.label
                        if reference_coverage is CoverageCategory.DEPENDENCIES
                        else None,
                        start_line=provenance.start_line,
                        end_line=provenance.end_line,
                        range_sha256=provenance.range_sha256,
                    )
                )
            return
        category, reason = edge_categories[edge.edge_type]
        accepted = add(
            reference.target,
            category,
            f"{reason}: {edge.label}",
            relevance=relevance,
            confidence=1.0,
            required=required,
            coverage_categories=((reference_coverage,) if reference_coverage is not None else ()),
            checkpoint_id=primary_ids[0].upper() if primary_ids else None,
            dependency=(
                edge.label if reference_coverage is CoverageCategory.DEPENDENCIES else None
            ),
        )
        if required and not accepted and reference_coverage is not None:
            provenance = reference.target.provenance
            unresolved_required.append(
                OmittedRequiredEvidence(
                    category=reference_coverage,
                    reason="required graph target excluded by context-pack filters",
                    document_id=provenance.document_id,
                    document_path=provenance.document_path,
                    section_id=provenance.section_id,
                    checkpoint_id=primary_ids[0].upper(),
                    dependency=(
                        edge.label if reference_coverage is CoverageCategory.DEPENDENCIES else None
                    ),
                    start_line=provenance.start_line,
                    end_line=provenance.end_line,
                    range_sha256=provenance.range_sha256,
                )
            )

    for section_id in seeds:
        for reference in engine.store.get_references(section_id):
            if reference.edge.edge_type is EdgeType.PARENT_OF and reference.target is not None:
                checkpoint_children.append(reference.target.provenance.section_id)
            add_reference(reference, 25.0)
    for section_id in checkpoint_children:
        for reference in engine.store.get_references(section_id):
            add_reference(reference, 20.0)

    # For an exact checkpoint, add only heading-evidenced global normative security and
    # acceptance sources when those fields are not descendants. This is structural evidence,
    # not an unconditional generic semantic query.
    if primary_ids:
        existing_coverage = {
            coverage
            for candidate in candidates.values()
            for coverage in candidate.coverage_categories
        }
        structural_global = (
            (
                "Architecture",
                "interface_or_model",
                CoverageCategory.ARCHITECTURE,
                "global normative architecture heading",
            ),
            (
                "Security",
                "security_constraint",
                CoverageCategory.SECURITY,
                "security constraint: global normative heading",
            ),
            (
                "Tests",
                "acceptance_or_verify",
                CoverageCategory.ACCEPTANCE,
                "global acceptance heading",
            ),
            (
                "Acceptance",
                "acceptance_or_verify",
                CoverageCategory.ACCEPTANCE,
                "global acceptance heading",
            ),
            (
                "Verify",
                "acceptance_or_verify",
                CoverageCategory.VERIFICATION,
                "global verification heading",
            ),
        )
        for query, category, global_coverage, reason in structural_global:
            if global_coverage in existing_coverage:
                continue
            for hit in engine.search_exact(
                query,
                limit=3,
                authority_floor=Authority.NORMATIVE,
                **{key: value for key, value in filter_kwargs.items() if key != "authority_floor"},
            ):
                if canonical_heading(query) in canonical_heading(
                    hit.source.provenance.heading_path[-1]
                ):
                    add(
                        hit.source,
                        category,
                        reason,
                        relevance=15.0,
                        confidence=1.0,
                        required=True,
                        coverage_categories=(global_coverage,),
                        checkpoint_id=primary_ids[0].upper(),
                    )

    # Generic searches are conditional fallback, never unconditional pack pollution.
    folded_task = task.casefold()
    fallbacks: list[tuple[str, str, str]] = []
    if len(candidates) < 3:
        fallbacks.append((task, "semantic_fallback", "insufficient explicit evidence"))
    if "security" in folded_task:
        fallbacks.append(("security", "security_constraint", "task explicitly requests security"))
    if any(word in folded_task for word in ("error", "failure", "timeout", "exception")):
        fallbacks.append((task, "error_or_failure", "task-specific failure/error evidence"))
    if any(word in folded_task for word in ("test", "acceptance", "verify")):
        fallbacks.append((task, "acceptance_or_verify", "task-specific acceptance evidence"))
    fallback_coverage = {
        "security_constraint": CoverageCategory.SECURITY,
        "acceptance_or_verify": CoverageCategory.ACCEPTANCE,
    }
    for query, category, reason in fallbacks:
        for rank, hit in enumerate(engine.search(query, limit=3, **filter_kwargs), start=1):
            fallback_category = fallback_coverage.get(category)
            add(
                hit.source,
                category,
                f"{reason}; rank {rank}",
                relevance=hit.score,
                confidence=0.6,
                required=fallback_category is not None,
                coverage_categories=((fallback_category,) if fallback_category is not None else ()),
            )

    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            not item.primary,
            not item.required,
            item.category_rank,
            -item.relevance,
            -int(item.source.provenance.authority),
            -item.source.provenance.priority,
            item.sequence,
        ),
    )
    selected: list[ContextPackItem] = []
    selected_by_key: dict[tuple[str, int, int], ContextPackItem] = {}
    selected_keys: set[tuple[str, int, int]] = set()
    working_budget = token_budget
    all_keys = [
        (
            item.source.provenance.section_id,
            item.source.provenance.start_offset,
            item.source.provenance.end_offset,
        )
        for item in ordered
    ]

    def omission_label(key: tuple[str, int, int]) -> str:
        # Legacy optional omission summary remains compact. Required omissions below carry the
        # exact document/section/range location.
        return key[0]

    def item(candidate: _Candidate, source: SourceItem) -> ContextPackItem:
        return ContextPackItem(
            source=source,
            reason=candidate.reason,
            category=candidate.category,
            relevance=candidate.relevance,
            confidence=candidate.confidence,
            estimated_tokens=token_counter.count_text(source.text),
            section_sha256=source.provenance.section_sha256,
            range_sha256=source.provenance.range_sha256,
            index_generation=generation,
        )

    for candidate in ordered:
        source = candidate.source
        options = [source]
        excerpt = _match_excerpt(engine, source, task, filters)
        if (
            excerpt is not None
            and excerpt.provenance.range_sha256 != source.provenance.range_sha256
        ):
            options.append(excerpt)
        accepted = False
        for option in options:
            candidate_item = item(candidate, option)
            tentative = [*selected, candidate_item]
            key = _candidate_range_key(candidate)
            tentative_keys = selected_keys | {key}
            tentative_by_key = {**selected_by_key, key: candidate_item}
            omitted = [omission_label(value) for value in all_keys if value not in tentative_keys]
            conflicts = _possible_conflicts(tentative)
            status, category_coverage, required_omissions = _coverage_state(
                ordered,
                tentative_by_key,
                unresolved_required,
                ambiguities,
                conflicts,
            )
            pack = _pack(
                task=task,
                requested_budget=token_budget,
                budget=working_budget,
                budget_expanded=working_budget > token_budget,
                counter=token_counter,
                items=tentative,
                completeness_status=status,
                category_coverage=category_coverage,
                omitted_required=required_omissions,
                ambiguous=ambiguities,
                omitted=omitted,
                conflicts=conflicts,
                generation=generation,
                retrieval_metadata={
                    **retrieval_metadata,
                    "candidate_count": len(ordered),
                    "selected_count": len(tentative),
                },
            )
            next_budget = working_budget
            if (
                pack.serialized_estimated_tokens > working_budget
                and candidate.required
                and allow_required_budget_expansion
            ):
                next_budget = pack.serialized_estimated_tokens
                for _ in range(4):
                    if next_budget > engine.config.limits.max_token_budget:
                        break
                    pack = _pack(
                        task=task,
                        requested_budget=token_budget,
                        budget=next_budget,
                        budget_expanded=True,
                        counter=token_counter,
                        items=tentative,
                        completeness_status=status,
                        category_coverage=category_coverage,
                        omitted_required=required_omissions,
                        ambiguous=ambiguities,
                        omitted=omitted,
                        conflicts=conflicts,
                        generation=generation,
                        retrieval_metadata={
                            **retrieval_metadata,
                            "candidate_count": len(ordered),
                            "selected_count": len(tentative),
                        },
                    )
                    if pack.serialized_estimated_tokens <= next_budget:
                        break
                    next_budget = pack.serialized_estimated_tokens
            if (
                pack.serialized_estimated_tokens <= next_budget
                and next_budget <= engine.config.limits.max_token_budget
            ):
                selected = tentative
                selected_by_key = tentative_by_key
                selected_keys = tentative_keys
                working_budget = next_budget
                accepted = True
                break
        if candidate.primary and not accepted:
            raise PrimaryRequirementTooLarge(
                requested_budget=token_budget,
                minimum_required=max(token_budget + 1, pack.serialized_estimated_tokens),
                task_token_estimate=token_counter.count_text(task),
                metadata_overhead=pack.metadata_tokens,
            )

    omitted = [omission_label(value) for value in all_keys if value not in selected_keys]
    conflicts = _possible_conflicts(selected)
    status, category_coverage, required_omissions = _coverage_state(
        ordered,
        selected_by_key,
        unresolved_required,
        ambiguities,
        conflicts,
    )
    result = _pack(
        task=task,
        requested_budget=token_budget,
        budget=working_budget,
        budget_expanded=working_budget > token_budget,
        counter=token_counter,
        items=selected,
        completeness_status=status,
        category_coverage=category_coverage,
        omitted_required=required_omissions,
        ambiguous=ambiguities,
        omitted=omitted,
        conflicts=conflicts,
        generation=generation,
        retrieval_metadata={
            **retrieval_metadata,
            "candidate_count": len(ordered),
            "selected_count": len(selected),
            "primary_retained": not primary_section_ids
            or primary_section_ids <= {item.source.provenance.section_id for item in selected},
        },
    )
    if result.serialized_estimated_tokens > working_budget:
        raise RuntimeError("internal error: serialized context pack exceeds promised budget")
    if engine.store.index_generation() != generation or any(
        item.index_generation != generation for item in result.items
    ):
        raise RuntimeError("index generation changed while building context pack")
    return result
