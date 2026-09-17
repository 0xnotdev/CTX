"""Budgeted, structurally safe context-pack construction."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from markdown_it import MarkdownIt

from ctx.models import (
    Authority,
    ContextPack,
    ContextPackItem,
    EdgeType,
    PossibleConflict,
    SourceItem,
)
from ctx.parser import sha256_text
from ctx.retrieval import classify_query

if TYPE_CHECKING:
    from ctx.service import ContextEngine


class TokenCounter(Protocol):
    @property
    def name(self) -> str: ...

    def count(self, text: str) -> int: ...


class ApproximateTokenCounter:
    """Stable conservative approximation; explicitly not a universal tokenizer."""

    @property
    def name(self) -> str:
        return "utf8_bytes_div4_ceiling"

    def count(self, text: str) -> int:
        return max(1, (len(text.encode("utf-8")) + 3) // 4)


@dataclass
class _Candidate:
    source: SourceItem
    category_rank: int
    category: str
    reason: str
    sequence: int


_CATEGORIES = {
    "direct_requirement": 0,
    "direct_reference": 1,
    "explicit_dependency": 2,
    "interface_or_model": 3,
    "global_constraint": 4,
    "error_or_failure": 5,
    "security_constraint": 6,
    "acceptance_or_verify": 7,
    "neighbor_context": 8,
}


def _safe_prefix(
    source: SourceItem, max_text_tokens: int, counter: TokenCounter
) -> SourceItem | None:
    if counter.count(source.text) <= max_text_tokens:
        return source
    lines = source.text.splitlines(keepends=True)
    markdown = MarkdownIt("commonmark", {"html": False}).enable("table")
    boundaries: set[int] = set()
    heading_end: int | None = None
    for token in markdown.parse(source.text):
        if token.level == 0 and token.map is not None:
            boundaries.add(token.map[1])
            if token.type == "heading_open" and token.map[0] == 0:
                heading_end = token.map[1]
    best: tuple[int, str] | None = None
    for end in sorted(boundaries):
        # Do not return a detached heading when its first content block cannot fit.
        if heading_end is not None and end <= heading_end:
            continue
        text = "".join(lines[:end])
        if text and counter.count(text) <= max_text_tokens:
            best = (end, text)
    if best is None:
        return None
    end, text = best
    provenance = source.provenance.model_copy(
        update={"end_line": source.provenance.start_line + end - 1}
    )
    return SourceItem(text=text, provenance=provenance, score=source.score, reason=source.reason)


def _item_cost(source: SourceItem, reason: str, category: str, counter: TokenCounter) -> int:
    provenance = source.provenance.model_dump_json()
    return (
        counter.count(source.text)
        + counter.count(provenance)
        + counter.count(reason + category)
        + 8
    )


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
                        sources=(left.source, right.source),
                    )
                )
                if len(conflicts) >= 10:
                    return tuple(conflicts)
    return tuple(conflicts)


def build_context_pack(
    engine: ContextEngine,
    task: str,
    token_budget: int,
    *,
    documents: set[str] | None = None,
    authority_floor: Authority | None = None,
    counter: TokenCounter | None = None,
) -> ContextPack:
    """Build a deliberate source package in requirement/dependency/constraint order."""
    if token_budget < 64:
        raise ValueError("token_budget must be at least 64")
    if token_budget > 1_000_000:
        raise ValueError("token_budget exceeds safe maximum")
    token_counter = counter or ApproximateTokenCounter()
    candidates: dict[str, _Candidate] = {}
    sequence = 0

    def add(source: SourceItem, category: str, reason: str) -> None:
        nonlocal sequence
        # Graph rows are navigation-only; re-enter the shared exact-source stale guard before
        # any candidate can become source-bearing output.
        source = engine.get_section(source.provenance.section_id)
        if documents is not None and source.provenance.document_path not in documents:
            return
        if authority_floor is not None and source.provenance.authority < authority_floor:
            return
        identifier = source.provenance.section_id
        candidate = _Candidate(source, _CATEGORIES[category], category, reason, sequence)
        sequence += 1
        existing = candidates.get(identifier)
        if existing is None or candidate.category_rank < existing.category_rank:
            candidates[identifier] = candidate
        elif reason not in existing.reason:
            existing.reason += f"; {reason}"

    direct = engine.search(task, limit=min(8, engine.config.limits.max_results))
    for rank, hit in enumerate(direct, start=1):
        add(hit.source, "direct_requirement", f"direct hybrid requirement match rank {rank}")

    categories = {
        EdgeType.REFERENCES: ("direct_reference", "explicit source reference"),
        EdgeType.DEPENDS_ON: ("explicit_dependency", "declared dependency"),
        EdgeType.USES_TYPE: ("interface_or_model", "referenced interface/model"),
        EdgeType.PARENT_OF: ("neighbor_context", "adjacent child section"),
        EdgeType.CHILD_OF: ("neighbor_context", "containing parent section"),
        EdgeType.RELATED_SECTION: ("direct_reference", "linked related section"),
    }
    checkpoint_children: list[SourceItem] = []
    for hit in direct[:4]:
        for reference in engine.store.get_references(hit.source.provenance.section_id):
            if reference.target is None:
                continue
            category, reason = categories[reference.edge.edge_type]
            add(reference.target, category, f"{reason}: {reference.edge.label}")
            if reference.edge.edge_type is EdgeType.PARENT_OF:
                checkpoint_children.append(reference.target)
    # Checkpoint field sections often carry dependencies/models beneath the root. Traverse that
    # one explicit structural level rather than relying on vector proximity.
    for child in checkpoint_children:
        for reference in engine.store.get_references(child.provenance.section_id):
            if reference.target is None or reference.edge.edge_type not in {
                EdgeType.DEPENDS_ON,
                EdgeType.REFERENCES,
                EdgeType.USES_TYPE,
                EdgeType.RELATED_SECTION,
            }:
                continue
            category, reason = categories[reference.edge.edge_type]
            add(reference.target, category, f"{reason}: {reference.edge.label}")

    auxiliary = (
        ("global constraints", "global_constraint", "workspace global constraint"),
        ("error failure invalid", "error_or_failure", "named error/failure rule"),
        ("security constraint", "security_constraint", "security constraint"),
        (
            "acceptance tests verify command",
            "acceptance_or_verify",
            "acceptance/test/verification material",
        ),
    )
    for query, category, reason in auxiliary:
        for hit in engine.search(query, limit=3):
            add(hit.source, category, reason)

    ordered = sorted(
        candidates.values(),
        key=lambda candidate: (
            candidate.category_rank,
            candidate.sequence,
            candidate.source.provenance.authority is Authority.GENERATED,
            -int(candidate.source.provenance.authority),
            -candidate.source.provenance.priority,
            candidate.source.provenance.document_path,
            candidate.source.provenance.start_line,
        ),
    )

    base_tokens = token_counter.count(task) + 24
    used = base_tokens
    selected: list[ContextPackItem] = []
    omitted: list[str] = []
    for candidate in ordered:
        cost = _item_cost(candidate.source, candidate.reason, candidate.category, token_counter)
        source = candidate.source
        if used + cost > token_budget:
            overhead = cost - token_counter.count(source.text)
            remaining = token_budget - used - overhead
            truncated = _safe_prefix(source, remaining, token_counter) if remaining > 0 else None
            if truncated is None:
                omitted.append(candidate.source.provenance.section_id)
                continue
            source = truncated
            cost = _item_cost(source, candidate.reason, candidate.category, token_counter)
            if used + cost > token_budget:
                omitted.append(candidate.source.provenance.section_id)
                continue
        selected.append(
            ContextPackItem(
                source=source,
                reason=candidate.reason,
                category=candidate.category,
                estimated_tokens=cost,
                range_sha256=sha256_text(source.text),
            )
        )
        used += cost

    classification = classify_query(task)
    return ContextPack(
        task=task,
        token_budget=token_budget,
        estimated_tokens=used,
        token_count_method=token_counter.name,
        items=tuple(selected),
        omitted_relevant_sections=tuple(dict.fromkeys(omitted)),
        possible_conflicts=_possible_conflicts(selected),
        index_version=engine.store.index_version(),
        retrieval_metadata={
            "strategy": "structural+graph+constraint-priority+rrf",
            "candidate_count": len(ordered),
            "selected_count": len(selected),
            "has_structural_query": bool(classification.structural_terms),
        },
    )
