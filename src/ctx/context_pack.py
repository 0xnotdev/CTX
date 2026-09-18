"""Exact-source context planning with complete serialized-response budgeting."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from ctx.models import (
    Authority,
    BudgetMethod,
    ContextPack,
    ContextPackItem,
    EdgeType,
    FilterSet,
    PossibleConflict,
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


def _pack(
    *,
    task: str,
    budget: int,
    counter: TokenCounter,
    items: Sequence[ContextPackItem],
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
            token_budget=budget,
            budget_method=counter.method,
            budget_counter_identity=counter.identity,
            budget_safety_margin=safety,
            content_tokens=content_tokens,
            metadata_tokens=metadata_tokens,
            serialized_estimated_tokens=serialized + safety,
            estimated_tokens=serialized + safety,
            token_count_method=counter.identity,
            items=tuple(items),
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
            budget=field_budget,
            counter=counter,
            items=(),
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

    candidates: dict[str, _Candidate] = {}
    sequence = 0

    def add(
        source: SourceItem,
        category: str,
        reason: str,
        *,
        relevance: float,
        confidence: float,
        primary: bool = False,
    ) -> None:
        nonlocal sequence
        p = source.provenance
        if p.index_generation != generation:
            raise RuntimeError("candidate generation changed while planning context")
        if (
            filters.documents
            and p.document_path not in filters.documents
            and p.document_id not in filters.documents
        ):
            return
        if p.document_path in filters.exclude_documents:
            return
        if filters.authority_floor is not None and p.authority < filters.authority_floor:
            return
        if filters.authorities and p.authority not in filters.authorities:
            return
        key = p.section_id
        candidate = _Candidate(
            source,
            _CATEGORIES[category],
            category,
            reason,
            relevance,
            confidence,
            sequence,
            primary,
        )
        sequence += 1
        existing = candidates.get(key)
        if (
            existing is None
            or (candidate.primary and not existing.primary)
            or candidate.category_rank < existing.category_rank
        ):
            candidates[key] = candidate
        elif reason not in existing.reason:
            existing.reason += f"; {reason}"

    filter_kwargs = _filter_kwargs(filters)
    primary_ids = classification.checkpoint_ids
    primary_section_ids: set[str] = set()
    for checkpoint_id in primary_ids:
        document_filter: str | None = None
        if filters.documents and len(filters.documents) == 1:
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
        )
        for child in checkpoint.sources[1:]:
            heading = child.provenance.heading_path[-1].casefold()
            if "depend" in heading:
                category = "explicit_dependency"
            elif "interface" in heading or "model" in heading:
                category = "interface_or_model"
            elif "security" in heading:
                category = "security_constraint"
            elif any(value in heading for value in ("test", "accept", "verify")):
                category = "acceptance_or_verify"
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
            )

    direct = engine.search(task, limit=min(12, engine.config.limits.max_results), **filter_kwargs)
    for rank, hit in enumerate(direct, start=1):
        add(
            hit.source,
            "direct_requirement" if not primary_ids else "semantic_fallback",
            f"direct retrieval rank {rank}; channels={','.join(hit.channels)}",
            relevance=hit.score,
            confidence=max(0.5, 1.0 - rank * 0.04),
        )

    edge_categories = {
        EdgeType.REFERENCES: ("direct_reference", "explicit source reference"),
        EdgeType.DEPENDS_ON: ("explicit_dependency", "declared dependency"),
        EdgeType.USES_TYPE: ("interface_or_model", "referenced interface/model"),
        EdgeType.RELATED_SECTION: ("direct_reference", "linked related section"),
    }
    seeds = list(primary_section_ids) or [hit.source.provenance.section_id for hit in direct[:4]]
    checkpoint_children: list[str] = []
    for section_id in seeds:
        for reference in engine.store.get_references(section_id):
            if reference.edge.edge_type is EdgeType.PARENT_OF and reference.target is not None:
                checkpoint_children.append(reference.target.provenance.section_id)
            if reference.target is None or reference.edge.edge_type not in edge_categories:
                continue  # UNRESOLVED/AMBIGUOUS is never treated as certainty.
            category, reason = edge_categories[reference.edge.edge_type]
            add(
                reference.target,
                category,
                f"{reason}: {reference.edge.label}",
                relevance=25.0,
                confidence=1.0,
            )
    for section_id in checkpoint_children:
        for reference in engine.store.get_references(section_id):
            if reference.target is None or reference.edge.edge_type not in edge_categories:
                continue
            category, reason = edge_categories[reference.edge.edge_type]
            add(
                reference.target,
                category,
                f"{reason}: {reference.edge.label}",
                relevance=20.0,
                confidence=1.0,
            )

    # For an exact checkpoint, add only heading-evidenced global normative security and
    # acceptance sources when those fields are not descendants. This is structural evidence,
    # not an unconditional generic semantic query.
    if primary_ids:
        existing_categories = {candidate.category for candidate in candidates.values()}
        structural_global = (
            ("Security", "security_constraint", "security constraint: global normative heading"),
            ("Tests", "acceptance_or_verify", "global acceptance/verification heading"),
            ("Verify", "acceptance_or_verify", "global verification heading"),
        )
        for query, category, reason in structural_global:
            if category in existing_categories:
                continue
            for hit in engine.search_exact(
                query,
                limit=3,
                authority_floor=Authority.NORMATIVE,
                **{key: value for key, value in filter_kwargs.items() if key != "authority_floor"},
            ):
                if query.casefold() in hit.source.provenance.heading_path[-1].casefold():
                    add(
                        hit.source,
                        category,
                        reason,
                        relevance=15.0,
                        confidence=1.0,
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
    for query, category, reason in fallbacks:
        for rank, hit in enumerate(engine.search(query, limit=3, **filter_kwargs), start=1):
            add(
                hit.source,
                category,
                f"{reason}; rank {rank}",
                relevance=hit.score,
                confidence=0.6,
            )

    ordered = sorted(
        candidates.values(),
        key=lambda item: (
            not item.primary,
            item.category_rank,
            -item.relevance,
            -int(item.source.provenance.authority),
            -item.source.provenance.priority,
            item.sequence,
        ),
    )
    selected: list[ContextPackItem] = []
    selected_ids: set[str] = set()
    all_ids = [item.source.provenance.section_id for item in ordered]

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
            tentative = [*selected, item(candidate, option)]
            tentative_ids = selected_ids | {candidate.source.provenance.section_id}
            omitted = [identifier for identifier in all_ids if identifier not in tentative_ids]
            conflicts = _possible_conflicts(tentative)
            pack = _pack(
                task=task,
                budget=token_budget,
                counter=token_counter,
                items=tentative,
                omitted=omitted,
                conflicts=conflicts,
                generation=generation,
                retrieval_metadata={
                    **retrieval_metadata,
                    "candidate_count": len(ordered),
                    "selected_count": len(tentative),
                },
            )
            if pack.serialized_estimated_tokens <= token_budget:
                selected = tentative
                selected_ids = tentative_ids
                accepted = True
                break
        if candidate.primary and not accepted:
            raise PrimaryRequirementTooLarge(
                requested_budget=token_budget,
                minimum_required=max(token_budget + 1, pack.serialized_estimated_tokens),
                task_token_estimate=token_counter.count_text(task),
                metadata_overhead=pack.metadata_tokens,
            )

    omitted = [identifier for identifier in all_ids if identifier not in selected_ids]
    conflicts = _possible_conflicts(selected)
    result = _pack(
        task=task,
        budget=token_budget,
        counter=token_counter,
        items=selected,
        omitted=omitted,
        conflicts=conflicts,
        generation=generation,
        retrieval_metadata={
            **retrieval_metadata,
            "candidate_count": len(ordered),
            "selected_count": len(selected),
            "primary_retained": not primary_section_ids or primary_section_ids <= selected_ids,
        },
    )
    if result.serialized_estimated_tokens > token_budget:
        raise RuntimeError("internal error: serialized context pack exceeds promised budget")
    if engine.store.index_generation() != generation or any(
        item.index_generation != generation for item in result.items
    ):
        raise RuntimeError("index generation changed while building context pack")
    return result
