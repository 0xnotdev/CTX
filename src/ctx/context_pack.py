"""Exact-source context planning with complete serialized-response budgeting."""

from __future__ import annotations

import hashlib
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
    ContextMode,
    ContextPack,
    ContextPackItem,
    CoverageCategory,
    CoverageStatus,
    EdgeType,
    FilterSet,
    OmittedRequiredEvidence,
    PossibleConflict,
    ResolutionStatus,
    RetrievalMode,
    SourceItem,
    SourceRef,
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
    "decision_constraint": 5,
    "current_state": 6,
    "global_constraint": 7,
    "error_or_failure": 8,
    "security_constraint": 9,
    "acceptance_or_verify": 10,
    "testing_constraint": 11,
    "normative_constraint": 12,
    "semantic_fallback": 13,
    "neighbor_context": 14,
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


_IDENTIFIER_PATTERN = re.compile(
    r"\bCP-\d+\b"
    r"|\b(?:[A-Z][a-z0-9]+){2,}\b"
    r"|\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b"
    r"|\b[A-Za-z_][\w-]*(?:\.[\w-]+)+(?:@\d+)?\b"
)
_WORD_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9_-]*")
_MODAL_NEGATIVE = re.compile(r"\b(?:must|shall|should|may)\s+not\b|\bmust\s+never\b", re.I)
_MODAL_POSITIVE = re.compile(r"\b(?:must|shall|should)\b(?!\s+(?:not|never)\b)", re.I)
_ENABLED = re.compile(r"\benabled\b", re.I)
_DISABLED = re.compile(r"\bdisabled\b", re.I)
_ALLOWED = re.compile(r"\ballowed\b", re.I)
_PROHIBITED = re.compile(r"\b(?:prohibited|disallowed)\b", re.I)
_REQUIRED = re.compile(r"\brequired\b", re.I)
_FORBIDDEN = re.compile(r"\bforbidden\b", re.I)
_LEADING_FIELD = re.compile(r"^[-*+]\s*(?:\*\*)?[^:\n]{1,80}:(?:\*\*)?\s*")
_TRAILING_QUALIFIER = re.compile(
    r"\b(?:for|in|during|when|under|on|within|against|by|to)\b.*", re.I
)
_STOP_SUBJECT_WORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "any",
        "are",
        "as",
        "be",
        "below",
        "by",
        "each",
        "every",
        "for",
        "from",
        "has",
        "have",
        "if",
        "in",
        "is",
        "it",
        "its",
        "no",
        "of",
        "or",
        "that",
        "the",
        "their",
        "them",
        "these",
        "this",
        "those",
        "to",
        "with",
    }
)
_PROPERTY_STOP_WORDS = frozenset({"a", "an", "be", "is", "are", "the", "to"})
_SUBJECT_WINDOW_CHARS = 96


@dataclass(frozen=True)
class _IdentifierSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _ClauseSpan:
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class _PolarityMarker:
    family: str
    polarity: int
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class _Assertion:
    subject: str
    subject_key: str
    property_key: str
    family: str
    polarity: int
    marker_label: str
    source_ref: SourceRef


def _identifiers(text: str) -> set[str]:
    return {match.group(0) for match in _IDENTIFIER_PATTERN.finditer(text)}


def _identifier_spans(text: str) -> tuple[_IdentifierSpan, ...]:
    return tuple(
        _IdentifierSpan(match.group(0), match.start(), match.end())
        for match in _IDENTIFIER_PATTERN.finditer(text)
    )


def _clean_inline(text: str) -> str:
    return re.sub(r"[`*_\[\](){}<>]", " ", text)


def _singularize(word: str) -> str:
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _normalize_key(text: str) -> str:
    words = [
        _singularize(match.group(0).casefold())
        for match in _WORD_PATTERN.finditer(_clean_inline(text).replace("/", " "))
    ]
    useful = [word for word in words if word not in _STOP_SUBJECT_WORDS]
    return " ".join(useful)


def _normalize_property(text: str) -> str:
    words = [
        _singularize(match.group(0).casefold())
        for match in _WORD_PATTERN.finditer(_clean_inline(text).replace("/", " "))
    ]
    useful = [word for word in words if word not in _PROPERTY_STOP_WORDS]
    return " ".join(useful) or "state"


def _line_payload(text: str) -> tuple[str, int]:
    match = _LEADING_FIELD.match(text.strip())
    if match is None:
        stripped = text.lstrip()
        return stripped, len(text) - len(stripped)
    stripped_prefix = text[: len(text) - len(text.lstrip())]
    leading = len(stripped_prefix) + match.end()
    return text[leading:].lstrip(), leading + len(text[leading:]) - len(text[leading:].lstrip())


def _clause_spans(text: str) -> tuple[_ClauseSpan, ...]:
    clauses: list[_ClauseSpan] = []
    in_fence = False
    offset = 0
    for line in text.splitlines(keepends=True):
        line_without_newline = line.rstrip("\r\n")
        stripped = line_without_newline.strip()
        fence = stripped.startswith("```") or stripped.startswith("~~~")
        if fence:
            in_fence = not in_fence
            offset += len(line)
            continue
        if in_fence or not stripped or stripped.startswith("#"):
            offset += len(line)
            continue
        payload, payload_offset = _line_payload(line_without_newline)
        base = offset + payload_offset
        for part in re.finditer(r"[^.!?;]+(?:[.!?;]+|$)", payload):
            raw = part.group(0)
            start = part.start()
            end = part.end()
            while start < end and raw[start - part.start()].isspace():
                start += 1
            while end > start and raw[end - part.start() - 1].isspace():
                end -= 1
            if end > start:
                clauses.append(_ClauseSpan(payload[start:end], base + start, base + end))
        offset += len(line)
    return tuple(clauses)


def _markers(text: str) -> tuple[_PolarityMarker, ...]:
    specs: tuple[tuple[str, int, str, re.Pattern[str]], ...] = (
        ("modal", -1, "MUST NOT", _MODAL_NEGATIVE),
        ("modal", 1, "MUST", _MODAL_POSITIVE),
        ("state", 1, "enabled", _ENABLED),
        ("state", -1, "disabled", _DISABLED),
        ("permission", 1, "allowed", _ALLOWED),
        ("permission", -1, "prohibited", _PROHIBITED),
        ("requirement", 1, "required", _REQUIRED),
        ("requirement", -1, "forbidden", _FORBIDDEN),
    )
    found: list[_PolarityMarker] = []
    occupied: list[tuple[int, int]] = []
    for family, polarity, label, pattern in specs:
        for match in pattern.finditer(text):
            span = (match.start(), match.end())
            if any(
                span[0] < used_end and span[1] > used_start for used_start, used_end in occupied
            ):
                continue
            found.append(_PolarityMarker(family, polarity, label, span[0], span[1]))
            occupied.append(span)
    return tuple(sorted(found, key=lambda item: (item.start, item.end, item.label)))


def _nearest_identifier_before(
    identifiers: Sequence[_IdentifierSpan], marker: _PolarityMarker
) -> _IdentifierSpan | None:
    before = [item for item in identifiers if item.end <= marker.start]
    if not before:
        return None
    candidate = before[-1]
    if marker.start - candidate.end > _SUBJECT_WINDOW_CHARS:
        return None
    return candidate


def _heading_subject(source: SourceItem, marker: _PolarityMarker, clause_text: str) -> str | None:
    if clause_text[: marker.start].strip():
        return None
    heading_ids = _identifiers(" ".join(source.provenance.heading_path))
    if len(heading_ids) == 1:
        return next(iter(heading_ids))
    return None


def _word_before(text: str, end: int) -> str | None:
    words = [match.group(0) for match in _WORD_PATTERN.finditer(text[:end])]
    for word in reversed(words):
        key = _normalize_key(word)
        if key:
            return word
    return None


def _word_after(text: str, start: int) -> str | None:
    for match in _WORD_PATTERN.finditer(text[start:]):
        word = match.group(0)
        if _normalize_key(word):
            return word
    return None


def _modal_common_subject(text: str, marker: _PolarityMarker) -> str | None:
    prefix = _clean_inline(text[: marker.start])
    prefix = re.split(r"\b(?:and|but|while|when)\b|[,()]", prefix)[-1]
    words = [match.group(0) for match in _WORD_PATTERN.finditer(prefix)]
    useful = [word for word in words if _normalize_key(word)]
    if not useful:
        return None
    return " ".join(useful[:3])


def _state_subject(text: str, marker: _PolarityMarker) -> str | None:
    before = _word_before(text, marker.start)
    after = _word_after(text, marker.end)
    if before and before.casefold() not in {"is", "are", "be", "been", "when", "if"}:
        return before
    return after


def _assertion_subject(
    source: SourceItem,
    clause: _ClauseSpan,
    marker: _PolarityMarker,
    identifiers: Sequence[_IdentifierSpan],
) -> str | None:
    identifier = _nearest_identifier_before(identifiers, marker)
    if identifier is not None:
        return identifier.text
    heading_subject = _heading_subject(source, marker, clause.text)
    if heading_subject is not None:
        return heading_subject
    if marker.family == "state":
        return _state_subject(clause.text, marker)
    return _modal_common_subject(clause.text, marker)


def _assertion_property(text: str, marker: _PolarityMarker, subject: str) -> str:
    if marker.family == "state":
        suffix = text[marker.end :]
        qualifier = _TRAILING_QUALIFIER.search(suffix)
        if qualifier is not None:
            return _normalize_property(qualifier.group(0))
        return "state"
    suffix = text[marker.end :]
    suffix = re.sub(r"^\s+(?:be|to|that)\b", " ", suffix, flags=re.I)
    subject_key = _normalize_key(subject)
    prop = _normalize_property(suffix)
    if prop == subject_key:
        return "state"
    return prop


def _span_ref(source: SourceItem, start: int, end: int) -> SourceRef:
    p = source.provenance
    text = source.text
    span_text = text[start:end]
    prefix = text[:start]
    through = text[:end]
    start_line = p.start_line + prefix.count("\n")
    end_line = p.start_line + through.count("\n")
    start_line_prefix = prefix.rsplit("\n", 1)[-1]
    end_line_prefix = through.rsplit("\n", 1)[-1]
    start_column = len(start_line_prefix)
    end_column = len(end_line_prefix)
    if start_line == p.start_line:
        start_column += p.start_column
    if end_line == p.start_line:
        end_column += p.start_column
    return SourceRef(
        document_id=p.document_id,
        document_path=p.document_path,
        section_id=p.section_id,
        heading_path=p.heading_path,
        start_line=start_line,
        end_line=end_line,
        start_column=start_column,
        end_column=end_column,
        section_sha256=p.section_sha256,
        range_sha256=hashlib.sha256(span_text.encode("utf-8")).hexdigest(),
        authority=p.authority,
        priority=p.priority,
        index_generation=p.index_generation,
    )


def _assertions(source: SourceItem) -> tuple[_Assertion, ...]:
    assertions: list[_Assertion] = []
    for clause in _clause_spans(source.text):
        identifiers = _identifier_spans(clause.text)
        for marker in _markers(clause.text):
            subject = _assertion_subject(source, clause, marker, identifiers)
            if subject is None:
                continue
            subject_key = _normalize_key(subject)
            if not subject_key:
                continue
            property_key = _assertion_property(clause.text, marker, subject)
            assertions.append(
                _Assertion(
                    subject=subject,
                    subject_key=subject_key,
                    property_key=property_key,
                    family=marker.family,
                    polarity=marker.polarity,
                    marker_label=marker.label,
                    source_ref=_span_ref(source, clause.start, clause.end),
                )
            )
    return tuple(assertions)


def _conflict_reason(left: _Assertion, right: _Assertion) -> str | None:
    if left.family != right.family or left.polarity == right.polarity:
        return None
    if left.subject_key != right.subject_key:
        return None
    if left.property_key != right.property_key:
        return None
    return (
        "deterministic contradictory assertion: "
        f"subject={left.subject_key!r}; property={left.property_key!r}; "
        f"markers={left.marker_label}/{right.marker_label}"
    )


def _possible_conflicts(items: Sequence[ContextPackItem]) -> tuple[PossibleConflict, ...]:
    conflicts: list[PossibleConflict] = []
    assertions_by_item = tuple(_assertions(item.source) for item in items)
    seen: set[tuple[str, str, str, str, str]] = set()
    for index, left_assertions in enumerate(assertions_by_item):
        for right_assertions in assertions_by_item[index + 1 :]:
            for left in left_assertions:
                for right in right_assertions:
                    reason = _conflict_reason(left, right)
                    if reason is None:
                        continue
                    key = (
                        left.subject_key,
                        left.property_key,
                        left.source_ref.range_sha256,
                        right.source_ref.range_sha256,
                        reason,
                    )
                    if key in seen:
                        continue
                    seen.add(key)
                    conflicts.append(
                        PossibleConflict(
                            identifier=left.subject,
                            reason=reason,
                            sources=(left.source_ref, right.source_ref),
                        )
                    )
                    if len(conflicts) >= 10:
                        return tuple(conflicts)
    return tuple(conflicts)


def _counter_name(counter: TokenCounter) -> str:
    return counter.identity


def _source_refs_overlap(left: SourceRef, right: SourceRef) -> bool:
    return (
        left.section_id == right.section_id
        and left.start_line <= right.end_line
        and right.start_line <= left.end_line
    )


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
                reason=(
                    "required evidence did not fit the context-pack budget; " + candidate.reason
                ),
                confidence=candidate.confidence,
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

    conflict_sources = tuple(source for conflict in conflicts for source in conflict.sources)
    conflict_hashes = {source.range_sha256 for source in conflict_sources}
    base_categories = (
        CoverageCategory.PRIMARY,
        CoverageCategory.DEPENDENCIES,
        CoverageCategory.ARCHITECTURE,
        CoverageCategory.SECURITY,
        CoverageCategory.ACCEPTANCE,
        CoverageCategory.VERIFICATION,
        CoverageCategory.CHECKPOINT_DESCENDANTS,
    )
    applicable_categories = {
        category for candidate in candidates for category in candidate.coverage_categories
    }
    applicable_categories.update(item.category for item in unresolved)
    applicable_categories.update(item.category for item in ambiguous)
    categories = (
        *base_categories,
        *(
            item
            for item in CoverageCategory
            if item in applicable_categories and item not in base_categories
        ),
    )
    coverage_records: list[CategoryCoverage] = []
    for category in categories:
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
        category_conflicting = any(
            item.range_sha256 in conflict_hashes
            or any(
                _source_refs_overlap(item, conflict_source) for conflict_source in conflict_sources
            )
            for item in evidence
        )
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
        for category in (
            CoverageCategory.PRIMARY,
            CoverageCategory.DEPENDENCIES,
            CoverageCategory.ARCHITECTURE,
            CoverageCategory.SECURITY,
            CoverageCategory.ACCEPTANCE,
            CoverageCategory.VERIFICATION,
            CoverageCategory.CHECKPOINT_DESCENDANTS,
        )
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


@dataclass(frozen=True)
class _DiscoverySignal:
    query: str
    origins: tuple[str, ...]


@dataclass
class _DiscoveredSection:
    source: SourceItem
    category: CoverageCategory
    candidate_category: str
    recognized_category: bool
    optional: bool
    best_rank: int
    signal_origins: set[str]
    signal_queries: set[str]
    channels: set[str]
    overlap_terms: set[str]
    semantic_score: float


# Conservative cosine gates keep a role-named normative document from becoming required merely
# because every vector search has a top result. Repeated independent structured signals permit a
# slightly lower score; direct lexical/identifier overlap remains independently material.
_SEMANTIC_MATERIAL_SCORE = 0.50
_REPEATED_SEMANTIC_SCORE = 0.42
_GENERIC_SEMANTIC_SCORE = 0.64
_GENERIC_REPEATED_SCORE = 0.52

_DISCOVERY_STOPWORDS = {
    "about",
    "after",
    "also",
    "before",
    "being",
    "criteria",
    "during",
    "exactly",
    "from",
    "goal",
    "implement",
    "implementation",
    "into",
    "must",
    "only",
    "required",
    "requires",
    "shall",
    "should",
    "that",
    "their",
    "then",
    "this",
    "through",
    "using",
    "verify",
    "when",
    "with",
    "without",
}


def _meaningful_terms(text: str) -> set[str]:
    return {
        term.casefold()
        for term in re.findall(r"[A-Za-z][A-Za-z0-9_.@/-]{3,}", text)
        if term.casefold() not in _DISCOVERY_STOPWORDS
    }


def _bounded_signal(value: str) -> str:
    compact = " ".join(value.split())
    return compact[:512].rstrip()


def _discovery_signals(task: str, checkpoints: Sequence[Any]) -> tuple[_DiscoverySignal, ...]:
    rows: list[_DiscoverySignal] = []

    def add(value: str, origin: str) -> None:
        query = _bounded_signal(value)
        if len(_meaningful_terms(query)) < 1:
            return
        key = query.casefold()
        if any(item.query.casefold() == key for item in rows):
            return
        rows.append(_DiscoverySignal(query=query, origins=(origin,)))

    add(task, "task")
    field_order = (
        "goal",
        "why",
        "exact_scope",
        "dependencies",
        "files_modules",
        "interfaces_models",
        "cli_behavior",
        "tests_acceptance_criteria",
        "failure_conditions",
        "security",
        "verify",
    )
    for checkpoint in checkpoints:
        metadata = checkpoint.metadata
        add(metadata.title, "checkpoint_title")
        for field in field_order:
            for value in metadata.fields.get(field, ()):
                add(value, field)
        for source in checkpoint.sources:
            for line in source.text.splitlines():
                stripped = re.sub(r"^[#>*+\-\d.()\s]+", "", line).strip(" `")
                if re.search(
                    r"(?i)\b(?:must|shall|required|requires|never|only|cannot|ensure|retain|"
                    r"reject|refuse|preserve)\b",
                    stripped,
                ):
                    add(stripped, "normative_requirement")
    # The bound is deterministic and favors the structured evidence order above.
    return tuple(rows[:16])


def _discovery_category(
    source: SourceItem,
) -> tuple[CoverageCategory, str, bool, bool]:
    provenance = source.provenance
    path = provenance.document_path.casefold()
    heading = " ".join(provenance.heading_path).casefold()
    label = f"{path} {heading}"
    optional = bool(
        re.search(
            r"\b(?:background|rationale|overview|history|historical|appendix|example|"
            r"research|notes|glossary|archive)\b",
            label,
        )
    )
    categories = (
        (
            r"\b(?:decision|decisions|adr[-_ ]?\d*|rfc[-_ ]?\d*)\b",
            CoverageCategory.DECISIONS,
            "decision_constraint",
        ),
        (
            r"\b(?:progress|current[-_ ]state|delivery[-_ ]state|"
            r"implementation[-_ ]state|status)\b",
            CoverageCategory.CURRENT_STATE,
            "current_state",
        ),
        (
            r"\b(?:architecture|architectural|design|topology|interface|model)\b",
            CoverageCategory.ARCHITECTURE,
            "interface_or_model",
        ),
        (
            r"\b(?:security|secure|threat|trust[-_ ]boundary|vulnerability)\b",
            CoverageCategory.SECURITY,
            "security_constraint",
        ),
        (
            r"\b(?:acceptance|acceptance[-_ ]criteria)\b",
            CoverageCategory.ACCEPTANCE,
            "acceptance_or_verify",
        ),
        (
            r"\b(?:verification|verify|validation)\b",
            CoverageCategory.VERIFICATION,
            "acceptance_or_verify",
        ),
        (
            r"\b(?:testing|tests?|quality|qualification|qa)\b",
            CoverageCategory.TESTING,
            "testing_constraint",
        ),
        (
            r"\b(?:dependencies|dependency|prerequisite)\b",
            CoverageCategory.DEPENDENCIES,
            "explicit_dependency",
        ),
    )
    for pattern, coverage, category in categories:
        if re.search(pattern, label):
            return coverage, category, True, optional
    recognized = bool(
        re.search(r"\b(?:constraint|requirement|contract|policy|rule|mandate)\b", heading)
    )
    return CoverageCategory.NORMATIVE, "normative_constraint", recognized, optional


def _strict_discovery(
    engine: ContextEngine,
    task: str,
    checkpoints: Sequence[Any],
    excluded_section_ids: set[str],
) -> tuple[_DiscoveredSection, ...]:
    signals = _discovery_signals(task, checkpoints)
    discovered: dict[str, _DiscoveredSection] = {}
    source_terms: dict[str, set[str]] = {}
    focused_limit = min(6, engine.config.limits.max_results)
    global_limit = engine.config.limits.max_results
    normative_documents = sorted(
        document.path
        for document in engine.config.documents
        if document.authority >= Authority.NORMATIVE
    )

    def collect(
        signal: _DiscoverySignal,
        *,
        limit: int,
        documents: set[str] | None = None,
    ) -> None:
        signal_terms = _meaningful_terms(signal.query)
        semantic_hits = engine.search_semantic(
            signal.query,
            limit=limit,
            authority_floor=Authority.NORMATIVE,
            documents=documents,
            require_semantic=True,
        )
        semantic_scores: dict[str, float] = {}
        for semantic_hit in semantic_hits:
            section_id = semantic_hit.source.provenance.section_id
            semantic_scores[section_id] = max(
                semantic_scores.get(section_id, -1.0), semantic_hit.score
            )
        hits = engine.search(
            signal.query,
            limit=limit,
            authority_floor=Authority.NORMATIVE,
            documents=documents,
            require_semantic=True,
        )
        ranked_hits = list(enumerate(hits, start=1))
        hybrid_section_ids = {hit.source.provenance.section_id for hit in hits}
        ranked_hits.extend(
            (rank, hit)
            for rank, hit in enumerate(semantic_hits, start=1)
            if hit.source.provenance.section_id not in hybrid_section_ids
        )
        for rank, hit in ranked_hits:
            provenance = hit.source.provenance
            section_id = provenance.section_id
            if section_id in excluded_section_ids:
                continue
            existing = discovered.get(section_id)
            source = existing.source if existing is not None else engine.get_section(section_id)
            terms = source_terms.get(section_id)
            if terms is None:
                terms = _meaningful_terms(source.text)
                source_terms[section_id] = terms
            overlap = signal_terms & terms
            semantic_score = semantic_scores.get(section_id, -1.0)
            if existing is None:
                coverage, category, recognized, optional = _discovery_category(source)
                discovered[section_id] = _DiscoveredSection(
                    source=source,
                    category=coverage,
                    candidate_category=category,
                    recognized_category=recognized,
                    optional=optional,
                    best_rank=rank,
                    signal_origins=set(signal.origins),
                    signal_queries={signal.query},
                    channels=set(hit.channels),
                    overlap_terms=set(overlap),
                    semantic_score=semantic_score,
                )
            else:
                existing.best_rank = min(existing.best_rank, rank)
                existing.signal_origins.update(signal.origins)
                existing.signal_queries.add(signal.query)
                existing.channels.update(hit.channels)
                existing.overlap_terms.update(overlap)
                existing.semantic_score = max(existing.semantic_score, semantic_score)

    # Four deterministic global query groups retain every structured signal without multiplying
    # search cost by every document. One compact per-document probe ensures a large high-scoring
    # document cannot hide another configured normative document behind the global cutoff.
    grouped_signals: list[_DiscoverySignal] = []
    for start in range(0, len(signals), 4):
        group = signals[start : start + 4]
        grouped_signals.append(
            _DiscoverySignal(
                query=_bounded_signal(" ".join(item.query for item in group)),
                origins=tuple(origin for item in group for origin in item.origins),
            )
        )
    for signal in grouped_signals:
        collect(signal, limit=global_limit)
    focused_signal = _DiscoverySignal(
        query=_bounded_signal(" ".join(item.query[:28] for item in signals)),
        origins=tuple(origin for item in signals for origin in item.origins),
    )
    for document_path in normative_documents:
        if focused_signal.query:
            collect(focused_signal, limit=focused_limit, documents={document_path})

    material: list[_DiscoveredSection] = []
    for value in discovered.values():
        structurally_related = bool(value.overlap_terms)
        semantic_material = value.semantic_score >= _SEMANTIC_MATERIAL_SCORE
        repeated_semantic = (
            value.semantic_score >= _REPEATED_SEMANTIC_SCORE and len(value.signal_queries) >= 2
        )
        if value.optional:
            if semantic_material or structurally_related:
                material.append(value)
            continue
        if value.recognized_category and (
            semantic_material or repeated_semantic or structurally_related
        ):
            material.append(value)
            continue
        if (
            len(value.overlap_terms) >= 2
            or value.semantic_score >= _GENERIC_SEMANTIC_SCORE
            or (value.semantic_score >= _GENERIC_REPEATED_SCORE and len(value.signal_queries) >= 2)
        ):
            material.append(value)

    # Retrieval is bounded per signal/document. Every item that passes deterministic materiality
    # classification is required; only explicitly optional categories may be dropped harmlessly.
    material.sort(
        key=lambda item: (
            item.optional,
            list(CoverageCategory).index(item.category),
            -len(item.signal_queries),
            item.best_rank,
            -len(item.overlap_terms),
            item.source.provenance.document_path,
            item.source.provenance.start_line,
            item.source.provenance.section_id,
        )
    )
    return tuple(material)


def build_context_pack(
    engine: ContextEngine,
    task: str,
    token_budget: int,
    *,
    filters: FilterSet | None = None,
    counter: TokenCounter | None = None,
    allow_required_budget_expansion: bool = False,
    strict_agent: bool = False,
    require_semantic: bool = False,
    checkpoint_document: str | None = None,
) -> ContextPack:
    if token_budget < 1:
        raise ValueError("token_budget must be positive")
    if token_budget > engine.config.limits.max_token_budget:
        raise ValueError("token_budget exceeds configured maximum")
    token_counter = counter or ApproximateGenericCounter()
    filters = filters or FilterSet()
    if strict_agent and not require_semantic:
        require_semantic = True
    semantic_active = False
    if engine.embedder is not None:
        try:
            engine.require_semantic_ready()
            semantic_active = True
        except RuntimeError:
            if require_semantic:
                raise
    elif require_semantic:
        engine.require_semantic_ready()
    generation = engine.store.index_generation()
    classification = classify_query(task)
    context_mode = ContextMode.STRICT_AGENT if strict_agent else ContextMode.STANDARD
    retrieval_mode = (
        RetrievalMode.HYBRID_SEMANTIC if semantic_active else RetrievalMode.LEXICAL_ONLY
    )
    active_channels = (
        ["structural", "lexical", "semantic"] if semantic_active else ["structural", "lexical"]
    )
    retrieval_metadata: dict[str, str | int | bool | list[str]] = {
        "strategy": "evidence-first+graph+hybrid",
        "context_mode": context_mode.value,
        "retrieval_mode": retrieval_mode.value,
        "require_semantic": require_semantic,
        "active_channels": active_channels,
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
    checkpoint_results: list[Any] = []
    for checkpoint_id in primary_ids:
        document_filter = checkpoint_document
        if document_filter is None and filters.documents and len(filters.documents) == 1:
            document_filter = next(iter(filters.documents))
        checkpoint = engine.get_checkpoint(checkpoint_id, document=document_filter)
        checkpoint_results.append(checkpoint)
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

    direct = engine.search(
        task,
        limit=min(12, engine.config.limits.max_results),
        require_semantic=require_semantic,
        _semantic_enabled=semantic_active,
        **filter_kwargs,
    )
    uncovered_terms = {
        term.casefold()
        for term in re.findall(r"[\w.@/:-]+", task, flags=re.UNICODE)
        if len(term) > 2 and term.casefold() not in {"implement", "using", "with", "from"}
    }
    checkpoint_source_ids = {
        source.provenance.section_id
        for checkpoint in checkpoint_results
        for source in checkpoint.sources
    }
    for rank, hit in enumerate(direct, start=1):
        if (
            strict_agent
            and primary_ids
            and hit.source.provenance.section_id not in checkpoint_source_ids
        ):
            # Strict cross-document evidence is admitted only after deterministic materiality
            # classification below; raw hybrid neighbors cannot masquerade as required context.
            continue
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

    if strict_agent:
        excluded_section_ids = {
            source.provenance.section_id
            for checkpoint in checkpoint_results
            for source in checkpoint.sources
        }
        for discovered in _strict_discovery(engine, task, checkpoint_results, excluded_section_ids):
            provenance = discovered.source.provenance
            confidence = min(
                0.99,
                0.68
                + 0.04 * min(len(discovered.signal_queries), 4)
                + 0.03 * min(len(discovered.overlap_terms), 3)
                + 0.08 * max(0.0, min(discovered.semantic_score, 1.0))
                + (0.08 if discovered.best_rank <= 3 else 0.0),
            )
            query_sample = sorted(discovered.signal_queries, key=lambda value: (len(value), value))[
                0
            ]
            reason = (
                "strict cross-document discovery; "
                f"category={discovered.category.value}; "
                f"signals={','.join(sorted(discovered.signal_origins))}; "
                f"query={query_sample[:160]!r}; "
                f"channels={','.join(sorted(discovered.channels))}; "
                f"semantic_score={discovered.semantic_score:.4f}; "
                f"best_rank={discovered.best_rank}"
            )
            required = not discovered.optional
            accepted = add(
                discovered.source,
                discovered.candidate_category,
                reason,
                relevance=(
                    10.0 * len(discovered.signal_queries)
                    + len(discovered.overlap_terms)
                    - discovered.best_rank / 100.0
                ),
                confidence=confidence,
                required=required,
                coverage_categories=(discovered.category,),
                checkpoint_id=primary_ids[0].upper() if primary_ids else None,
            )
            if required and not accepted:
                unresolved_required.append(
                    OmittedRequiredEvidence(
                        category=discovered.category,
                        reason=(
                            "required strict cross-document discovery evidence was excluded "
                            f"by context-pack filters; {reason}"
                        ),
                        confidence=confidence,
                        document_id=provenance.document_id,
                        document_path=provenance.document_path,
                        section_id=provenance.section_id,
                        checkpoint_id=primary_ids[0].upper() if primary_ids else None,
                        start_line=provenance.start_line,
                        end_line=provenance.end_line,
                        range_sha256=provenance.range_sha256,
                    )
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
        for rank, hit in enumerate(
            engine.search(
                query,
                limit=3,
                require_semantic=require_semantic,
                _semantic_enabled=semantic_active,
                **filter_kwargs,
            ),
            start=1,
        ):
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
            not candidate.required
            and excerpt is not None
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
