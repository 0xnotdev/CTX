"""Deterministic syntax classification and staged hybrid rank fusion."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ctx.heading import canonical_heading
from ctx.models import SearchHit, StrictModel

RRF_K = 60
MAX_SPANS_PER_SECTION = 3
RETRIEVAL_VERSION = "ctx-rrf-span-diversity:4"


class QueryClassification(StrictModel):
    checkpoint_ids: tuple[str, ...] = ()
    section_marks: tuple[str, ...] = ()
    camel_case: tuple[str, ...] = ()
    snake_case: tuple[str, ...] = ()
    kebab_case: tuple[str, ...] = ()
    namespace_symbols: tuple[str, ...] = ()
    dotted_or_versioned: tuple[str, ...] = ()
    file_symbols: tuple[str, ...] = ()
    cli_flags: tuple[str, ...] = ()
    error_codes: tuple[str, ...] = ()
    anchors: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    quoted_phrases: tuple[str, ...] = ()

    @property
    def structural_terms(self) -> tuple[str, ...]:
        return tuple(
            dict.fromkeys(
                (
                    *self.checkpoint_ids,
                    *self.section_marks,
                    *self.camel_case,
                    *self.snake_case,
                    *self.kebab_case,
                    *self.namespace_symbols,
                    *self.dotted_or_versioned,
                    *self.file_symbols,
                    *self.cli_flags,
                    *self.error_codes,
                    *self.anchors,
                    *self.paths,
                    *self.quoted_phrases,
                )
            )
        )


def classify_query(query: str) -> QueryClassification:
    checkpoint_ids = tuple(re.findall(r"\bCP-\d+\b", query, flags=re.IGNORECASE))
    section_marks = tuple(
        re.findall(r"§\s*\d+(?:\.\d+)*|\b(?:section|chapter)\s+\d+(?:\.\d+)*", query, re.I)
    )
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", query)
    camel = tuple(word for word in words if re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", word))
    snake = tuple(word for word in words if "_" in word and not word.isupper())
    kebab = tuple(re.findall(r"(?<!-)\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b(?!-)", query))
    namespaces = tuple(re.findall(r"\b[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+\b", query))
    file_symbols = tuple(re.findall(r"\b[\w./-]+\.py:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?\b", query))
    dotted = tuple(re.findall(r"\b[A-Za-z_][\w-]*(?:\.[\w-]+)+(?:@\d+)?\b|\b[\w.-]+@\d+\b", query))
    cli_flags = tuple(re.findall(r"(?<!\w)--[a-z0-9][a-z0-9-]*\b", query))
    error_codes = tuple(
        re.findall(r"\b(?:[A-Z][A-Z0-9]*_[A-Z0-9_]*\d[A-Z0-9_]*|[A-Z]+\d{2,})\b", query)
    )
    anchors = tuple(re.findall(r"#[a-z0-9][a-z0-9_-]*(?:-[a-z0-9_-]+)*", query, re.I))
    paths = tuple(re.findall(r"(?:^|\s)([\w.-]+(?:[/\\][\w.-]+)+)(?=$|\s|[,:;])", query))
    phrases = tuple(match[1] for match in re.findall(r"([\"'])(.+?)\1", query))
    return QueryClassification(
        checkpoint_ids=checkpoint_ids,
        section_marks=section_marks,
        camel_case=camel,
        snake_case=snake,
        kebab_case=kebab,
        namespace_symbols=namespaces,
        dotted_or_versioned=dotted,
        file_symbols=file_symbols,
        cli_flags=cli_flags,
        error_codes=error_codes,
        anchors=anchors,
        paths=paths,
        quoted_phrases=phrases,
    )


def _is_direct(term: str, hit: SearchHit) -> bool:
    heading = hit.source.provenance.heading_path[-1] if hit.source.provenance.heading_path else ""
    return canonical_heading(term.lstrip("#")) in canonical_heading(heading)


def _overlaps(left: SearchHit, right: SearchHit) -> bool:
    left_p = left.source.provenance
    right_p = right.source.provenance
    return (
        left_p.section_id == right_p.section_id
        and left_p.start_offset < right_p.end_offset
        and right_p.start_offset < left_p.end_offset
    )


def _near_duplicate(left: SearchHit, right: SearchHit) -> bool:
    left_p = left.source.provenance
    right_p = right.source.provenance
    if left_p.range_sha256 == right_p.range_sha256:
        return True
    if _overlaps(left, right):
        return True
    # Repeated boilerplate can be independently relevant at distant offsets in one huge section.
    if left_p.section_id == right_p.section_id:
        return False
    left_words = set(re.findall(r"[\w.@/:-]+", left.source.text.casefold(), flags=re.UNICODE))
    right_words = set(re.findall(r"[\w.@/:-]+", right.source.text.casefold(), flags=re.UNICODE))
    if not left_words or not right_words:
        return False
    return len(left_words & right_words) / len(left_words | right_words) >= 0.92


def diversify_spans(
    hits: Sequence[SearchHit],
    limit: int,
    *,
    max_spans_per_section: int = MAX_SPANS_PER_SECTION,
) -> list[SearchHit]:
    """Select deterministic, exact ranges without globally collapsing a large section.

    The first pass reserves room for distinct sections and documents.  The second admits up to
    three non-overlapping, independently ranked ranges from a section.  Any overlapping window,
    identical range hash, or >=92% token-set duplicate collapses to its higher-ranked peer.
    """
    if limit < 1:
        return []
    selected: list[SearchHit] = []
    deferred: list[SearchHit] = []
    per_section: dict[str, int] = {}
    per_document: dict[str, int] = {}

    def duplicate(hit: SearchHit) -> bool:
        return any(_near_duplicate(hit, existing) for existing in selected)

    def accept(hit: SearchHit) -> bool:
        section_id = hit.source.provenance.section_id
        if per_section.get(section_id, 0) >= max_spans_per_section or duplicate(hit):
            return False
        selected.append(hit)
        per_section[section_id] = per_section.get(section_id, 0) + 1
        document_id = hit.source.provenance.document_id
        per_document[document_id] = per_document.get(document_id, 0) + 1
        return True

    # Distinct sections first; an early document cap stops one document from consuming the
    # complete small top-k before another relevant document is considered.
    for hit in hits:
        provenance = hit.source.provenance
        if per_section.get(provenance.section_id, 0) or (
            per_document.get(provenance.document_id, 0) >= 2 and len(selected) < min(limit, 6)
        ):
            deferred.append(hit)
            continue
        accept(hit)
        if len(selected) >= limit:
            return selected

    for hit in deferred:
        if accept(hit) and len(selected) >= limit:
            break
    return selected


def _fusion_key(
    hit: SearchHit, representatives: dict[tuple[str, int, int], SearchHit]
) -> tuple[str, int, int]:
    provenance = hit.source.provenance
    for key, previous in representatives.items():
        if _overlaps(hit, previous):
            return key
    return (provenance.section_id, provenance.start_offset, provenance.end_offset)


def fuse_ranked(
    query: str,
    classification: QueryClassification,
    channels: Sequence[tuple[str, Sequence[SearchHit]]],
    limit: int,
) -> list[SearchHit]:
    """Fuse relevance first, then let authority break materially comparable results."""
    representatives: dict[tuple[str, int, int], SearchHit] = {}
    scores: dict[tuple[str, int, int], float] = {}
    names: dict[tuple[str, int, int], list[str]] = {}
    terms: dict[tuple[str, int, int], set[str]] = {}
    for channel_index, (channel_name, hits) in enumerate(channels):
        for rank, hit in enumerate(hits, start=1):
            key = _fusion_key(hit, representatives)
            previous = representatives.get(key)
            # Structural/lexical excerpts retain exact match location ahead of semantic fallbacks.
            if previous is None or (channel_index < 2 and previous.channels == ("semantic",)):
                representatives[key] = hit
            scores[key] = scores.get(key, 0.0) + 1.0 / (RRF_K + rank)
            names.setdefault(key, []).append(channel_name)
            terms.setdefault(key, set()).update(hit.matched_terms)

    query_folded = query.strip().casefold()
    staged: list[tuple[float, SearchHit]] = []
    for key, representative in representatives.items():
        source = representative.source
        heading = source.provenance.heading_path[-1] if source.provenance.heading_path else ""
        score = scores[key]
        if canonical_heading(heading) == canonical_heading(query_folded):
            score += 4.0
        direct_terms = [
            term for term in classification.structural_terms if _is_direct(term, representative)
        ]
        if direct_terms:
            score += 2.0 + 0.1 * len(direct_terms)
            if any(re.fullmatch(r"CP-\d+", term, re.IGNORECASE) for term in direct_terms):
                score += 10.0
        staged.append(
            (
                score,
                SearchHit(
                    source=source,
                    score=score,
                    channels=tuple(dict.fromkeys(names[key])),
                    matched_terms=tuple(sorted(terms.get(key, set()) | set(direct_terms))),
                    chunk_id=representative.chunk_id,
                    match_start_line=representative.match_start_line,
                    match_end_line=representative.match_end_line,
                    index_generation=representative.index_generation,
                ),
            )
        )

    # A 0.02 score bucket is "materially comparable" at ordinary RRF scale; exact/direct boosts
    # remain dominant. Authority and then configured priority decide within a bucket.
    staged.sort(
        key=lambda pair: (
            -round(pair[0] / 0.02),
            -int(pair[1].source.provenance.authority),
            -pair[1].source.provenance.priority,
            -pair[0],
            pair[1].source.provenance.document_path,
            pair[1].source.provenance.start_line,
            pair[1].source.provenance.section_id,
        )
    )

    return diversify_spans([hit for _, hit in staged], limit)
