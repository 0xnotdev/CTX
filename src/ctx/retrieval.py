"""Deterministic syntax classification and staged hybrid rank fusion."""

from __future__ import annotations

import re
from collections.abc import Sequence

from ctx.models import SearchHit, StrictModel

RRF_K = 60
RETRIEVAL_VERSION = "ctx-rrf-staged-authority:2"


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
    return term.casefold().lstrip("#") in heading.casefold()


def fuse_ranked(
    query: str,
    classification: QueryClassification,
    channels: Sequence[tuple[str, Sequence[SearchHit]]],
    limit: int,
) -> list[SearchHit]:
    """Fuse relevance first, then let authority break materially comparable results."""
    representatives: dict[str, SearchHit] = {}
    scores: dict[str, float] = {}
    names: dict[str, list[str]] = {}
    terms: dict[str, set[str]] = {}
    for channel_index, (channel_name, hits) in enumerate(channels):
        for rank, hit in enumerate(hits, start=1):
            section_id = hit.source.provenance.section_id
            previous = representatives.get(section_id)
            # Structural/lexical excerpts retain exact match location ahead of semantic fallbacks.
            if previous is None or (channel_index < 2 and previous.channels == ("semantic",)):
                representatives[section_id] = hit
            scores[section_id] = scores.get(section_id, 0.0) + 1.0 / (RRF_K + rank)
            names.setdefault(section_id, []).append(channel_name)
            terms.setdefault(section_id, set()).update(hit.matched_terms)

    query_folded = query.strip().casefold()
    staged: list[tuple[float, SearchHit]] = []
    for section_id, representative in representatives.items():
        source = representative.source
        heading = source.provenance.heading_path[-1] if source.provenance.heading_path else ""
        score = scores[section_id]
        if heading.strip().casefold() == query_folded:
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
                    channels=tuple(dict.fromkeys(names[section_id])),
                    matched_terms=tuple(sorted(terms.get(section_id, set()) | set(direct_terms))),
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

    # Modest early diversity: suppress adjacent/duplicate sections after exact relevance while
    # permitting at least two per document and filling from deferred candidates.
    selected: list[SearchHit] = []
    deferred: list[SearchHit] = []
    per_document: dict[str, int] = {}
    seen_hashes: set[str] = set()
    for _, hit in staged:
        path = hit.source.provenance.document_path
        duplicate = hit.source.provenance.range_sha256 in seen_hashes
        if (per_document.get(path, 0) >= 2 and len(selected) < min(limit, 6)) or duplicate:
            deferred.append(hit)
            continue
        selected.append(hit)
        per_document[path] = per_document.get(path, 0) + 1
        seen_hashes.add(hit.source.provenance.range_sha256)
        if len(selected) >= limit:
            return selected
    for hit in deferred:
        if len(selected) >= limit:
            break
        selected.append(hit)
    return selected
