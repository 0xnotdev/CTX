"""Deterministic query classification and reciprocal-rank fusion."""

from __future__ import annotations

import re
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from ctx.models import Authority, SearchHit, SourceItem

RRF_K = 60


class QueryClassification(BaseModel):
    model_config = ConfigDict(frozen=True)

    checkpoint_ids: tuple[str, ...] = ()
    section_marks: tuple[str, ...] = ()
    camel_case: tuple[str, ...] = ()
    snake_case: tuple[str, ...] = ()
    dotted_or_versioned: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    quoted_phrases: tuple[str, ...] = ()

    @property
    def structural_terms(self) -> tuple[str, ...]:
        ordered = (
            *self.checkpoint_ids,
            *self.section_marks,
            *self.camel_case,
            *self.snake_case,
            *self.dotted_or_versioned,
            *self.paths,
            *self.quoted_phrases,
        )
        return tuple(dict.fromkeys(ordered))


def classify_query(query: str) -> QueryClassification:
    """Classify syntax that should be matched structurally before fuzzier channels."""
    checkpoint_ids = tuple(re.findall(r"\bCP-\d+\b", query, flags=re.IGNORECASE))
    section_marks = tuple(re.findall(r"§\s*\d+(?:\.\d+)*", query))
    words = re.findall(r"\b[A-Za-z][A-Za-z0-9_]*\b", query)
    camel = tuple(word for word in words if re.search(r"[a-z][A-Z]|[A-Z][a-z]+[A-Z]", word))
    snake = tuple(word for word in words if "_" in word)
    dotted = tuple(re.findall(r"\b[A-Za-z_][\w-]*(?:\.[\w-]+)+(?:@\d+)?\b|\b[\w.-]+@\d+\b", query))
    paths = tuple(re.findall(r"(?:^|\s)([\w.-]+(?:/[\w.-]+)+)(?=$|\s|[,:;])", query))
    phrases = tuple(match[1] for match in re.findall(r"([\"'])(.+?)\1", query))
    return QueryClassification(
        checkpoint_ids=checkpoint_ids,
        section_marks=section_marks,
        camel_case=camel,
        snake_case=snake,
        dotted_or_versioned=dotted,
        paths=paths,
        quoted_phrases=phrases,
    )


def _is_direct(term: str, source: SourceItem) -> bool:
    heading = source.provenance.heading_path[-1] if source.provenance.heading_path else ""
    return term.casefold() in heading.casefold()


def fuse_ranked(
    query: str,
    classification: QueryClassification,
    channels: Sequence[tuple[str, Sequence[SearchHit]]],
    limit: int,
) -> list[SearchHit]:
    """Fuse channel ranks with RRF, deterministic source/authority boosts and tie-breaking."""
    sources: dict[str, SourceItem] = {}
    scores: dict[str, float] = {}
    names: dict[str, list[str]] = {}
    terms: dict[str, set[str]] = {}
    for channel_name, hits in channels:
        for rank, hit in enumerate(hits, start=1):
            section_id = hit.source.provenance.section_id
            sources[section_id] = hit.source
            scores[section_id] = scores.get(section_id, 0.0) + 1.0 / (RRF_K + rank)
            names.setdefault(section_id, []).append(channel_name)
            terms.setdefault(section_id, set()).update(hit.matched_terms)

    query_folded = query.strip().casefold()
    result: list[SearchHit] = []
    for section_id, source in sources.items():
        heading = source.provenance.heading_path[-1] if source.provenance.heading_path else ""
        score = scores[section_id]
        if heading.strip().casefold() == query_folded:
            score += 4.0
        direct_terms = [
            term for term in classification.structural_terms if _is_direct(term, source)
        ]
        if direct_terms:
            score += 2.0 + 0.1 * len(direct_terms)
            if any(re.fullmatch(r"CP-\d+", term, re.IGNORECASE) for term in direct_terms):
                score += 10.0
        score += int(source.provenance.authority) * 0.01
        score += source.provenance.priority * 0.00001
        result.append(
            SearchHit(
                source=source,
                score=score,
                channels=tuple(dict.fromkeys(names[section_id])),
                matched_terms=tuple(sorted(terms.get(section_id, set()) | set(direct_terms))),
            )
        )

    # Navigation generated from sources can assist but never outranks an original source hit.
    result.sort(
        key=lambda hit: (
            hit.source.provenance.authority is Authority.GENERATED,
            -hit.score,
            -int(hit.source.provenance.authority),
            -hit.source.provenance.priority,
            hit.source.provenance.document_path,
            hit.source.provenance.start_line,
            hit.source.provenance.section_id,
        )
    )
    return result[:limit]
