"""Reproducible channel-separated retrieval and context-pack evaluation."""

from __future__ import annotations

from dataclasses import dataclass

from ctx.models import StrictModel
from ctx.service import ContextEngine


@dataclass(frozen=True)
class RetrievalCase:
    name: str
    query: str
    expected_section_id: str


@dataclass(frozen=True)
class PackCase:
    task: str
    token_budget: int
    required_section_ids: frozenset[str]


class ChannelMetrics(StrictModel):
    query_count: int
    recall_at_1: float
    recall_at_3: float
    mrr: float


class EvaluationMetrics(StrictModel):
    query_count: int
    recall_at_1: float
    recall_at_3: float
    mrr: float
    exact_section_accuracy: float
    context_pack_required_section_recall: float
    context_pack_tokens_returned: int
    context_pack_serialized_tokens: int
    required_section_recall_per_thousand_tokens: float


class EvaluationReport(StrictModel):
    lexical: ChannelMetrics
    semantic: ChannelMetrics | None
    hybrid: ChannelMetrics
    packs: EvaluationMetrics
    corpus_characters: int
    returned_source_characters: int
    compression_ratio: float
    embedding_identity: str


def _channel_metrics(
    engine: ContextEngine,
    cases: list[RetrievalCase],
    channel: str,
) -> ChannelMetrics:
    reciprocal_ranks: list[float] = []
    hits_at_1 = 0
    hits_at_3 = 0
    for case in cases:
        if channel == "lexical":
            results = engine.search_lexical(case.query, limit=3)
        elif channel == "semantic":
            results = engine.search_semantic(case.query, limit=3)
        elif channel == "hybrid":
            results = engine.search(case.query, limit=3)
        else:  # pragma: no cover - internal contract
            raise ValueError(f"unknown evaluation channel: {channel}")
        identifiers = [hit.source.provenance.section_id for hit in results]
        if identifiers and identifiers[0] == case.expected_section_id:
            hits_at_1 += 1
        if case.expected_section_id in identifiers:
            hits_at_3 += 1
            reciprocal_ranks.append(1.0 / (identifiers.index(case.expected_section_id) + 1))
        else:
            reciprocal_ranks.append(0.0)
    count = len(cases)
    return ChannelMetrics(
        query_count=count,
        recall_at_1=hits_at_1 / count if count else 1.0,
        recall_at_3=hits_at_3 / count if count else 1.0,
        mrr=sum(reciprocal_ranks) / count if count else 1.0,
    )


def evaluate(
    engine: ContextEngine,
    retrieval_cases: list[RetrievalCase],
    pack_cases: list[PackCase],
) -> EvaluationMetrics:
    """Compatibility aggregate: hybrid retrieval plus exact serialized pack use."""
    hybrid = _channel_metrics(engine, retrieval_cases, "hybrid")
    required_total = 0
    required_found = 0
    tokens = 0
    for pack_case in pack_cases:
        pack = engine.get_context_pack(pack_case.task, pack_case.token_budget)
        returned = {item.source.provenance.section_id for item in pack.items}
        required_total += len(pack_case.required_section_ids)
        required_found += len(pack_case.required_section_ids & returned)
        tokens += pack.serialized_estimated_tokens
    pack_recall = required_found / required_total if required_total else 1.0
    per_thousand = pack_recall / (tokens / 1_000) if tokens else 0.0
    return EvaluationMetrics(
        query_count=hybrid.query_count,
        recall_at_1=hybrid.recall_at_1,
        recall_at_3=hybrid.recall_at_3,
        mrr=hybrid.mrr,
        exact_section_accuracy=hybrid.recall_at_1,
        context_pack_required_section_recall=pack_recall,
        context_pack_tokens_returned=tokens,
        context_pack_serialized_tokens=tokens,
        required_section_recall_per_thousand_tokens=per_thousand,
    )


def evaluate_report(
    engine: ContextEngine,
    retrieval_cases: list[RetrievalCase],
    pack_cases: list[PackCase],
) -> EvaluationReport:
    lexical = _channel_metrics(engine, retrieval_cases, "lexical")
    semantic = _channel_metrics(engine, retrieval_cases, "semantic") if engine.embedder else None
    hybrid = _channel_metrics(engine, retrieval_cases, "hybrid")
    packs = evaluate(engine, retrieval_cases, pack_cases)
    corpus = sum(
        len(section.text)
        for document in engine.store.list_documents()
        for section in engine.store.document_sections(document.path)
    )
    returned = 0
    for case in pack_cases:
        returned += sum(
            len(item.source.text)
            for item in engine.get_context_pack(case.task, case.token_budget).items
        )
    return EvaluationReport(
        lexical=lexical,
        semantic=semantic,
        hybrid=hybrid,
        packs=packs,
        corpus_characters=corpus,
        returned_source_characters=returned,
        compression_ratio=(1.0 - returned / corpus) if corpus else 1.0,
        embedding_identity=engine.embedder.identity if engine.embedder else "none",
    )
