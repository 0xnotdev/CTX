"""Small reproducible retrieval/context-pack evaluation helpers."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

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


class EvaluationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    query_count: int
    recall_at_1: float
    recall_at_3: float
    mrr: float
    exact_section_accuracy: float
    context_pack_required_section_recall: float
    context_pack_tokens_returned: int
    required_section_recall_per_thousand_tokens: float


def evaluate(
    engine: ContextEngine,
    retrieval_cases: list[RetrievalCase],
    pack_cases: list[PackCase],
) -> EvaluationMetrics:
    """Evaluate deterministic section IDs; no answer-generation metric is involved."""
    reciprocal_ranks: list[float] = []
    hits_at_1 = 0
    hits_at_3 = 0
    for retrieval_case in retrieval_cases:
        results = engine.search(retrieval_case.query, limit=3)
        identifiers = [hit.source.provenance.section_id for hit in results]
        if identifiers and identifiers[0] == retrieval_case.expected_section_id:
            hits_at_1 += 1
        if retrieval_case.expected_section_id in identifiers:
            hits_at_3 += 1
            rank = identifiers.index(retrieval_case.expected_section_id) + 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

    required_total = 0
    required_found = 0
    tokens = 0
    for pack_case in pack_cases:
        pack = engine.get_context_pack(pack_case.task, pack_case.token_budget)
        returned = {item.source.provenance.section_id for item in pack.items}
        required_total += len(pack_case.required_section_ids)
        required_found += len(pack_case.required_section_ids & returned)
        tokens += pack.estimated_tokens

    query_count = len(retrieval_cases)
    pack_recall = required_found / required_total if required_total else 1.0
    per_thousand = pack_recall / (tokens / 1_000) if tokens else 0.0
    return EvaluationMetrics(
        query_count=query_count,
        recall_at_1=hits_at_1 / query_count if query_count else 1.0,
        recall_at_3=hits_at_3 / query_count if query_count else 1.0,
        mrr=sum(reciprocal_ranks) / query_count if query_count else 1.0,
        exact_section_accuracy=hits_at_1 / query_count if query_count else 1.0,
        context_pack_required_section_recall=pack_recall,
        context_pack_tokens_returned=tokens,
        required_section_recall_per_thousand_tokens=per_thousand,
    )
