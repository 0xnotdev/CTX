from pathlib import Path

from ctx.config import add_document_config, initialize_workspace
from ctx.embeddings import HashEmbedding
from ctx.evaluation import PackCase, RetrievalCase, evaluate
from ctx.models import Authority
from ctx.service import ContextEngine


def test_retrieval_and_pack_quality_thresholds(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "spec.md").write_text(
        """# NetworkPolicy
NetworkPolicy controls ingress and egress policy.
# RunManifest
Interface model RunManifest records deterministic seeds.
# CP-2 — Setup
Prepare RunManifest fixtures.
# CP-14 — Reproducibility
Dependencies: CP-2, RunManifest
Repeated trials preserve deterministic evidence.
# State stabilization
Network changes must settle before the stabilization timeout.
# Error handling
Return INVALID_EVIDENCE when proof material is incomplete.
# Security Policy
Scripts and hostile Markdown are inert and must never execute.
# Proof compiler
Invoke proof.contract.compile after validation.
# Tests and Verify
Acceptance requires three identical seeded runs. Verify with pytest.
""",
        encoding="utf-8",
    )
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    with ContextEngine(tmp_path, embedder=HashEmbedding(64)) as engine:
        engine.index_workspace()
        by_heading = {
            item.provenance.heading_path[-1]: item.provenance.section_id
            for item in engine.document_outline("spec.md")
        }
        cases = [
            RetrievalCase("exact heading", "NetworkPolicy", by_heading["NetworkPolicy"]),
            RetrievalCase("exact symbol", "RunManifest", by_heading["RunManifest"]),
            RetrievalCase(
                "paraphrase",
                "network changes settle before timeout",
                by_heading["State stabilization"],
            ),
            RetrievalCase(
                "cross-reference",
                "dependency CP-14",
                by_heading["CP-14 — Reproducibility"],
            ),
            RetrievalCase("checkpoint", "CP-14", by_heading["CP-14 — Reproducibility"]),
            RetrievalCase("error", "INVALID_EVIDENCE", by_heading["Error handling"]),
            RetrievalCase(
                "policy", "hostile scripts inert never execute", by_heading["Security Policy"]
            ),
        ]
        required = frozenset(
            {
                by_heading["CP-14 — Reproducibility"],
                by_heading["CP-2 — Setup"],
                by_heading["RunManifest"],
                by_heading["Security Policy"],
                by_heading["Tests and Verify"],
            }
        )
        metrics = evaluate(
            engine,
            cases,
            [PackCase("Implement CP-14 using RunManifest", 7_000, required)],
        )
        assert metrics.recall_at_1 >= 0.85
        assert metrics.recall_at_3 == 1.0
        assert metrics.mrr >= 0.90
        assert metrics.exact_section_accuracy >= 0.85
        assert metrics.context_pack_required_section_recall == 1.0
        assert metrics.context_pack_tokens_returned <= 7_000
        assert metrics.required_section_recall_per_thousand_tokens > 0
