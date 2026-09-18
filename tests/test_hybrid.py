from pathlib import Path

from ctx.config import add_document_config, initialize_workspace
from ctx.embeddings import HashEmbedding
from ctx.models import Authority
from ctx.retrieval import classify_query
from ctx.service import ContextEngine
from tests.source_fixtures import write_exact_source


def test_query_classifier_recognizes_structural_syntax() -> None:
    query = (
        "Implement CP-14 and §25.2 for RunManifest, retry_count, "
        'network.remove_ingress@1 in src/net/policy.py using "exact phrase"'
    )
    result = classify_query(query)
    assert result.checkpoint_ids == ("CP-14",)
    assert result.section_marks == ("§25.2",)
    assert "RunManifest" in result.camel_case
    assert "retry_count" in result.snake_case
    assert "network.remove_ingress@1" in result.dotted_or_versioned
    assert "src/net/policy.py" in result.paths
    assert result.quoted_phrases == ("exact phrase",)


def test_cp_direct_structural_match_beats_semantic_similarity(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "spec.md").write_text(
        """# CP-13 — Trial preparation
Repeated trials and reproducibility setup. CP-14 is referenced here.
# CP-14 — Repeated trials and reproducibility
The RunManifest records deterministic seeds and repeated trial evidence.
# Reproducibility discussion
Repeated experiments should preserve deterministic trial evidence and seeds.
""",
        encoding="utf-8",
    )
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    with ContextEngine(tmp_path, embedder=HashEmbedding(64)) as engine:
        engine.index_workspace()
        hits = engine.search("CP-14", limit=3)
        assert hits[0].source.provenance.heading_path[-1].startswith("CP-14 —")
        assert "structural" in hits[0].channels
        assert "bm25" in hits[0].channels
        assert "semantic" in hits[0].channels


def test_hybrid_returns_bounded_whole_sections_with_stable_order(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    write_exact_source(
        tmp_path / "a.md",
        "# Timeout policy\nStabilize network state before timeout failure.\n"
        "# Other\nUnrelated material.\n",
    )
    write_exact_source(
        tmp_path / "b.md",
        "# Generated timeout\nStabilize network state before timeout failure.\n",
    )
    add_document_config(tmp_path, "a.md", Authority.NORMATIVE)
    add_document_config(tmp_path, "b.md", Authority.GENERATED, 100)
    with ContextEngine(tmp_path, embedder=HashEmbedding(64)) as engine:
        engine.index_workspace()
        first = engine.search("stabilize state timeout", limit=2)
        second = engine.search("stabilize state timeout", limit=2)
        assert [hit.source.provenance.section_id for hit in first] == [
            hit.source.provenance.section_id for hit in second
        ]
        assert len(first) == 2
        assert first[0].source.provenance.authority is Authority.NORMATIVE
        assert first[0].source.text.startswith("# Timeout policy")
        assert first[0].source.text.endswith("failure.\n")
