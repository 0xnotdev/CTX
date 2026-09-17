from pathlib import Path

import pytest

from ctx.config import add_document_config, initialize_workspace
from ctx.models import Authority
from ctx.service import ContextEngine


@pytest.fixture
def indexed(tmp_path: Path) -> ContextEngine:
    initialize_workspace(tmp_path)
    (tmp_path / "spec.md").write_text(
        """# RunManifest
The RunManifest controls execution.
# State stabilization
StateStabilizationTimeout is INVALID_EVIDENCE when exceeded.
# Network rule
Call network.remove_ingress@1 exactly once.
# Compiler
The required operation is proof.contract.compile.
""",
        encoding="utf-8",
    )
    (tmp_path / "notes.md").write_text(
        "# Similar notes\nRun manifest and compiler background.\n", encoding="utf-8"
    )
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE, 5)
    add_document_config(tmp_path, "notes.md", Authority.INFORMAL, 100)
    engine = ContextEngine(tmp_path)
    engine.index_workspace()
    yield engine
    engine.close()


@pytest.mark.parametrize(
    ("query", "heading"),
    [
        ("RunManifest", "RunManifest"),
        ("StateStabilizationTimeout", "State stabilization"),
        ("network.remove_ingress@1", "Network rule"),
        ("INVALID_EVIDENCE", "State stabilization"),
        ("proof.contract.compile", "Compiler"),
    ],
)
def test_exact_technical_identifiers_rank_owner_first(
    indexed: ContextEngine, query: str, heading: str
) -> None:
    hits = indexed.search_lexical(query)
    assert hits
    assert hits[0].source.provenance.heading_path[-1] == heading
    assert query in hits[0].source.text
    assert hits[0].channels == ("bm25",)


def test_exact_heading_and_authority_priority_boosts(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "norm.md").write_text("# Retry policy\ncanonical rule\n", encoding="utf-8")
    (tmp_path / "generated.md").write_text(
        "# Retry policy\ncanonical rule generated copy\n", encoding="utf-8"
    )
    add_document_config(tmp_path, "norm.md", Authority.NORMATIVE, -100)
    add_document_config(tmp_path, "generated.md", Authority.GENERATED, 100)
    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        hits = engine.search_lexical("Retry policy")
        assert hits[0].source.provenance.document_path == "norm.md"
        assert hits[0].source.provenance.authority is Authority.NORMATIVE


def test_fts_query_syntax_is_inert(indexed: ContextEngine) -> None:
    indexed.search_lexical("' OR 1=1; DROP TABLE sections; --")
    assert indexed.store.section_ids()
