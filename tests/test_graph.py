from pathlib import Path

from ctx.config import add_document_config, initialize_workspace
from ctx.models import Authority, EdgeType
from ctx.service import ContextEngine


def test_deterministic_graph_symbols_traversal_and_unresolved_targets(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "spec.md").write_text(
        """# Platform
## RunManifest
Interface model RunManifest records a trial.
## CP-2 — Setup
Goal: create the RunManifest.
## CP-14 — Repeated trials
Dependencies: CP-2, MissingType, CP-99
Use RunManifest and follow §25. See [security](#25-security).
# §25 Security
The INVALID_EVIDENCE error blocks unsafe proof.contract.compile.
""",
        encoding="utf-8",
    )
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        cp14 = engine.search("CP-14", limit=1)[0].source
        edges = engine.get_references(cp14.provenance.section_id)
        assert all(edge.edge.source_section_id != edge.edge.target_section_id for edge in edges)
        kinds = {edge.edge.edge_type for edge in edges}
        assert EdgeType.DEPENDS_ON in kinds
        assert EdgeType.REFERENCES in kinds
        assert EdgeType.USES_TYPE in kinds
        assert EdgeType.RELATED_SECTION in kinds
        assert EdgeType.CHILD_OF in kinds

        dependencies = engine.get_dependencies(cp14.provenance.section_id)
        by_label = {edge.edge.label: edge for edge in dependencies}
        assert by_label["CP-2"].edge.resolved
        assert by_label["CP-2"].target is not None
        assert by_label["CP-2"].target.provenance.heading_path[-1].startswith("CP-2")
        assert not by_label["CP-99"].edge.resolved
        assert by_label["CP-99"].target is None
        assert not by_label["MissingType"].edge.resolved

        symbols = engine.find_symbol("RunManifest")
        assert symbols
        assert any(item.source.provenance.heading_path[-1] == "RunManifest" for item in symbols)
        assert all(item.source.provenance.document_sha256 for item in symbols)


def test_parent_child_edges_are_bidirectional(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "tree.md").write_text("# Parent\nbody\n## Child\nbody\n", encoding="utf-8")
    add_document_config(tmp_path, "tree.md", Authority.REFERENCE)
    with ContextEngine(tmp_path) as engine:
        engine.index_workspace()
        parent_id, child_id = engine.store.section_ids()
        parent = engine.get_references(parent_id)
        child = engine.get_references(child_id)
        assert any(
            edge.edge.edge_type is EdgeType.PARENT_OF and edge.edge.target_section_id == child_id
            for edge in parent
        )
        assert any(
            edge.edge.edge_type is EdgeType.CHILD_OF and edge.edge.target_section_id == parent_id
            for edge in child
        )
