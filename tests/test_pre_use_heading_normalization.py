from __future__ import annotations

from pathlib import Path

from ctx.config import add_document_config, initialize_workspace
from ctx.heading import canonical_heading
from ctx.models import Authority, EdgeType, ResolutionStatus
from ctx.parser import parse_markdown
from ctx.service import ContextEngine


def test_canonical_heading_scheme_covers_unicode_spacing_and_punctuation() -> None:
    cases = {
        "  API   Guide  ": "api-guide",
        "ＡＰＩ．Ｇｕｉｄｅ": "api-guide",
        "one_two---three": "one-two-three",
        "Periods... (parentheses) / slashes": "periods-parentheses-slashes",
        "emoji 🙂🚀 heading": "emoji-heading",
        "Straße": "strasse",
        "Καλημέρα Κόσμε": "καλημέρα-κόσμε",
        "Привет-мир": "привет-мир",
        "日本語／見出し": "日本語-見出し",
        "___ ... 🙂": "section",
    }
    assert {value: canonical_heading(value) for value in cases} == cases


def test_parser_uses_canonical_identity_and_duplicate_disambiguation() -> None:
    source = "# ＡＰＩ Guide\none\n# api__guide\ntwo\n# API...Guide\nthree\n"
    first = parse_markdown(source, "doc:id")
    second = parse_markdown(source, "doc:id")
    assert first == second
    assert len({section.id for section in first.sections}) == 3
    assert [section.id.split(":", 2)[1] for section in first.sections] == [
        "api-guide",
        "api-guide-2",
        "api-guide-3",
    ]
    assert [section.heading for section in first.sections] == [
        "ＡＰＩ Guide",
        "api__guide",
        "API...Guide",
    ]


def test_canonical_graph_anchor_resolution_and_duplicate_ambiguity(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "links.md").write_text(
        "# Caller\n"
        "See [local](#ＡＰＩ／Guide) and [external](other.md#API-Guide).\n"
        "# API / Guide\nFirst target.\n"
        "# API__Guide\nSecond canonical duplicate.\n",
        encoding="utf-8",
    )
    add_document_config(tmp_path, "links.md", Authority.NORMATIVE)
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        caller = engine.search_exact("Caller", limit=1)[0].source
        links = [
            item
            for item in engine.get_references(caller.provenance.section_id)
            if item.edge.edge_type is EdgeType.RELATED_SECTION
        ]
        local = next(item for item in links if item.edge.label == "#ＡＰＩ／Guide")
        assert local.edge.status is ResolutionStatus.AMBIGUOUS
        assert local.target is None
        assert len(local.candidates) == 2
        assert tuple(item.section_id for item in local.candidates) == tuple(
            sorted(
                (item.section_id for item in local.candidates),
                key=lambda section_id: engine.get_section(section_id).provenance.start_line,
            )
        )
        external = next(item for item in links if item.edge.label == "other.md#API-Guide")
        assert external.edge.status is ResolutionStatus.UNRESOLVED
        assert "external anchor" in external.edge.reason


def test_index_search_and_heading_prefix_use_same_canonical_key(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "spec.md").write_text(
        "# Platform／Core\n## API.Guide (v1) 🙂\nCanonical target evidence.\n",
        encoding="utf-8",
    )
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        hits = engine.search_exact("ＡＰＩ＿ＧＵＩＤＥ　Ｖ１", limit=5)
        assert hits
        assert hits[0].source.provenance.heading_path[-1] == "API.Guide (v1) 🙂"
        filtered = engine.search_exact(
            "api guide v1", limit=5, heading_prefix=("ＰＬＡＴＦＯＲＭ core",)
        )
        assert filtered
        assert filtered[0].source.provenance.heading_path == (
            "Platform／Core",
            "API.Guide (v1) 🙂",
        )
