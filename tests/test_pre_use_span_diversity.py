from __future__ import annotations

from pathlib import Path

from ctx.config import add_document_config, initialize_workspace
from ctx.models import Authority
from ctx.parser import sha256_text
from ctx.service import ContextEngine


def _workspace(tmp_path: Path) -> ContextEngine:
    initialize_workspace(tmp_path)
    distant = "DISTANT_ALPHA requires signed input before dispatch."
    distant_two = "DISTANT_OMEGA requires durable receipt verification after commit."
    huge = (
        "# Huge workflow\n"
        + distant
        + "\n"
        + ("ordinary transition material without query vocabulary. " * 500)
        + distant_two
        + "\n"
    )
    (tmp_path / "primary.md").write_text(
        huge
        + "# Independent safety\n"
        + "DISTANT_ALPHA and DISTANT_OMEGA also require an independent audit.\n",
        encoding="utf-8",
    )
    (tmp_path / "secondary.md").write_text(
        "# Secondary evidence\nDISTANT_ALPHA supports another relevant document.\n",
        encoding="utf-8",
    )
    add_document_config(tmp_path, "primary.md", Authority.NORMATIVE)
    add_document_config(tmp_path, "secondary.md", Authority.NORMATIVE)
    engine = ContextEngine(tmp_path)
    engine.sync_workspace()
    return engine


def test_two_distant_spans_same_section_and_other_section_survive(tmp_path: Path) -> None:
    with _workspace(tmp_path) as engine:
        hits = engine.search_lexical("DISTANT_ALPHA DISTANT_OMEGA", limit=5)
        huge = [hit for hit in hits if hit.source.provenance.heading_path[-1] == "Huge workflow"]
        assert len(huge) == 2
        huge.sort(key=lambda hit: hit.source.provenance.start_offset)
        assert huge[0].source.provenance.end_offset <= huge[1].source.provenance.start_offset
        assert any(hit.source.provenance.heading_path[-1] == "Independent safety" for hit in hits)
        assert any(hit.source.provenance.document_path == "secondary.md" for hit in hits)
        for hit in huge:
            provenance = hit.source.provenance
            assert provenance.range_sha256 == sha256_text(hit.source.text)
            assert (
                provenance.section_sha256
                == engine.get_section(provenance.section_id).provenance.section_sha256
            )

        pack = engine.get_context_pack("DISTANT_ALPHA DISTANT_OMEGA", 8_000)
        packed_huge = [
            item
            for item in pack.items
            if item.source.provenance.heading_path[-1] == "Huge workflow"
        ]
        assert len(packed_huge) == 2
        assert len({item.range_sha256 for item in packed_huge}) == 2


def test_adjacent_overlap_and_near_duplicate_windows_collapse(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    repeated = "OVERLAP_NEEDLE shared paragraph with exactly the same relevant requirement."
    source = (
        "# One huge section\n"
        + ("lead material. " * 180)
        + repeated
        + "\n"
        + ("tail material. " * 180)
    )
    (tmp_path / "one.md").write_text(source, encoding="utf-8")
    duplicate = f"# Copied\n{repeated}\n"
    (tmp_path / "duplicate.md").write_text(duplicate, encoding="utf-8")
    (tmp_path / "duplicate-copy.md").write_text(duplicate, encoding="utf-8")
    (tmp_path / "other.md").write_text(
        "# Other\nOVERLAP_NEEDLE independent control evidence.\n", encoding="utf-8"
    )
    for path in ("one.md", "duplicate.md", "duplicate-copy.md", "other.md"):
        add_document_config(tmp_path, path, Authority.NORMATIVE)

    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        raw = engine.store.connection.execute(
            "SELECT COUNT(*) FROM search_chunks WHERE source_text LIKE '%OVERLAP_NEEDLE%'"
        ).fetchone()[0]
        assert raw >= 2  # the paragraph lies in adjacent overlapping navigation windows
        hits = engine.search_lexical("OVERLAP_NEEDLE shared paragraph", limit=10)
        one = [hit for hit in hits if hit.source.provenance.document_path == "one.md"]
        assert len(one) == 1
        duplicate_hits = [
            hit
            for hit in hits
            if hit.source.provenance.document_path in {"duplicate.md", "duplicate-copy.md"}
        ]
        assert len(duplicate_hits) == 1
        assert any(hit.source.provenance.document_path == "other.md" for hit in hits)


def test_span_cap_prevents_one_section_from_flooding_results(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    occurrences = "\n".join(
        f"FLOOD_NEEDLE independently relevant distant rule number {index}." + (" filler" * 400)
        for index in range(8)
    )
    (tmp_path / "flood.md").write_text(f"# Flood\n{occurrences}\n", encoding="utf-8")
    (tmp_path / "survivor.md").write_text(
        "# Survivor\nFLOOD_NEEDLE separate document requirement.\n", encoding="utf-8"
    )
    add_document_config(tmp_path, "flood.md", Authority.NORMATIVE)
    add_document_config(tmp_path, "survivor.md", Authority.NORMATIVE)
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        hits = engine.search_lexical("FLOOD_NEEDLE", limit=10)
        flood = [hit for hit in hits if hit.source.provenance.document_path == "flood.md"]
        assert 1 < len(flood) <= 3
        assert any(hit.source.provenance.document_path == "survivor.md" for hit in hits)
