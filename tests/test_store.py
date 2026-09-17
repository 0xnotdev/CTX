from pathlib import Path

import pytest

from ctx.config import (
    ConfigError,
    add_document_config,
    database_path,
    initialize_workspace,
    read_source,
    remove_document_config,
    safe_source_path,
)
from ctx.models import Authority
from ctx.parser import parse_markdown
from ctx.store import SQLiteStore


def test_workspace_add_store_retrieve_remove_with_provenance(tmp_path: Path) -> None:
    source = tmp_path / "spec.md"
    text = "# Contract\r\nRunManifest is normative.\r\n"
    source.write_bytes(text.encode())
    initialize_workspace(tmp_path)
    config = add_document_config(tmp_path, "spec.md", Authority.NORMATIVE, 10)
    assert config.documents[0].authority is Authority.NORMATIVE
    assert read_source(tmp_path, config.documents[0], config.limits) == text

    with SQLiteStore(database_path(tmp_path)) as store:
        record = store.register_document("spec.md", Authority.NORMATIVE, 10)
        parsed = parse_markdown(text, "spec.md")
        assert store.replace_document(record, parsed) == 1
        item = store.get_section(parsed.sections[0].id)
        assert item.text == text
        assert item.provenance.document_path == "spec.md"
        assert item.provenance.document_id == record.id
        assert item.provenance.document_sha256 == parsed.sha256
        assert item.provenance.authority is Authority.NORMATIVE
        assert item.provenance.priority == 10
        assert item.provenance.heading_path == ("Contract",)
        assert item.provenance.start_line == 1
        assert item.provenance.end_line == 2
        assert item.provenance.section_sha256 == parsed.sections[0].sha256
        assert item.provenance.index_version == 1
        assert store.remove_document("spec.md")
        assert store.section_ids() == []

    updated = remove_document_config(tmp_path, "spec.md")
    assert updated.documents == ()


def test_schema_has_required_tables_and_fts(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    with SQLiteStore(database_path(tmp_path)) as store:
        names = {
            row[0]
            for row in store.connection.execute(
                "SELECT name FROM sqlite_master WHERE type IN ('table', 'view')"
            )
        }
    assert {
        "documents",
        "document_versions",
        "sections",
        "chunks",
        "embeddings",
        "references",
        "symbols",
        "checkpoint_metadata",
        "index_metadata",
        "chunks_fts",
    } <= names


def test_paths_are_contained_markdown_only_and_size_bounded(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "ok.md").write_text("ok", encoding="utf-8")
    (tmp_path / "no.txt").write_text("no", encoding="utf-8")
    outside = tmp_path.parent / f"{tmp_path.name}-outside.md"
    outside.write_text("outside", encoding="utf-8")
    try:
        assert safe_source_path(tmp_path, "ok.md", 2).name == "ok.md"
        with pytest.raises(ConfigError):
            safe_source_path(tmp_path, "../" + outside.name, 100)
        with pytest.raises(ConfigError):
            safe_source_path(tmp_path, str(outside), 100)
        with pytest.raises(ConfigError):
            safe_source_path(tmp_path, "no.txt", 100)
        with pytest.raises(ConfigError):
            safe_source_path(tmp_path, "ok.md", 1)
        link = tmp_path / "escape.md"
        link.symlink_to(outside)
        with pytest.raises(ConfigError):
            safe_source_path(tmp_path, "escape.md", 100)
    finally:
        outside.unlink()


def test_sql_values_are_parameterized(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    hostile = "odd'); DROP TABLE documents;--.md"
    (tmp_path / hostile).write_text("# Safe\n", encoding="utf-8")
    config = add_document_config(tmp_path, hostile, Authority.REFERENCE)
    with SQLiteStore(database_path(tmp_path)) as store:
        document = config.documents[0]
        record = store.register_document(document.path, document.authority, document.priority)
        store.replace_document(record, parse_markdown("# Safe\n", hostile))
        assert store.get_document_by_path(hostile).path == hostile
        assert len(store.list_documents()) == 1
