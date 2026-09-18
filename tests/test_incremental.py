from pathlib import Path

import pytest

from ctx.config import (
    add_document_config,
    initialize_workspace,
    remove_document_config,
)
from ctx.embeddings import HashEmbedding
from ctx.models import Authority
from ctx.service import ContextEngine, StaleIndexError


def workspace(tmp_path: Path) -> Path:
    initialize_workspace(tmp_path)
    (tmp_path / "spec.md").write_text("# A\nalpha\n# B\nbeta\n", encoding="utf-8")
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    return tmp_path


def test_unchanged_does_zero_work_and_one_change_is_incremental(tmp_path: Path) -> None:
    root = workspace(tmp_path)
    backend = HashEmbedding(16)
    with ContextEngine(root, embedder=backend) as engine:
        first = engine.index_workspace()
        assert first.documents_added == 1
        assert first.sections_added == 2
        version = first.index_version
        section_ids = engine.store.section_ids()

        unchanged = engine.sync_workspace()
        assert unchanged.documents_unchanged == 1
        assert unchanged.sections_added == unchanged.sections_changed == 0
        assert unchanged.index_version == version

        (root / "spec.md").write_text("# A\nalpha\n# B\nbeta changed\n", encoding="utf-8")
        changed = engine.sync_workspace()
        assert changed.documents_changed == 1
        assert changed.sections_changed == 1
        assert changed.sections_unchanged == 1
        assert changed.embeddings_retained == 1
        assert engine.store.section_ids() == section_ids
        embeddings = engine.store.connection.execute("SELECT COUNT(*) FROM embeddings").fetchone()
        assert embeddings[0] == 2


def test_section_deletion_and_deterministic_rename(tmp_path: Path) -> None:
    root = workspace(tmp_path)
    with ContextEngine(root) as engine:
        engine.index_workspace()
        original_document = engine.store.list_documents()[0]
        (root / "spec.md").write_text("# A\nalpha\n", encoding="utf-8")
        deleted = engine.sync_workspace()
        assert deleted.sections_removed == 1

        (root / "spec.md").rename(root / "renamed.md")
        remove_document_config(root, "spec.md")
        add_document_config(root, "renamed.md", Authority.NORMATIVE)
        renamed = engine.sync_workspace()
        assert renamed.documents_renamed == 1
        assert renamed.documents_unchanged == 1
        assert engine.store.list_documents()[0].id == original_document.id
        assert engine.store.list_documents()[0].path == "renamed.md"


def test_stale_source_is_never_returned_and_auto_sync_is_explicit(tmp_path: Path) -> None:
    root = workspace(tmp_path)
    with ContextEngine(root) as engine:
        engine.index_workspace()
        section_id = engine.store.section_ids()[0]
        old = engine.get_section(section_id)
        assert "alpha" in old.text
        (root / "spec.md").write_text("# A\nnew alpha\n# B\nbeta\n", encoding="utf-8")
        assert engine.status().stale_documents == ("spec.md",)
        with pytest.raises(StaleIndexError, match="STALE_INDEX"):
            engine.get_section(section_id)
        fresh = engine.get_section(section_id, auto_sync=True)
        assert "new alpha" in fresh.text
        assert fresh.provenance.document_sha256 != old.provenance.document_sha256
