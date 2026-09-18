from __future__ import annotations

import json
from pathlib import Path

import pytest

from ctx.checkpoints import checkpoint_artifact_path
from ctx.config import (
    add_document_config,
    initialize_workspace,
    remove_document_config,
)
from ctx.models import Authority
from ctx.service import AmbiguousCheckpointError, ContextEngine


def _add(tmp_path: Path, names: tuple[str, ...]) -> None:
    initialize_workspace(tmp_path)
    for name in names:
        (tmp_path / name).write_text(
            f"# CP-17 — {name}\nGoal: exact requirement from {name}.\n", encoding="utf-8"
        )
        add_document_config(tmp_path, name, Authority.NORMATIVE)


def _write_artifact(root: Path, document_id: str, owner: str) -> Path:
    path = checkpoint_artifact_path(root, document_id, "CP-17")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "provenance": {"document_id": document_id},
                "owner": owner,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_same_checkpoint_id_in_three_documents_never_cross_hydrates(tmp_path: Path) -> None:
    names = ("a.md", "b.md", "c.md")
    _add(tmp_path, names)
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        records = {item.path: item.id for item in engine.store.list_documents()}
        paths = {name: _write_artifact(tmp_path, records[name], name) for name in names}
        engine.sync_workspace()

        for name in names:
            result = engine.get_checkpoint("CP-17", document=name)
            artifact = result.metadata.generated_artifact
            assert artifact is not None
            assert artifact.data["owner"] == name
            assert artifact.path == paths[name].relative_to(tmp_path).as_posix()
        with pytest.raises(AmbiguousCheckpointError) as captured:
            engine.get_checkpoint("CP-17")
        assert len(captured.value.candidates) == 3


def test_delete_cleans_only_deleted_document_artifact_and_rename_is_stable(
    tmp_path: Path,
) -> None:
    _add(tmp_path, ("a.md", "b.md"))
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        records = {item.path: item.id for item in engine.store.list_documents()}
        a_artifact = _write_artifact(tmp_path, records["a.md"], "a")
        b_artifact = _write_artifact(tmp_path, records["b.md"], "b")
        engine.sync_workspace()

        (tmp_path / "a.md").rename(tmp_path / "renamed.md")
        remove_document_config(tmp_path, "a.md")
        add_document_config(tmp_path, "renamed.md", Authority.NORMATIVE)
        engine.sync_workspace()
        assert engine.store.get_document_by_path("renamed.md").id == records["a.md"]
        assert a_artifact.exists()
        renamed = engine.get_checkpoint("CP-17", document="renamed.md")
        assert renamed.metadata.generated_artifact is not None
        assert renamed.metadata.generated_artifact.data["owner"] == "a"

        remove_document_config(tmp_path, "renamed.md")
        engine.sync_workspace()
        assert not a_artifact.parent.exists()
        assert b_artifact.exists()
        survivor = engine.get_checkpoint("CP-17", document="b.md")
        assert survivor.metadata.generated_artifact is not None
        assert survivor.metadata.generated_artifact.data["owner"] == "b"


def test_namespaced_artifact_survives_rebuild(tmp_path: Path) -> None:
    _add(tmp_path, ("one.md",))
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        document_id = engine.store.get_document_by_path("one.md").id
        path = _write_artifact(tmp_path, document_id, "stable")
        engine.sync_workspace()
        generation = engine.store.index_generation()

    with ContextEngine(tmp_path) as rebuilt:
        rebuilt.sync_workspace()
        assert rebuilt.store.index_generation() == generation
        artifact = rebuilt.get_checkpoint("CP-17", document="one.md").metadata.generated_artifact
        assert path.exists()
        assert artifact is not None and artifact.data["owner"] == "stable"


def test_legacy_artifact_migrates_only_with_unique_provenance(tmp_path: Path) -> None:
    _add(tmp_path, ("a.md", "b.md"))
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        a_id = engine.store.get_document_by_path("a.md").id
        legacy = tmp_path / ".ctx" / "checkpoints" / "CP-17.json"
        legacy.parent.mkdir(exist_ok=True)
        legacy.write_text(
            json.dumps({"provenance": {"document_id": a_id}, "owner": "legacy-a"}),
            encoding="utf-8",
        )
        engine.sync_workspace()
        migrated = checkpoint_artifact_path(tmp_path, a_id, "CP-17")
        assert migrated.exists()
        assert not legacy.exists()
        a = engine.get_checkpoint("CP-17", document="a.md")
        b = engine.get_checkpoint("CP-17", document="b.md")
        assert a.metadata.generated_artifact is not None
        assert a.metadata.generated_artifact.data["owner"] == "legacy-a"
        assert b.metadata.generated_artifact is None

        legacy.write_text(json.dumps({"owner": "ambiguous-unscoped"}), encoding="utf-8")
        engine.sync_workspace()
        assert legacy.exists()  # explicitly invalidated/ignored, never reinterpreted
        assert engine.get_checkpoint("CP-17", document="b.md").metadata.generated_artifact is None
