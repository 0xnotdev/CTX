from __future__ import annotations

import socket
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ctx.config import (
    ConfigError,
    add_document_config,
    initialize_workspace,
    load_config,
    remove_document_config,
)
from ctx.context_pack import (
    ContextBudgetTooSmall,
    StrictByteUpperBound,
    build_context_pack,
    serialized_agent_response,
)
from ctx.embeddings import HashEmbedding, verify_model, write_manifest
from ctx.models import Authority, ResolutionStatus
from ctx.service import AmbiguousCheckpointError, ContextEngine, create_context_engine
from ctx.store import SCHEMA_VERSION, SQLiteStore


def configured(root: Path, files: dict[str, str]) -> None:
    initialize_workspace(root)
    for path, text in files.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        add_document_config(root, path, Authority.NORMATIVE)


def test_serialized_budget_boundaries_long_task_unicode_and_emoji(tmp_path: Path) -> None:
    configured(tmp_path, {"spec.md": "# Requirement\nExact source.\n"})
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        task = "é🙂" * 500
        with pytest.raises(ContextBudgetTooSmall) as captured:
            engine.get_context_pack(task, 64)
        error = captured.value
        assert error.requested_budget == 64
        assert error.minimum_required > 64
        assert error.task_token_estimate > 500
        assert error.metadata_overhead > 0
        with pytest.raises(ContextBudgetTooSmall):
            engine.get_context_pack(task, error.minimum_required - 1)
        boundary = engine.get_context_pack(task, error.minimum_required)
        assert boundary.serialized_estimated_tokens <= boundary.token_budget
        plus_one = engine.get_context_pack(task, error.minimum_required + 1)
        assert plus_one.serialized_estimated_tokens <= plus_one.token_budget

        strict = StrictByteUpperBound()
        strict_pack = build_context_pack(
            engine,
            "emoji 🙂 provenance",
            20_000,
            counter=strict,
        )
        encoded = serialized_agent_response(strict_pack)
        assert strict.count_serialized(encoded) <= strict_pack.serialized_estimated_tokens
        assert strict_pack.serialized_estimated_tokens <= strict_pack.token_budget


def test_multichunk_tail_search_and_match_centered_pack(tmp_path: Path) -> None:
    filler = "ordinary filler sentence. " * 700
    tail = "TAIL_NEEDLE requires the cobalt handshake before commit."
    configured(tmp_path, {"long.md": f"# Huge requirement\n{filler}{tail}\n"})
    with ContextEngine(tmp_path, embedder=HashEmbedding(32)) as engine:
        engine.sync_workspace()
        section_id = engine.store.section_ids()[0]
        assert engine.store.chunk_count(section_id) > 20
        hit = engine.search("TAIL_NEEDLE cobalt", limit=1)[0]
        full = engine.get_section(section_id)
        assert hit.source.source_type == "excerpt"
        assert tail in hit.source.text
        assert len(hit.source.text) < len(full.text) // 5
        assert hit.source.provenance.section_sha256 == full.provenance.section_sha256
        assert hit.source.provenance.range_sha256 != full.provenance.range_sha256
        pack = engine.get_context_pack("Implement TAIL_NEEDLE cobalt", 2_000)
        assert any("TAIL_NEEDLE" in item.source.text for item in pack.items)
        assert all(item.index_generation == pack.index_generation for item in pack.items)


def test_opaque_identity_rename_old_path_reuse_and_recreate(tmp_path: Path) -> None:
    configured(tmp_path, {"a.md": "# Original\none\n"})
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        original = engine.store.get_document_by_path("a.md").id
        assert original.startswith("doc:") and len(original) > 30

        (tmp_path / "a.md").rename(tmp_path / "b.md")
        remove_document_config(tmp_path, "a.md")
        add_document_config(tmp_path, "b.md", Authority.NORMATIVE)
        engine.sync_workspace()
        assert engine.store.get_document_by_path("b.md").id == original

        (tmp_path / "a.md").write_text("# Unrelated\ntwo\n", encoding="utf-8")
        add_document_config(tmp_path, "a.md", Authority.NORMATIVE)
        engine.sync_workspace()
        replacement = engine.store.get_document_by_path("a.md").id
        assert replacement != original
        assert engine.store.get_document_by_path("b.md").id == original

        remove_document_config(tmp_path, "a.md")
        engine.sync_workspace()
        add_document_config(tmp_path, "a.md", Authority.NORMATIVE)
        engine.sync_workspace()
        assert engine.store.get_document_by_path("a.md").id != replacement


def test_checkpoint_authority_and_equal_tie_ambiguity(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    (tmp_path / "norm.md").write_text("# CP-14 — Normative\nMust apply.\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("# CP-14 — Notes\nMay apply.\n", encoding="utf-8")
    add_document_config(tmp_path, "norm.md", Authority.NORMATIVE)
    add_document_config(tmp_path, "notes.md", Authority.INFORMAL, 100)
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        assert engine.get_checkpoint("CP-14").metadata.title == "Normative"
        assert engine.get_checkpoint("CP-14", document="notes.md").metadata.title == "Notes"

        add_document_config(tmp_path, "notes.md", Authority.NORMATIVE, 0)
        engine.sync_workspace()
        with pytest.raises(AmbiguousCheckpointError) as captured:
            engine.get_checkpoint("CP-14")
        assert len(captured.value.candidates) == 2


def test_reference_same_document_preference_and_cross_document_ambiguity(tmp_path: Path) -> None:
    configured(
        tmp_path,
        {
            "one.md": "# CP-2 — One\nTarget.\n# Caller\nSee CP-2.\n",
            "two.md": "# CP-2 — Two\nTarget.\n",
            "three.md": "# Caller three\nSee CP-2.\n",
        },
    )
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        same = engine.search_exact("Caller", limit=10)
        one = next(hit for hit in same if hit.source.provenance.document_path == "one.md")
        edge = next(
            item
            for item in engine.get_references(one.source.provenance.section_id)
            if item.edge.label == "CP-2"
        )
        assert edge.edge.status is ResolutionStatus.RESOLVED
        assert edge.target is not None
        assert edge.target.provenance.document_path == "one.md"

        three = next(hit for hit in same if hit.source.provenance.document_path == "three.md")
        ambiguous = next(
            item
            for item in engine.get_references(three.source.provenance.section_id)
            if item.edge.label == "CP-2"
        )
        assert ambiguous.edge.status is ResolutionStatus.AMBIGUOUS
        assert ambiguous.target is None
        assert len(ambiguous.candidates) == 2


def test_authority_priority_and_algorithm_metadata_advance_generation(tmp_path: Path) -> None:
    configured(tmp_path, {"spec.md": "# A\ntext\n"})
    with ContextEngine(tmp_path) as engine:
        first = engine.sync_workspace().index_generation
        add_document_config(tmp_path, "spec.md", Authority.REFERENCE, 0)
        authority = engine.sync_workspace().index_generation
        assert authority == first + 1
        add_document_config(tmp_path, "spec.md", Authority.REFERENCE, 50)
        priority = engine.sync_workspace().index_generation
        assert priority == authority + 1
        assert engine.store.get_metadata("parser_version")
        assert engine.store.get_metadata("chunker_version")
        assert engine.store.get_metadata("graph_version")
        assert engine.store.get_metadata("retrieval_version")


def test_filters_are_applied_before_top_k(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    for index in range(20):
        name = f"noise-{index}.md"
        (tmp_path / name).write_text("# Needle\nneedle needle needle\n", encoding="utf-8")
        add_document_config(tmp_path, name, Authority.INFORMAL, 100)
    (tmp_path / "spec.md").write_text("# Needle requirement\nneedle normative\n", encoding="utf-8")
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        hit = engine.search("needle", limit=1, authority_floor=Authority.NORMATIVE)[0]
        assert hit.source.provenance.document_path == "spec.md"
        excluded = engine.search("needle", limit=5, exclude_documents={"spec.md"})
        assert all(item.source.provenance.document_path != "spec.md" for item in excluded)


def test_concurrent_reads_and_sync_observe_complete_generations(tmp_path: Path) -> None:
    class BlockingEmbedding(HashEmbedding):
        def __init__(self) -> None:
            super().__init__(16)
            self.block = False
            self.started = threading.Event()
            self.release = threading.Event()

        def embed_documents(self, texts):  # type: ignore[no-untyped-def]
            if self.block:
                self.started.set()
                assert self.release.wait(timeout=10)
            return super().embed_documents(texts)

    configured(tmp_path, {"spec.md": "# A\nneedle old\n# B\nother\n"})
    backend = BlockingEmbedding()
    with ContextEngine(tmp_path, embedder=backend) as engine:
        engine.sync_workspace()

        def read_once(_: int) -> tuple[int, int]:
            hits = engine.search("needle", limit=2)
            pack = engine.get_context_pack("needle", 1_500)
            generations = {item.index_generation for item in pack.items}
            assert generations <= {pack.index_generation}
            return hits[0].index_generation, pack.index_generation

        with ThreadPoolExecutor(max_workers=20) as pool:
            before = list(pool.map(read_once, range(20)))
        assert len(before) == 20

        (tmp_path / "spec.md").write_text("# A\nneedle new\n# B\nother\n", encoding="utf-8")
        backend.block = True
        with ThreadPoolExecutor(max_workers=8) as pool:
            future_sync = pool.submit(engine.sync_workspace)
            assert backend.started.wait(timeout=10)
            futures = [pool.submit(read_once, index) for index in range(20)]
            backend.release.set()
            generation = future_sync.result().index_generation
            observed = [future.result() for future in futures]
        assert all(left == right for left, right in observed)
        assert all(left <= generation for left, _ in observed)


def test_v1_database_migrates_document_config_to_v2(tmp_path: Path) -> None:
    database = tmp_path / "legacy.sqlite3"
    connection = sqlite3.connect(database)
    connection.executescript(
        """
        CREATE TABLE documents (
          id TEXT PRIMARY KEY, path TEXT UNIQUE, authority INTEGER, priority INTEGER,
          current_sha256 TEXT, indexed_at TEXT
        );
        CREATE TABLE index_metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        INSERT INTO documents VALUES ('doc:legacy','spec.md',4,7,'abc','then');
        INSERT INTO index_metadata VALUES ('index_version','9');
        PRAGMA user_version=1;
        """
    )
    connection.close()
    with SQLiteStore(database) as store:
        assert int(store.connection.execute("PRAGMA user_version").fetchone()[0]) == SCHEMA_VERSION
        record = store.get_document_by_path("spec.md")
        assert record.authority is Authority.NORMATIVE
        assert record.priority == 7
        assert record.id != "doc:legacy"
        assert record.sha256 is None  # derived V1 index is explicitly rebuilt


def test_malformed_duplicate_and_future_config_are_actionable(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    config_path = tmp_path / ".ctx" / "config.toml"
    config_path.write_text("version=2\nversion=2\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="invalid"):
        load_config(tmp_path)
    config_path.write_text("version=999\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="unsupported config version"):
        load_config(tmp_path)
    config_path.write_text(
        'version=2\n[[documents]]\npath="A.md"\n[[documents]]\npath="a.md"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="duplicate normalized"):
        load_config(tmp_path)


def test_adversarial_huge_blocks_unicode_and_random_data_remain_exact(tmp_path: Path) -> None:
    random_like = "A1b2C3d4+/=" * 2_000
    fence = "```text\n" + ("fenced payload 🙂\n" * 500) + "```\n"
    table = "| key | value |\n|---|---|\n" + ("| k | emoji🙂 |\n" * 500)
    source = "# " + ("巨大🙂" * 500) + "\n" + random_like + "\n" + fence + table
    configured(tmp_path, {"adversarial.md": source})
    with ContextEngine(tmp_path) as engine:
        engine.sync_workspace()
        sections = engine.store.document_sections("adversarial.md")
        assert "".join(item.text for item in sections) == source
        rows = engine.store.connection.execute(
            "SELECT source_text,source_sha256,embedding_text FROM search_chunks ORDER BY ordinal"
        ).fetchall()
        assert "".join(str(row["source_text"]) for row in rows) == source
        assert all(len(str(row["embedding_text"]).encode()) <= 448 for row in rows)
        assert all(row["source_sha256"] for row in rows)


def test_same_content_documents_subdirectories_and_100_document_scale(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    for index in range(100):
        relative = f"group-{index % 10}/same.md" if index < 10 else f"docs/{index}.md"
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("# Same\nidentical content\n", encoding="utf-8")
        add_document_config(tmp_path, relative, Authority.REFERENCE)
    with ContextEngine(tmp_path) as engine:
        stats = engine.sync_workspace()
        assert stats.documents_added == 100
        records = engine.store.list_documents()
        assert len(records) == 100
        assert len({record.id for record in records}) == 100
        assert len({record.path for record in records}) == 100


def test_corrupt_future_database_and_wrong_model_checksum_fail_closed(tmp_path: Path) -> None:
    future = tmp_path / "future.sqlite3"
    connection = sqlite3.connect(future)
    connection.execute("PRAGMA user_version=999")
    connection.close()
    with pytest.raises(RuntimeError, match="newer than supported"):
        SQLiteStore(future)

    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(b"not a sqlite database")
    with pytest.raises(sqlite3.DatabaseError):
        SQLiteStore(corrupt)

    model = tmp_path / "model"
    model.mkdir()
    artifact = model / "model.onnx"
    artifact.write_bytes(b"verified")
    locks = model / ".locks"
    locks.mkdir()
    (locks / "runtime.lock").write_bytes(b"")
    write_manifest(
        model,
        model_name="fixture/model",
        revision="abc123",
        dimensions=8,
        runtime_version="fixture",
    )
    verify_model(model)
    (locks / "runtime.lock").write_bytes(b"transient")
    verify_model(model)
    artifact.write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        verify_model(model)


def test_windows_drive_case_and_backslash_paths_are_rejected_or_normalized(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    config = tmp_path / ".ctx" / "config.toml"
    config.write_text('version=2\n[[documents]]\npath="C:\\\\outside.md"\n', encoding="utf-8")
    with pytest.raises(ConfigError, match="relative"):
        load_config(tmp_path)
    config.write_text(
        'version=2\n[[documents]]\npath="Docs/Spec.md"\n[[documents]]\npath="docs/spec.md"\n',
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="duplicate normalized"):
        load_config(tmp_path)


def test_normal_factory_never_attempts_network_when_model_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    configured(tmp_path, {"spec.md": "# A\noffline needle\n"})

    def denied(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("network attempted")

    monkeypatch.setattr(socket, "create_connection", denied)
    monkeypatch.setattr(socket.socket, "connect", denied)
    monkeypatch.setenv("CTX_MODEL_DIR", str(tmp_path / "missing-models"))
    with create_context_engine(tmp_path) as engine:
        assert engine.active_channels == ("structural", "lexical")
        engine.sync_workspace()
        assert engine.search("offline needle")
        assert engine.get_context_pack("offline needle", 1_500).items
