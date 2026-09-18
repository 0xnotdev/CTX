"""SQLite V2 persistence with thread-local connections and atomic index generations."""

from __future__ import annotations

import json
import re
import sqlite3
import threading
import uuid
from collections import OrderedDict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager, suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

from ctx.checkpoints import CheckpointMetadata
from ctx.embeddings import cosine_scores
from ctx.graph import ExtractedGraph, GraphSection
from ctx.heading import canonical_heading, canonical_heading_path_json
from ctx.models import (
    Authority,
    DocumentRecord,
    EdgeType,
    FilterSet,
    OutlineEntry,
    ParsedDocument,
    Provenance,
    ReferenceOrigin,
    ReferenceRecord,
    ReferenceResult,
    ResolutionStatus,
    SearchHit,
    SourceExcerpt,
    SourceItem,
    SourceRef,
    SourceSection,
    SymbolResult,
    SyncStats,
)
from ctx.retrieval import diversify_spans

SCHEMA_VERSION = 2
SQLITE_BUSY_TIMEOUT_MS = 5_000
SQLITE_SYNCHRONOUS = "NORMAL"  # WAL+NORMAL: durable against process crash, not power-loss perfect.

_LOCKS_GUARD = threading.Lock()
_WRITE_LOCKS: dict[str, threading.RLock] = {}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    authority INTEGER NOT NULL CHECK(authority BETWEEN 0 AND 4),
    priority INTEGER NOT NULL CHECK(priority BETWEEN -100 AND 100),
    current_sha256 TEXT,
    indexed_at TEXT
);
CREATE TABLE IF NOT EXISTS document_versions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    sha256 TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE(document_id, sha256, parser_version, chunker_version)
);
CREATE TABLE IF NOT EXISTS sections (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    document_version_id INTEGER NOT NULL REFERENCES document_versions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    level INTEGER NOT NULL,
    heading TEXT NOT NULL,
    heading_path TEXT NOT NULL,
    parent_id TEXT,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    text TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    UNIQUE(document_id, ordinal)
);
CREATE TABLE IF NOT EXISTS search_chunks (
    id TEXT PRIMARY KEY,
    section_id TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    start_column INTEGER NOT NULL,
    end_column INTEGER,
    start_offset INTEGER NOT NULL,
    end_offset INTEGER NOT NULL,
    source_text TEXT NOT NULL,
    source_sha256 TEXT NOT NULL,
    embedding_text TEXT NOT NULL,
    embedding_sha256 TEXT NOT NULL,
    chunker_version TEXT NOT NULL,
    token_estimate INTEGER NOT NULL,
    UNIQUE(section_id, ordinal)
);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES search_chunks(id) ON DELETE CASCADE,
    embedding_identity TEXT NOT NULL,
    provider TEXT NOT NULL,
    model_name TEXT NOT NULL,
    revision TEXT NOT NULL,
    artifact_sha256 TEXT NOT NULL,
    runtime_version TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL,
    chunk_source_sha256 TEXT NOT NULL,
    embedding_text_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS "references" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_section_id TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    target_section_id TEXT REFERENCES sections(id) ON DELETE SET NULL,
    candidate_target_ids TEXT NOT NULL,
    edge_type TEXT NOT NULL,
    label TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT NOT NULL,
    evidence TEXT NOT NULL,
    origin TEXT NOT NULL,
    UNIQUE(source_section_id, edge_type, label, origin)
);
CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT NOT NULL,
    section_id TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    origin TEXT NOT NULL,
    confidence REAL NOT NULL,
    PRIMARY KEY(symbol, section_id, kind, origin)
);
CREATE TABLE IF NOT EXISTS checkpoints (
    document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    checkpoint_id TEXT NOT NULL,
    root_section_id TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    title TEXT NOT NULL,
    fields_json TEXT NOT NULL,
    PRIMARY KEY(document_id, checkpoint_id)
);
CREATE TABLE IF NOT EXISTS index_generations (
    generation INTEGER PRIMARY KEY,
    created_at TEXT NOT NULL,
    reason TEXT NOT NULL,
    behavior_fingerprint TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS index_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS search_chunks_fts USING fts5(
    chunk_id UNINDEXED,
    section_id UNINDEXED,
    heading,
    source_text,
    tokenize='unicode61 tokenchars ''._@:-'''
);
CREATE INDEX IF NOT EXISTS idx_sections_document ON sections(document_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON search_chunks(section_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_references_source ON "references"(source_section_id);
CREATE INDEX IF NOT EXISTS idx_references_target ON "references"(target_section_id);
CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON symbols(symbol);
CREATE INDEX IF NOT EXISTS idx_checkpoints_id ON checkpoints(checkpoint_id);
"""

_COMPAT = """
CREATE VIEW IF NOT EXISTS chunks AS
SELECT id, section_id, ordinal, start_line, end_line, source_text AS text,
       source_sha256 AS sha256
FROM search_chunks;
CREATE VIEW IF NOT EXISTS checkpoint_metadata AS
SELECT root_section_id AS section_id, checkpoint_id, title, fields_json FROM checkpoints;
CREATE VIEW IF NOT EXISTS chunks_fts AS SELECT * FROM search_chunks_fts;
"""


def document_id(_path: str | None = None) -> str:
    """Generate an opaque identity. The path argument is ignored for V0 API compatibility."""
    return f"doc:{uuid.uuid4()}"


class SQLiteStore:
    """Thread-local SQLite repository with explicit serialized writes.

    Every thread receives its own checked connection. WAL permits concurrent readers; all writes
    additionally take a process-local path lock and use ``BEGIN IMMEDIATE``. Logical workspace
    replacement is one transaction and increments the generation once.
    """

    def __init__(self, path: Path, *, read_only: bool = False):
        if not read_only:
            path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.path = path
        self.read_only = read_only
        self._local = threading.local()
        self._matrix_cache: OrderedDict[
            tuple[int, str, str], tuple[tuple[str, ...], NDArray[np.float32]]
        ] = OrderedDict()
        self._matrix_cache_lock = threading.Lock()
        with _LOCKS_GUARD:
            self._write_lock = _WRITE_LOCKS.setdefault(str(path.resolve()), threading.RLock())
        if not read_only:
            self.migrate()

    def _connect(self) -> sqlite3.Connection:
        if self.read_only:
            connection = sqlite3.connect(
                f"file:{self.path.resolve()}?mode=ro",
                uri=True,
                timeout=SQLITE_BUSY_TIMEOUT_MS / 1_000,
            )
        else:
            connection = sqlite3.connect(self.path, timeout=SQLITE_BUSY_TIMEOUT_MS / 1_000)
        connection.row_factory = sqlite3.Row
        connection.create_function("ctx_heading_key", 1, canonical_heading, deterministic=True)
        connection.create_function(
            "ctx_heading_path_key", 1, canonical_heading_path_json, deterministic=True
        )
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {SQLITE_BUSY_TIMEOUT_MS}")
        if not self.read_only:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(f"PRAGMA synchronous = {SQLITE_SYNCHRONOUS}")
        return connection

    @property
    def connection(self) -> sqlite3.Connection:
        connection = getattr(self._local, "connection", None)
        if connection is None:
            connection = self._connect()
            self._local.connection = connection
        return connection

    def close(self) -> None:
        connection = getattr(self._local, "connection", None)
        if connection is not None:
            connection.close()
            self._local.connection = None

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @contextmanager
    def consistent_read(self) -> Iterator[None]:
        """Hold a logical-generation snapshot across a multi-query service operation."""
        with self._write_lock:
            yield

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        if self.read_only:
            raise RuntimeError("read-only SQLiteStore cannot start a write transaction")
        with self._write_lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    def migrate(self) -> None:
        with self._write_lock:
            connection = self.connection
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if current > SCHEMA_VERSION:
                raise RuntimeError(
                    f"index schema {current} is newer than supported {SCHEMA_VERSION}"
                )
            if current == SCHEMA_VERSION:
                connection.executescript(_SCHEMA)
                connection.executescript(_COMPAT)
                return
            if current == 1:
                self._migrate_v1(connection)
                return
            with connection:
                connection.executescript(_SCHEMA)
                connection.executescript(_COMPAT)
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                connection.execute(
                    "INSERT OR IGNORE INTO index_metadata(key,value) VALUES('index_generation','0')"
                )

    def _migrate_v1(self, connection: sqlite3.Connection) -> None:
        """Preserve document configuration and rebuild disposable V1 derived indexes."""
        rows = connection.execute(
            "SELECT path,authority,priority,current_sha256,indexed_at FROM documents ORDER BY path"
        ).fetchall()
        metadata = {
            str(row[0]): str(row[1])
            for row in connection.execute("SELECT key,value FROM index_metadata").fetchall()
        }
        connection.execute("PRAGMA foreign_keys = OFF")
        try:
            with connection:
                connection.execute("DROP TABLE IF EXISTS chunks_fts")
                for name in (
                    "checkpoint_metadata",
                    "symbols",
                    "references",
                    "embeddings",
                    "chunks",
                    "sections",
                    "document_versions",
                    "documents",
                    "index_metadata",
                ):
                    connection.execute(f'DROP TABLE IF EXISTS "{name}"')
                connection.executescript(_SCHEMA)
                connection.executescript(_COMPAT)
                for row in rows:
                    connection.execute(
                        "INSERT INTO documents(id,path,authority,priority,current_sha256,"
                        "indexed_at) VALUES(?,?,?,?,NULL,NULL)",
                        (document_id(), row[0], row[1], row[2]),
                    )
                generation = max(1, int(metadata.get("index_version", "0")))
                connection.execute(
                    "INSERT INTO index_metadata(key,value) VALUES('index_generation',?)",
                    (str(generation),),
                )
                connection.execute(
                    "INSERT INTO index_generations(generation,created_at,reason,"
                    "behavior_fingerprint) VALUES(?,?,?,?)",
                    (
                        generation,
                        datetime.now(UTC).isoformat(),
                        "V1_TO_V2_MIGRATION",
                        "rebuild-required",
                    ),
                )
                connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        finally:
            connection.execute("PRAGMA foreign_keys = ON")

    def index_generation(self) -> int:
        row = self.connection.execute(
            "SELECT value FROM index_metadata WHERE key='index_generation'"
        ).fetchone()
        return int(row[0]) if row else 0

    def index_version(self) -> int:
        return self.index_generation()

    @staticmethod
    def _increment_generation(
        connection: sqlite3.Connection, reason: str, behavior_fingerprint: str = ""
    ) -> int:
        row = connection.execute(
            "SELECT value FROM index_metadata WHERE key='index_generation'"
        ).fetchone()
        generation = int(row[0]) + 1 if row else 1
        connection.execute(
            "INSERT INTO index_metadata(key,value) VALUES('index_generation',?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(generation),),
        )
        connection.execute(
            "INSERT INTO index_generations(generation,created_at,reason,behavior_fingerprint) "
            "VALUES(?,?,?,?)",
            (generation, datetime.now(UTC).isoformat(), reason, behavior_fingerprint),
        )
        return generation

    def touch_index(self, reason: str = "METADATA_CHANGED") -> int:
        with self.transaction() as connection:
            return self._increment_generation(connection, reason)

    def set_metadata(self, key: str, value: str, *, touch: bool = False) -> None:
        with self.transaction() as connection:
            connection.execute(
                "INSERT INTO index_metadata(key,value) VALUES(?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            if touch:
                self._increment_generation(connection, f"METADATA:{key}")

    def get_metadata(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM index_metadata WHERE key=?", (key,)
        ).fetchone()
        return str(row[0]) if row else None

    def register_document(self, path: str, authority: Authority, priority: int) -> DocumentRecord:
        """Register a path with a one-time opaque ID; metadata updates preserve that ID."""
        with self.transaction() as connection:
            existing = connection.execute(
                "SELECT id FROM documents WHERE path=?", (path,)
            ).fetchone()
            if existing is None:
                connection.execute(
                    "INSERT INTO documents(id,path,authority,priority) VALUES(?,?,?,?)",
                    (document_id(), path, int(authority), priority),
                )
            else:
                connection.execute(
                    "UPDATE documents SET authority=?,priority=? WHERE path=?",
                    (int(authority), priority, path),
                )
        return self.get_document_by_path(path)

    def get_document_by_path(self, path: str) -> DocumentRecord:
        row = self.connection.execute("SELECT * FROM documents WHERE path=?", (path,)).fetchone()
        if row is None:
            raise KeyError(f"document not found: {path}")
        return self._document(row)

    def get_document(self, identifier: str) -> DocumentRecord:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE id=?", (identifier,)
        ).fetchone()
        if row is None:
            raise KeyError(f"document not found: {identifier}")
        return self._document(row)

    def list_documents(self) -> list[DocumentRecord]:
        return [
            self._document(row)
            for row in self.connection.execute("SELECT * FROM documents ORDER BY path").fetchall()
        ]

    @staticmethod
    def _document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=str(row["id"]),
            path=str(row["path"]),
            authority=Authority(row["authority"]),
            priority=int(row["priority"]),
            sha256=str(row["current_sha256"]) if row["current_sha256"] is not None else None,
            indexed_at=str(row["indexed_at"]) if row["indexed_at"] is not None else None,
        )

    def document_parser_version(self, document_id_value: str) -> str | None:
        row = self.connection.execute(
            "SELECT parser_version FROM document_versions WHERE document_id=? "
            "ORDER BY id DESC LIMIT 1",
            (document_id_value,),
        ).fetchone()
        return str(row[0]) if row else None

    def remove_document(self, path: str) -> bool:
        with self.transaction() as connection:
            cursor = connection.execute("DELETE FROM documents WHERE path=?", (path,))
            connection.execute(
                "DELETE FROM search_chunks_fts WHERE chunk_id NOT IN (SELECT id FROM search_chunks)"
            )
            if cursor.rowcount:
                self._increment_generation(connection, "DOCUMENT_REMOVED")
            return cursor.rowcount > 0

    def rename_document(self, old_path: str, new_path: str) -> DocumentRecord:
        with self.transaction() as connection:
            cursor = connection.execute(
                "UPDATE documents SET path=? WHERE path=?", (new_path, old_path)
            )
            if not cursor.rowcount:
                raise KeyError(f"document not found: {old_path}")
            self._increment_generation(connection, "DOCUMENT_RENAMED")
        return self.get_document_by_path(new_path)

    def replace_document(self, record: DocumentRecord, parsed: ParsedDocument) -> int:
        return self.sync_document(record, parsed).index_generation

    def sync_document(self, record: DocumentRecord, parsed: ParsedDocument) -> SyncStats:
        """Compatibility single-document atomic replacement used by repository-level callers."""
        old_sections = {
            str(row[0]): str(row[1])
            for row in self.connection.execute(
                "SELECT id,sha256 FROM sections WHERE document_id=?", (record.id,)
            ).fetchall()
        }
        incoming = {item.id: item.sha256 for item in parsed.sections}
        with self.transaction() as connection:
            self._delete_document_derived(connection, record.id)
            self._insert_parsed(connection, record, parsed)
            connection.execute(
                "UPDATE documents SET current_sha256=?,indexed_at=? WHERE id=?",
                (parsed.sha256, datetime.now(UTC).isoformat(), record.id),
            )
            generation = self._increment_generation(connection, "DOCUMENT_SYNC")
        return SyncStats(
            sections_added=len(set(incoming) - set(old_sections)),
            sections_changed=sum(
                old_sections[key] != incoming[key] for key in set(incoming) & set(old_sections)
            ),
            sections_unchanged=sum(
                old_sections[key] == incoming[key] for key in set(incoming) & set(old_sections)
            ),
            sections_removed=len(set(old_sections) - set(incoming)),
            chunks_added=len(parsed.chunks),
            index_generation=generation,
        )

    @staticmethod
    def _delete_document_derived(connection: sqlite3.Connection, doc_id: str) -> None:
        chunk_rows = connection.execute(
            "SELECT c.id FROM search_chunks c JOIN sections s ON s.id=c.section_id "
            "WHERE s.document_id=?",
            (doc_id,),
        ).fetchall()
        connection.executemany(
            "DELETE FROM search_chunks_fts WHERE chunk_id=?", ((row[0],) for row in chunk_rows)
        )
        connection.execute("DELETE FROM document_versions WHERE document_id=?", (doc_id,))

    @staticmethod
    def _insert_parsed(
        connection: sqlite3.Connection, record: DocumentRecord, parsed: ParsedDocument
    ) -> None:
        cursor = connection.execute(
            "INSERT INTO document_versions(document_id,sha256,parser_version,chunker_version,"
            "created_at) VALUES(?,?,?,?,?)",
            (
                record.id,
                parsed.sha256,
                parsed.parser_version,
                parsed.chunker_version,
                datetime.now(UTC).isoformat(),
            ),
        )
        version_id = int(cursor.lastrowid or 0)
        connection.executemany(
            "INSERT INTO sections(id,document_id,document_version_id,ordinal,level,heading,"
            "heading_path,parent_id,start_line,end_line,start_offset,end_offset,text,sha256) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                (
                    item.id,
                    record.id,
                    version_id,
                    item.ordinal,
                    item.level,
                    item.heading,
                    json.dumps(item.heading_path, ensure_ascii=False),
                    item.parent_id,
                    item.start_line,
                    item.end_line,
                    item.start_offset,
                    item.end_offset,
                    item.text,
                    item.sha256,
                )
                for item in parsed.sections
            ),
        )
        headings = {item.id: canonical_heading(item.heading) for item in parsed.sections}
        connection.executemany(
            "INSERT INTO search_chunks(id,section_id,ordinal,start_line,end_line,start_column,"
            "end_column,start_offset,end_offset,source_text,source_sha256,embedding_text,"
            "embedding_sha256,chunker_version,token_estimate) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                (
                    item.id,
                    item.section_id,
                    item.ordinal,
                    item.start_line,
                    item.end_line,
                    item.start_column,
                    item.end_column,
                    item.start_offset,
                    item.end_offset,
                    item.source_text,
                    item.source_sha256,
                    item.embedding_text,
                    item.embedding_sha256,
                    item.chunker_version,
                    item.token_estimate,
                )
                for item in parsed.chunks
            ),
        )
        connection.executemany(
            "INSERT INTO search_chunks_fts(chunk_id,section_id,heading,source_text) "
            "VALUES(?,?,?,?)",
            (
                (item.id, item.section_id, headings[item.section_id], item.source_text)
                for item in parsed.chunks
            ),
        )

    def apply_workspace(
        self,
        records: Sequence[DocumentRecord],
        parsed: dict[str, ParsedDocument],
        graph: ExtractedGraph,
        checkpoints: tuple[CheckpointMetadata, ...],
        embeddings: list[tuple[str, str, str, NDArray[np.float32]]],
        *,
        embedding_identity: str,
        embedding_dimensions: int,
        embedding_metadata: dict[str, str],
        behavior_metadata: dict[str, str],
        reason: str,
        behavior_fingerprint: str,
    ) -> int:
        """Apply one complete logical generation atomically."""
        now = datetime.now(UTC).isoformat()
        incoming_ids = {record.id for record in records}
        with self.transaction() as connection:
            if incoming_ids:
                placeholders = ",".join("?" for _ in incoming_ids)
                connection.execute(
                    f"DELETE FROM documents WHERE id NOT IN ({placeholders})",  # noqa: S608
                    tuple(sorted(incoming_ids)),
                )
            else:
                connection.execute("DELETE FROM documents")
            for record in records:
                connection.execute(
                    "INSERT INTO documents(id,path,authority,priority,current_sha256,indexed_at) "
                    "VALUES(?,?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET path=excluded.path,"
                    "authority=excluded.authority,priority=excluded.priority,"
                    "current_sha256=excluded.current_sha256,indexed_at=excluded.indexed_at",
                    (
                        record.id,
                        record.path,
                        int(record.authority),
                        record.priority,
                        parsed[record.id].sha256,
                        now,
                    ),
                )
            connection.execute("DELETE FROM search_chunks_fts")
            connection.execute("DELETE FROM document_versions")
            for record in records:
                self._insert_parsed(connection, record, parsed[record.id])
            self._insert_graph(connection, graph)
            self._insert_checkpoints(connection, checkpoints)
            if embeddings:
                provider = embedding_metadata.get("provider", "unknown")
                model_name = embedding_metadata.get("model_name", "unknown")
                revision = embedding_metadata.get("revision", "unknown")
                artifact = embedding_metadata.get("artifact_sha256", "")
                runtime = embedding_metadata.get("runtime_version", "unknown")
                connection.executemany(
                    "INSERT INTO embeddings(chunk_id,embedding_identity,provider,model_name,"
                    "revision,artifact_sha256,runtime_version,dimensions,vector,chunk_source_sha256,"
                    "embedding_text_sha256) VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        (
                            chunk_id,
                            embedding_identity,
                            provider,
                            model_name,
                            revision,
                            artifact,
                            runtime,
                            embedding_dimensions,
                            np.asarray(vector, dtype="<f4").tobytes(),
                            source_hash,
                            embedding_hash,
                        )
                        for chunk_id, source_hash, embedding_hash, vector in embeddings
                    ),
                )
            for key, value in {
                **behavior_metadata,
                "embedding_identity": embedding_identity,
            }.items():
                connection.execute(
                    "INSERT INTO index_metadata(key,value) VALUES(?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                    (key, value),
                )
            return self._increment_generation(connection, reason, behavior_fingerprint)

    @staticmethod
    def _insert_graph(connection: sqlite3.Connection, graph: ExtractedGraph) -> None:
        connection.execute('DELETE FROM "references"')
        connection.execute("DELETE FROM symbols")
        connection.executemany(
            'INSERT INTO "references"(source_section_id,target_section_id,candidate_target_ids,'
            "edge_type,label,status,reason,evidence,origin) VALUES(?,?,?,?,?,?,?,?,?)",
            (
                (
                    edge.source_section_id,
                    edge.target_section_id,
                    json.dumps(edge.candidate_target_ids),
                    edge.edge_type.value,
                    edge.label,
                    edge.status.value,
                    edge.reason,
                    edge.evidence,
                    edge.origin.value,
                )
                for edge in graph.edges
            ),
        )
        connection.executemany(
            "INSERT INTO symbols(symbol,section_id,kind,origin,confidence) VALUES(?,?,?,?,?)",
            (
                (item.symbol, item.section_id, item.kind, item.origin.value, item.confidence)
                for item in graph.symbols
            ),
        )

    @staticmethod
    def _insert_checkpoints(
        connection: sqlite3.Connection, checkpoints: tuple[CheckpointMetadata, ...]
    ) -> None:
        connection.execute("DELETE FROM checkpoints")
        connection.executemany(
            "INSERT INTO checkpoints(document_id,checkpoint_id,root_section_id,title,fields_json) "
            "VALUES(?,?,?,?,?)",
            (
                (
                    item.document_id,
                    item.checkpoint_id,
                    item.root_section_id,
                    item.title,
                    item.model_dump_json(),
                )
                for item in checkpoints
            ),
        )

    def _provenance(
        self,
        row: sqlite3.Row,
        generation: int,
        *,
        excerpt: bool,
    ) -> Provenance:
        if excerpt:
            start_line = int(row["chunk_start_line"])
            end_line = int(row["chunk_end_line"])
            start_column = int(row["start_column"])
            end_column = int(row["end_column"]) if row["end_column"] is not None else None
            start_offset = int(row["chunk_start_offset"])
            end_offset = int(row["chunk_end_offset"])
            range_hash = str(row["source_sha256"])
        else:
            start_line = int(row["start_line"])
            end_line = int(row["end_line"])
            start_column = 0
            end_column = None
            start_offset = int(row["start_offset"])
            end_offset = int(row["end_offset"])
            range_hash = str(row["sha256"])
        return Provenance(
            document_id=str(row["document_id"]),
            document_path=str(row["path"]),
            document_sha256=str(row["current_sha256"]),
            authority=Authority(row["authority"]),
            priority=int(row["priority"]),
            section_id=str(row["id"]),
            heading_path=tuple(json.loads(row["heading_path"])),
            section_start_line=int(row["start_line"]),
            section_end_line=int(row["end_line"]),
            start_line=start_line,
            end_line=end_line,
            start_column=start_column,
            end_column=end_column,
            start_offset=start_offset,
            end_offset=end_offset,
            section_sha256=str(row["sha256"]),
            range_sha256=range_hash,
            index_generation=generation,
        )

    def _source_item(self, row: sqlite3.Row, generation: int | None = None) -> SourceSection:
        generation = self.index_generation() if generation is None else generation
        return SourceSection(
            text=str(row["text"]), provenance=self._provenance(row, generation, excerpt=False)
        )

    def _excerpt_item(self, row: sqlite3.Row, generation: int) -> SourceExcerpt:
        return SourceExcerpt(
            text=str(row["source_text"]),
            provenance=self._provenance(row, generation, excerpt=True),
        )

    @staticmethod
    def _source_select() -> str:
        return (
            "s.*,d.path,d.authority,d.priority,d.current_sha256,"
            "c.id AS chunk_id,c.start_line AS chunk_start_line,c.end_line AS chunk_end_line,"
            "c.start_column,c.end_column,c.start_offset AS chunk_start_offset,"
            "c.end_offset AS chunk_end_offset,c.source_text,c.source_sha256"
        )

    @staticmethod
    def _filter_sql(filters: FilterSet | None, alias: str = "d") -> tuple[str, list[object]]:
        if filters is None:
            return "", []
        clauses: list[str] = []
        values: list[object] = []
        if filters.documents:
            clauses.append(
                f"({alias}.path IN ({','.join('?' for _ in filters.documents)}) OR "
                f"{alias}.id IN ({','.join('?' for _ in filters.documents)}))"
            )
            ordered = sorted(filters.documents)
            values.extend(ordered)
            values.extend(ordered)
        if filters.exclude_documents:
            clauses.append(
                f"{alias}.path NOT IN ({','.join('?' for _ in filters.exclude_documents)})"
            )
            values.extend(sorted(filters.exclude_documents))
        if filters.authority_floor is not None:
            clauses.append(f"{alias}.authority>=?")
            values.append(int(filters.authority_floor))
        if filters.authorities:
            clauses.append(f"{alias}.authority IN ({','.join('?' for _ in filters.authorities)})")
            values.extend(int(item) for item in sorted(filters.authorities))
        if filters.scope:
            clauses.append(f"{alias}.path LIKE ? ESCAPE '\\'")
            escaped = filters.scope.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            values.append(escaped.rstrip("/") + "/%")
        if filters.heading_prefix:
            clauses.append(
                "(ctx_heading_path_key(s.heading_path)=? OR "
                "ctx_heading_path_key(s.heading_path) LIKE ? ESCAPE '\\')"
            )
            prefix = "/".join(canonical_heading(part) for part in filters.heading_prefix)
            escaped = prefix.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            values.extend((prefix, escaped + "/%"))
        return (" AND " + " AND ".join(clauses) if clauses else ""), values

    def get_section(self, section_id: str) -> SourceSection:
        generation = self.index_generation()
        row = self.connection.execute(
            "SELECT s.*,d.path,d.authority,d.priority,d.current_sha256 FROM sections s "
            "JOIN documents d ON d.id=s.document_id WHERE s.id=?",
            (section_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"section not found: {section_id}")
        return self._source_item(row, generation)

    def structural_search(
        self,
        terms: tuple[str, ...],
        limit: int = 10,
        *,
        filters: FilterSet | None = None,
        include_full_section: bool = False,
    ) -> list[SearchHit]:
        generation = self.index_generation()
        filter_sql, filter_values = self._filter_sql(filters)
        best: dict[tuple[str, str], tuple[sqlite3.Row, int, str]] = {}
        for term in terms:
            canonical_term = canonical_heading(term)
            escaped = self._like_prefix(canonical_term)
            rows = self.connection.execute(
                f"SELECT {self._source_select()},CASE WHEN ctx_heading_key(s.heading)=? THEN 0 "
                "WHEN ctx_heading_key(s.heading) LIKE ? ESCAPE '\\' THEN 1 ELSE 2 END "
                "AS structural_rank FROM search_chunks c JOIN sections s ON s.id=c.section_id "
                "JOIN documents d ON d.id=s.document_id WHERE (ctx_heading_key(s.heading)=? "
                "OR ctx_heading_key(s.heading) LIKE ? ESCAPE '\\' OR instr(c.source_text,?)>0 "
                "OR d.path=?)"
                f"{filter_sql} ORDER BY structural_rank,d.path,s.ordinal,c.ordinal LIMIT ?",
                (
                    canonical_term,
                    escaped,
                    canonical_term,
                    escaped,
                    term,
                    term,
                    *filter_values,
                    min(limit * 20, 1000),
                ),
            ).fetchall()
            for row in rows:
                section_id = str(row["id"])
                rank = int(row["structural_rank"])
                # Heading matches identify a section once; source matches identify exact spans.
                key = (section_id, "heading" if rank < 2 else str(row["chunk_id"]))
                previous = best.get(key)
                if previous is None or rank < previous[1]:
                    best[key] = (row, rank, term)
        ordered = sorted(
            best.values(),
            key=lambda item: (
                item[1],
                -int(item[0]["authority"]),
                -int(item[0]["priority"]),
                str(item[0]["path"]),
                int(item[0]["ordinal"]),
            ),
        )
        hits: list[SearchHit] = []
        for row, rank, term in ordered:
            source: SourceItem = (
                self._source_item(row, generation)
                if include_full_section
                else self._excerpt_item(row, generation)
            )
            hits.append(
                SearchHit(
                    source=source,
                    score=float(100 - rank),
                    channels=("structural",),
                    matched_terms=(term,),
                    chunk_id=str(row["chunk_id"]),
                    match_start_line=int(row["chunk_start_line"]),
                    match_end_line=int(row["chunk_end_line"]),
                    index_generation=generation,
                )
            )
        return diversify_spans(hits, limit)

    @staticmethod
    def _like_prefix(term: str) -> str:
        return term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"

    def lexical_search(
        self,
        query: str,
        limit: int = 10,
        *,
        filters: FilterSet | None = None,
        include_full_section: bool = False,
    ) -> list[SearchHit]:
        terms = re.findall(r"[\w][\w.@/:-]*", query, flags=re.UNICODE)
        if not terms or limit < 1:
            return []
        generation = self.index_generation()
        expression = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        filter_sql, filter_values = self._filter_sql(filters)
        rows = self.connection.execute(
            f"SELECT {self._source_select()},bm25(search_chunks_fts,0.0,0.0,8.0,1.0) "
            "AS lexical_rank FROM search_chunks_fts JOIN search_chunks c "
            "ON c.id=search_chunks_fts.chunk_id JOIN sections s ON s.id=c.section_id "
            "JOIN documents d ON d.id=s.document_id WHERE search_chunks_fts MATCH ?"
            f"{filter_sql} ORDER BY lexical_rank,d.path,s.ordinal,c.ordinal LIMIT ?",
            (expression, *filter_values, min(limit * 8, 800)),
        ).fetchall()
        exact_rows = self.connection.execute(
            f"SELECT {self._source_select()},0.0 AS lexical_rank FROM search_chunks c "
            "JOIN sections s ON s.id=c.section_id JOIN documents d ON d.id=s.document_id "
            "WHERE (instr(s.heading,?)>0 OR instr(c.source_text,?)>0)"
            f"{filter_sql} ORDER BY d.path,s.ordinal,c.ordinal LIMIT ?",
            (query, query, *filter_values, min(limit * 4, 400)),
        ).fetchall()
        by_chunk: dict[str, sqlite3.Row] = {}
        for row in [*rows, *exact_rows]:
            by_chunk.setdefault(str(row["chunk_id"]), row)
        query_folded = query.strip().casefold()
        identifier = re.compile(rf"(?<![\w]){re.escape(query.strip())}(?![\w])")
        hits: list[SearchHit] = []
        for row in by_chunk.values():
            exact_heading = canonical_heading(str(row["heading"])) == canonical_heading(
                query_folded
            )
            exact_identifier = bool(query.strip()) and bool(
                identifier.search(str(row["source_text"]))
            )
            score = (
                -float(row["lexical_rank"])
                + (4 if exact_heading else 0)
                + (2 if exact_identifier else 0)
            )
            source = (
                self._source_item(row, generation)
                if include_full_section
                else self._excerpt_item(row, generation)
            )
            hits.append(
                SearchHit(
                    source=source,
                    score=score,
                    channels=("bm25",),
                    matched_terms=tuple(
                        term
                        for term in terms
                        if term.casefold() in str(row["source_text"]).casefold()
                    ),
                    chunk_id=str(row["chunk_id"]),
                    match_start_line=int(row["chunk_start_line"]),
                    match_end_line=int(row["chunk_end_line"]),
                    index_generation=generation,
                )
            )
        hits.sort(
            key=lambda hit: (
                -hit.score,
                -int(hit.source.provenance.authority),
                -hit.source.provenance.priority,
                hit.source.provenance.document_path,
                hit.source.provenance.start_line,
                hit.chunk_id or "",
            )
        )
        return diversify_spans(hits, limit)

    def semantic_search(
        self,
        query_vector: NDArray[np.float32],
        identity: str,
        limit: int = 10,
        *,
        filters: FilterSet | None = None,
        include_full_section: bool = False,
    ) -> list[SearchHit]:
        generation = self.index_generation()
        rows, matrix = self.load_embeddings(identity, filters=filters, with_rows=True)
        if not rows:
            return []
        scores = cosine_scores(query_vector, matrix)
        ranked = sorted(
            range(len(rows)),
            key=lambda index: (-float(scores[index]), str(rows[index]["chunk_id"])),
        )
        result: list[SearchHit] = []
        for index in ranked:
            row = rows[index]
            source = (
                self._source_item(row, generation)
                if include_full_section
                else self._excerpt_item(row, generation)
            )
            result.append(
                SearchHit(
                    source=source,
                    score=float(scores[index]),
                    channels=("semantic",),
                    chunk_id=str(row["chunk_id"]),
                    match_start_line=int(row["chunk_start_line"]),
                    match_end_line=int(row["chunk_end_line"]),
                    index_generation=generation,
                )
            )
        return diversify_spans(result, limit)

    def graph_sections(self) -> list[GraphSection]:
        rows = self.connection.execute(
            "SELECT s.id,s.parent_id,s.heading,s.text,s.document_id,s.ordinal,"
            "d.authority,d.priority FROM sections s JOIN documents d ON d.id=s.document_id "
            "ORDER BY s.document_id,s.ordinal"
        ).fetchall()
        return [
            GraphSection(
                id=str(row["id"]),
                parent_id=str(row["parent_id"]) if row["parent_id"] is not None else None,
                heading=str(row["heading"]),
                text=str(row["text"]),
                document_id=str(row["document_id"]),
                authority=Authority(row["authority"]),
                priority=int(row["priority"]),
                ordinal=int(row["ordinal"]),
            )
            for row in rows
        ]

    def replace_graph(self, graph: ExtractedGraph) -> None:
        with self.transaction() as connection:
            self._insert_graph(connection, graph)

    def get_references(
        self,
        section_id: str,
        *,
        edge_types: tuple[EdgeType, ...] | None = None,
        incoming: bool = False,
    ) -> list[ReferenceResult]:
        column = "target_section_id" if incoming else "source_section_id"
        parameters: list[object] = [section_id]
        sql = f'SELECT * FROM "references" WHERE {column}=?'  # noqa: S608
        if edge_types:
            sql += " AND edge_type IN (" + ",".join("?" for _ in edge_types) + ")"
            parameters.extend(edge.value for edge in edge_types)
        sql += " ORDER BY edge_type,label,source_section_id"
        rows = self.connection.execute(sql, parameters).fetchall()
        results: list[ReferenceResult] = []
        for row in rows:
            edge = ReferenceRecord(
                source_section_id=str(row["source_section_id"]),
                target_section_id=str(row["target_section_id"])
                if row["target_section_id"]
                else None,
                candidate_target_ids=tuple(json.loads(row["candidate_target_ids"])),
                edge_type=EdgeType(row["edge_type"]),
                label=str(row["label"]),
                status=ResolutionStatus(row["status"]),
                reason=str(row["reason"]),
                evidence=str(row["evidence"]),
                origin=ReferenceOrigin(row["origin"]),
            )
            candidates: list[SourceRef] = []
            for candidate_id in edge.candidate_target_ids:
                with suppress(KeyError):
                    candidates.append(self.get_section(candidate_id).ref)
            results.append(
                ReferenceResult(
                    edge=edge,
                    source=self.get_section(edge.source_section_id),
                    target=self.get_section(edge.target_section_id)
                    if edge.target_section_id
                    else None,
                    candidates=tuple(candidates),
                )
            )
        return results

    def find_symbol(
        self, symbol: str, limit: int = 20, *, filters: FilterSet | None = None
    ) -> list[SymbolResult]:
        filter_sql, filter_values = self._filter_sql(filters)
        rows = self.connection.execute(
            "SELECT y.symbol,y.section_id,y.kind,y.origin,y.confidence FROM symbols y "
            "JOIN sections s ON s.id=y.section_id JOIN documents d ON d.id=s.document_id "
            f"WHERE y.symbol=? COLLATE NOCASE{filter_sql} "
            "ORDER BY y.confidence DESC,d.authority DESC,d.priority DESC,y.section_id LIMIT ?",
            (symbol, *filter_values, limit),
        ).fetchall()
        return [
            SymbolResult(
                symbol=str(row["symbol"]),
                kind=str(row["kind"]),
                origin=ReferenceOrigin(row["origin"]),
                confidence=float(row["confidence"]),
                source=self.get_section(str(row["section_id"])),
            )
            for row in rows
        ]

    def replace_checkpoints(self, checkpoints: tuple[CheckpointMetadata, ...]) -> None:
        with self.transaction() as connection:
            self._insert_checkpoints(connection, checkpoints)

    def checkpoint_candidates(
        self, checkpoint_id: str, document: str | None = None
    ) -> list[tuple[CheckpointMetadata, DocumentRecord]]:
        parameters: list[object] = [checkpoint_id]
        where = "c.checkpoint_id=? COLLATE NOCASE"
        if document:
            where += " AND (d.path=? OR d.id=?)"
            parameters.extend((document, document))
        rows = self.connection.execute(
            "SELECT c.fields_json,d.* FROM checkpoints c JOIN documents d ON d.id=c.document_id "
            f"WHERE {where} ORDER BY d.authority DESC,d.priority DESC,d.path",
            parameters,
        ).fetchall()
        return [
            (CheckpointMetadata.model_validate_json(row["fields_json"]), self._document(row))
            for row in rows
        ]

    def get_checkpoint_metadata(
        self, checkpoint_id: str, document: str | None = None
    ) -> CheckpointMetadata:
        candidates = self.checkpoint_candidates(checkpoint_id, document)
        if not candidates:
            raise KeyError(f"checkpoint not found: {checkpoint_id}")
        return candidates[0][0]

    def list_checkpoint_metadata(self) -> list[CheckpointMetadata]:
        return [
            CheckpointMetadata.model_validate_json(row[0])
            for row in self.connection.execute(
                "SELECT fields_json FROM checkpoints ORDER BY checkpoint_id,document_id"
            ).fetchall()
        ]

    def chunks_needing_embeddings(self, identity: str) -> list[tuple[str, str, str]]:
        rows = self.connection.execute(
            "SELECT c.id,c.embedding_text,c.source_sha256 FROM search_chunks c "
            "LEFT JOIN embeddings e ON e.chunk_id=c.id WHERE e.chunk_id IS NULL "
            "OR e.embedding_identity<>? OR e.chunk_source_sha256<>c.source_sha256 "
            "OR e.embedding_text_sha256<>c.embedding_sha256 "
            "ORDER BY c.id",
            (identity,),
        ).fetchall()
        return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]

    def existing_embedding_vectors(
        self, identity: str
    ) -> dict[tuple[str, str, str], NDArray[np.float32]]:
        rows = self.connection.execute(
            "SELECT chunk_id,chunk_source_sha256,embedding_text_sha256,dimensions,vector "
            "FROM embeddings WHERE embedding_identity=?",
            (identity,),
        ).fetchall()
        return {
            (
                str(row["chunk_id"]),
                str(row["chunk_source_sha256"]),
                str(row["embedding_text_sha256"]),
            ): np.frombuffer(row["vector"], dtype="<f4", count=int(row["dimensions"])).copy()
            for row in rows
        }

    def save_embeddings(
        self,
        model: str,
        dimensions: int,
        rows: list[tuple[str, str, NDArray[np.float32]]],
    ) -> int:
        with self.transaction() as connection:
            connection.executemany(
                "INSERT INTO embeddings(chunk_id,embedding_identity,provider,model_name,revision,"
                "artifact_sha256,runtime_version,dimensions,vector,chunk_source_sha256,"
                "embedding_text_sha256) SELECT ?,?,'legacy',?,'','','',?,?,?,embedding_sha256 "
                "FROM search_chunks WHERE id=? ON CONFLICT(chunk_id) DO UPDATE SET "
                "embedding_identity=excluded.embedding_identity,dimensions=excluded.dimensions,"
                "vector=excluded.vector,chunk_source_sha256=excluded.chunk_source_sha256,"
                "embedding_text_sha256=excluded.embedding_text_sha256",
                (
                    (
                        chunk_id,
                        model,
                        model,
                        dimensions,
                        np.asarray(vector, dtype="<f4").tobytes(),
                        chunk_hash,
                        chunk_id,
                    )
                    for chunk_id, chunk_hash, vector in rows
                ),
            )
            connection.execute(
                "INSERT INTO index_metadata(key,value) VALUES('embedding_identity',?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (model,),
            )
            return self._increment_generation(connection, "EMBEDDINGS_UPDATED")

    @overload
    def load_embeddings(
        self,
        identity: str,
        *,
        filters: FilterSet | None = None,
        with_rows: Literal[False] = False,
    ) -> tuple[list[str], NDArray[np.float32]]: ...

    @overload
    def load_embeddings(
        self,
        identity: str,
        *,
        filters: FilterSet | None = None,
        with_rows: Literal[True],
    ) -> tuple[list[sqlite3.Row], NDArray[np.float32]]: ...

    def load_embeddings(
        self,
        identity: str,
        *,
        filters: FilterSet | None = None,
        with_rows: bool = False,
    ) -> tuple[list[str], NDArray[np.float32]] | tuple[list[sqlite3.Row], NDArray[np.float32]]:
        filter_sql, filter_values = self._filter_sql(filters)
        rows = self.connection.execute(
            f"SELECT {self._source_select()},e.dimensions,e.vector FROM embeddings e "
            "JOIN search_chunks c ON c.id=e.chunk_id JOIN sections s ON s.id=c.section_id "
            "JOIN documents d ON d.id=s.document_id WHERE e.embedding_identity=?"
            f"{filter_sql} ORDER BY e.chunk_id",
            (identity, *filter_values),
        ).fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)
        chunk_ids = tuple(str(row["chunk_id"]) for row in rows)
        filter_key = json.dumps(
            filters.model_dump(mode="json") if filters else {},
            ensure_ascii=False,
            sort_keys=True,
        )
        cache_key = (self.index_generation(), identity, filter_key)
        with self._matrix_cache_lock:
            cached = self._matrix_cache.get(cache_key)
            if cached is not None and cached[0] == chunk_ids:
                self._matrix_cache.move_to_end(cache_key)
                matrix = cached[1]
            else:
                dimensions = int(rows[0]["dimensions"])
                matrix = np.stack(
                    [np.frombuffer(row["vector"], dtype="<f4", count=dimensions) for row in rows]
                )
                matrix.setflags(write=False)
                self._matrix_cache[cache_key] = (chunk_ids, matrix)
                self._matrix_cache.move_to_end(cache_key)
                while len(self._matrix_cache) > 4:
                    self._matrix_cache.popitem(last=False)
        if with_rows:
            return list(rows), matrix
        return list(chunk_ids), matrix

    def document_sections(self, path: str) -> list[SourceSection]:
        generation = self.index_generation()
        rows = self.connection.execute(
            "SELECT s.*,d.path,d.authority,d.priority,d.current_sha256 FROM sections s "
            "JOIN documents d ON d.id=s.document_id WHERE d.path=? ORDER BY s.ordinal",
            (path,),
        ).fetchall()
        if (
            not rows
            and not self.connection.execute(
                "SELECT 1 FROM documents WHERE path=?", (path,)
            ).fetchone()
        ):
            raise KeyError(f"document not found: {path}")
        return [self._source_item(row, generation) for row in rows]

    def document_outline(self, path: str) -> list[OutlineEntry]:
        generation = self.index_generation()
        rows = self.connection.execute(
            "SELECT s.id,s.heading,s.heading_path,s.level,s.ordinal,s.start_line,s.end_line,"
            "s.sha256,s.document_id,d.path,d.current_sha256,d.authority,d.priority FROM sections s "
            "JOIN documents d ON d.id=s.document_id WHERE d.path=? ORDER BY s.ordinal",
            (path,),
        ).fetchall()
        if (
            not rows
            and not self.connection.execute(
                "SELECT 1 FROM documents WHERE path=?", (path,)
            ).fetchone()
        ):
            raise KeyError(f"document not found: {path}")
        return [
            OutlineEntry(
                section_id=str(row["id"]),
                heading=str(row["heading"]),
                heading_path=tuple(json.loads(row["heading_path"])),
                level=int(row["level"]),
                ordinal=int(row["ordinal"]),
                start_line=int(row["start_line"]),
                end_line=int(row["end_line"]),
                section_sha256=str(row["sha256"]),
                document_id=str(row["document_id"]),
                document_path=str(row["path"]),
                document_sha256=str(row["current_sha256"]),
                authority=Authority(row["authority"]),
                priority=int(row["priority"]),
                index_generation=generation,
            )
            for row in rows
        ]

    def section_ids(self, document_id_value: str | None = None) -> list[str]:
        if document_id_value is None:
            rows = self.connection.execute(
                "SELECT id FROM sections ORDER BY document_id,ordinal"
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT id FROM sections WHERE document_id=? ORDER BY ordinal", (document_id_value,)
            ).fetchall()
        return [str(row[0]) for row in rows]

    def chunk_count(self, section_id: str | None = None) -> int:
        if section_id:
            row = self.connection.execute(
                "SELECT COUNT(*) FROM search_chunks WHERE section_id=?", (section_id,)
            ).fetchone()
        else:
            row = self.connection.execute("SELECT COUNT(*) FROM search_chunks").fetchone()
        return int(row[0])
