"""SQLite persistence, migrations, and provenance-preserving source retrieval."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from ctx.models import Authority, DocumentRecord, ParsedDocument, Provenance, SourceItem

SCHEMA_VERSION = 1

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
    embedding_model TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    UNIQUE(document_id, sha256, parser_version, embedding_model)
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
    text TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    UNIQUE(document_id, ordinal)
);
CREATE TABLE IF NOT EXISTS chunks (
    id TEXT PRIMARY KEY,
    section_id TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL,
    start_line INTEGER NOT NULL,
    end_line INTEGER NOT NULL,
    text TEXT NOT NULL,
    sha256 TEXT NOT NULL,
    UNIQUE(section_id, ordinal)
);
CREATE TABLE IF NOT EXISTS embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    model TEXT NOT NULL,
    dimensions INTEGER NOT NULL,
    vector BLOB NOT NULL,
    chunk_sha256 TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS "references" (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_section_id TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    target_section_id TEXT REFERENCES sections(id) ON DELETE SET NULL,
    edge_type TEXT NOT NULL,
    label TEXT NOT NULL,
    resolved INTEGER NOT NULL CHECK(resolved IN (0, 1)),
    UNIQUE(source_section_id, target_section_id, edge_type, label)
);
CREATE TABLE IF NOT EXISTS symbols (
    symbol TEXT NOT NULL,
    section_id TEXT NOT NULL REFERENCES sections(id) ON DELETE CASCADE,
    kind TEXT NOT NULL,
    PRIMARY KEY(symbol, section_id, kind)
);
CREATE TABLE IF NOT EXISTS checkpoint_metadata (
    section_id TEXT PRIMARY KEY REFERENCES sections(id) ON DELETE CASCADE,
    checkpoint_id TEXT NOT NULL,
    title TEXT NOT NULL,
    fields_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS index_metadata (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    section_id UNINDEXED,
    heading,
    text,
    tokenize='unicode61 tokenchars ''._@-'''
);
CREATE INDEX IF NOT EXISTS idx_sections_document ON sections(document_id, ordinal);
CREATE INDEX IF NOT EXISTS idx_chunks_section ON chunks(section_id);
CREATE INDEX IF NOT EXISTS idx_references_source ON "references"(source_section_id);
CREATE INDEX IF NOT EXISTS idx_references_target ON "references"(target_section_id);
CREATE INDEX IF NOT EXISTS idx_symbols_symbol ON symbols(symbol);
"""


def document_id(path: str) -> str:
    return f"doc:{hashlib.sha256(path.encode()).hexdigest()[:16]}"


class SQLiteStore:
    """Small synchronous repository used by both CLI and MCP application services."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.connection = sqlite3.connect(path)
        self.connection.row_factory = sqlite3.Row
        self.connection.execute("PRAGMA foreign_keys = ON")
        self.connection.execute("PRAGMA journal_mode = WAL")
        self.migrate()

    def close(self) -> None:
        self.connection.close()

    def __enter__(self) -> SQLiteStore:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self.connection:
            yield self.connection

    def migrate(self) -> None:
        current = int(self.connection.execute("PRAGMA user_version").fetchone()[0])
        if current > SCHEMA_VERSION:
            raise RuntimeError(f"index schema {current} is newer than supported {SCHEMA_VERSION}")
        if current < 1:
            with self.connection:
                self.connection.executescript(_SCHEMA)
                self.connection.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
                self.connection.execute(
                    "INSERT OR IGNORE INTO index_metadata(key, value) VALUES(?, ?)",
                    ("index_version", "0"),
                )

    def index_version(self) -> int:
        row = self.connection.execute(
            "SELECT value FROM index_metadata WHERE key = ?", ("index_version",)
        ).fetchone()
        return int(row[0]) if row else 0

    def set_metadata(self, key: str, value: str) -> None:
        with self.connection:
            self.connection.execute(
                "INSERT INTO index_metadata(key, value) VALUES(?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )

    def get_metadata(self, key: str) -> str | None:
        row = self.connection.execute(
            "SELECT value FROM index_metadata WHERE key = ?", (key,)
        ).fetchone()
        return str(row[0]) if row else None

    def register_document(self, path: str, authority: Authority, priority: int) -> DocumentRecord:
        identifier = document_id(path)
        with self.connection:
            self.connection.execute(
                "INSERT INTO documents(id, path, authority, priority) VALUES(?, ?, ?, ?) "
                "ON CONFLICT(path) DO UPDATE SET authority=excluded.authority, "
                "priority=excluded.priority",
                (identifier, path, int(authority), priority),
            )
        return self.get_document_by_path(path)

    def get_document_by_path(self, path: str) -> DocumentRecord:
        row = self.connection.execute("SELECT * FROM documents WHERE path = ?", (path,)).fetchone()
        if row is None:
            raise KeyError(f"document not found: {path}")
        return self._document(row)

    def list_documents(self) -> list[DocumentRecord]:
        rows = self.connection.execute("SELECT * FROM documents ORDER BY path").fetchall()
        return [self._document(row) for row in rows]

    @staticmethod
    def _document(row: sqlite3.Row) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            path=row["path"],
            authority=Authority(row["authority"]),
            priority=row["priority"],
            sha256=row["current_sha256"],
            indexed_at=row["indexed_at"],
        )

    def remove_document(self, path: str) -> bool:
        with self.connection:
            cursor = self.connection.execute("DELETE FROM documents WHERE path = ?", (path,))
            if cursor.rowcount:
                self._increment_index_version(self.connection)
            return cursor.rowcount > 0

    def replace_document(self, record: DocumentRecord, parsed: ParsedDocument) -> int:
        """Atomically replace derived rows for one document and return index version."""
        now = datetime.now(UTC).isoformat()
        with self.transaction() as connection:
            version = connection.execute(
                "SELECT id FROM document_versions WHERE document_id=? AND sha256=? "
                "AND parser_version=? AND embedding_model=''",
                (record.id, parsed.sha256, parsed.parser_version),
            ).fetchone()
            if version is None:
                cursor = connection.execute(
                    "INSERT INTO document_versions(document_id, sha256, parser_version, "
                    "embedding_model, created_at) VALUES(?, ?, ?, '', ?)",
                    (record.id, parsed.sha256, parsed.parser_version, now),
                )
                if cursor.lastrowid is None:  # pragma: no cover - sqlite contract
                    raise RuntimeError("SQLite did not return a document version ID")
                version_id = cursor.lastrowid
            else:
                version_id = int(version[0])
            old_chunk_ids = connection.execute(
                "SELECT c.id FROM chunks c JOIN sections s ON s.id=c.section_id "
                "WHERE s.document_id=?",
                (record.id,),
            ).fetchall()
            connection.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id=?",
                ((row[0],) for row in old_chunk_ids),
            )
            connection.execute("DELETE FROM sections WHERE document_id=?", (record.id,))
            for section in parsed.sections:
                connection.execute(
                    "INSERT INTO sections(id, document_id, document_version_id, ordinal, level, "
                    "heading, heading_path, parent_id, start_line, end_line, text, sha256) "
                    "VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        section.id,
                        record.id,
                        version_id,
                        section.ordinal,
                        section.level,
                        section.heading,
                        json.dumps(section.heading_path, ensure_ascii=False),
                        section.parent_id,
                        section.start_line,
                        section.end_line,
                        section.text,
                        section.sha256,
                    ),
                )
            headings = {section.id: section.heading for section in parsed.sections}
            for chunk in parsed.chunks:
                connection.execute(
                    "INSERT INTO chunks(id, section_id, ordinal, start_line, end_line, text, "
                    "sha256) VALUES(?, ?, ?, ?, ?, ?, ?)",
                    (
                        chunk.id,
                        chunk.section_id,
                        chunk.ordinal,
                        chunk.start_line,
                        chunk.end_line,
                        chunk.text,
                        chunk.sha256,
                    ),
                )
                connection.execute(
                    "INSERT INTO chunks_fts(chunk_id, section_id, heading, text) "
                    "VALUES(?, ?, ?, ?)",
                    (chunk.id, chunk.section_id, headings[chunk.section_id], chunk.text),
                )
            connection.execute(
                "UPDATE documents SET current_sha256=?, indexed_at=? WHERE id=?",
                (parsed.sha256, now, record.id),
            )
            return self._increment_index_version(connection)

    @staticmethod
    def _increment_index_version(connection: sqlite3.Connection) -> int:
        row = connection.execute(
            "SELECT value FROM index_metadata WHERE key='index_version'"
        ).fetchone()
        version = int(row[0]) + 1
        connection.execute(
            "UPDATE index_metadata SET value=? WHERE key='index_version'", (str(version),)
        )
        return version

    def get_section(self, section_id: str) -> SourceItem:
        row = self.connection.execute(
            "SELECT s.*, d.path, d.authority, d.priority, d.current_sha256 "
            "FROM sections s JOIN documents d ON d.id=s.document_id WHERE s.id=?",
            (section_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"section not found: {section_id}")
        return SourceItem(
            text=row["text"],
            provenance=Provenance(
                document_id=row["document_id"],
                document_path=row["path"],
                document_sha256=row["current_sha256"],
                authority=Authority(row["authority"]),
                priority=row["priority"],
                section_id=row["id"],
                heading_path=tuple(json.loads(row["heading_path"])),
                start_line=row["start_line"],
                end_line=row["end_line"],
                section_sha256=row["sha256"],
                index_version=self.index_version(),
            ),
        )

    def section_ids(self, document_id_value: str | None = None) -> list[str]:
        if document_id_value is None:
            rows = self.connection.execute(
                "SELECT id FROM sections ORDER BY document_id, ordinal"
            ).fetchall()
        else:
            rows = self.connection.execute(
                "SELECT id FROM sections WHERE document_id=? ORDER BY ordinal",
                (document_id_value,),
            ).fetchall()
        return [str(row[0]) for row in rows]
