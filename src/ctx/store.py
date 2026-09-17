"""SQLite persistence, migrations, and provenance-preserving source retrieval."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ctx.checkpoints import CheckpointMetadata
from ctx.embeddings import cosine_scores
from ctx.graph import ExtractedGraph, GraphSection
from ctx.models import (
    Authority,
    DocumentRecord,
    EdgeType,
    ParsedDocument,
    Provenance,
    ReferenceRecord,
    ReferenceResult,
    SearchHit,
    SourceItem,
    SymbolResult,
    SyncStats,
)

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

    def touch_index(self) -> int:
        with self.connection:
            return self._increment_index_version(self.connection)

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

    def document_parser_version(self, document_id_value: str) -> str | None:
        row = self.connection.execute(
            "SELECT v.parser_version FROM document_versions v "
            "JOIN documents d ON d.id=v.document_id "
            "WHERE d.id=? AND v.sha256=d.current_sha256 "
            "ORDER BY v.id DESC LIMIT 1",
            (document_id_value,),
        ).fetchone()
        return str(row[0]) if row else None

    def remove_document(self, path: str) -> bool:
        with self.connection:
            chunks = self.connection.execute(
                "SELECT c.id FROM chunks c JOIN sections s ON s.id=c.section_id "
                "JOIN documents d ON d.id=s.document_id WHERE d.path=?",
                (path,),
            ).fetchall()
            self.connection.executemany(
                "DELETE FROM chunks_fts WHERE chunk_id=?", ((row[0],) for row in chunks)
            )
            cursor = self.connection.execute("DELETE FROM documents WHERE path = ?", (path,))
            if cursor.rowcount:
                self._increment_index_version(self.connection)
            return cursor.rowcount > 0

    def rename_document(self, old_path: str, new_path: str) -> DocumentRecord:
        """Rename a same-content document while retaining its durable identity."""
        with self.connection:
            cursor = self.connection.execute(
                "UPDATE documents SET path=? WHERE path=?", (new_path, old_path)
            )
            if not cursor.rowcount:
                raise KeyError(f"document not found: {old_path}")
            self._increment_index_version(self.connection)
        return self.get_document_by_path(new_path)

    def replace_document(self, record: DocumentRecord, parsed: ParsedDocument) -> int:
        """Compatibility wrapper for incremental replacement."""
        return self.sync_document(record, parsed).index_version

    def sync_document(self, record: DocumentRecord, parsed: ParsedDocument) -> SyncStats:
        """Update only changed structural material and preserve reusable embeddings."""
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

            existing_rows = connection.execute(
                "SELECT id, sha256 FROM sections WHERE document_id=?", (record.id,)
            ).fetchall()
            existing = {str(row["id"]): str(row["sha256"]) for row in existing_rows}
            incoming = {section.id: section for section in parsed.sections}
            removed = set(existing) - set(incoming)
            added = set(incoming) - set(existing)
            changed = {
                identifier
                for identifier in set(existing) & set(incoming)
                if existing[identifier] != incoming[identifier].sha256
            }
            unchanged = set(existing) & set(incoming) - changed
            chunks_removed = 0
            embeddings_retained = 0

            for section_id in removed | changed:
                chunks = connection.execute(
                    "SELECT id FROM chunks WHERE section_id=?", (section_id,)
                ).fetchall()
                chunks_removed += len(chunks)
                connection.executemany(
                    "DELETE FROM chunks_fts WHERE chunk_id=?", ((row[0],) for row in chunks)
                )
                connection.execute("DELETE FROM chunks WHERE section_id=?", (section_id,))
                connection.execute(
                    'DELETE FROM "references" WHERE source_section_id=?', (section_id,)
                )
                connection.execute("DELETE FROM symbols WHERE section_id=?", (section_id,))
                connection.execute(
                    "DELETE FROM checkpoint_metadata WHERE section_id=?", (section_id,)
                )
            for section_id in removed:
                connection.execute("DELETE FROM sections WHERE id=?", (section_id,))

            # Avoid transient UNIQUE(document_id, ordinal) collisions during reordered updates.
            connection.execute(
                "UPDATE sections SET ordinal = -ordinal - 1 WHERE document_id=?", (record.id,)
            )
            for section in parsed.sections:
                values = (
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
                    section.id,
                )
                if section.id in existing:
                    connection.execute(
                        "UPDATE sections SET document_version_id=?, ordinal=?, level=?, "
                        "heading=?, heading_path=?, parent_id=?, start_line=?, end_line=?, "
                        "text=?, sha256=? WHERE id=?",
                        values,
                    )
                else:
                    connection.execute(
                        "INSERT INTO sections(document_version_id, ordinal, level, heading, "
                        "heading_path, parent_id, start_line, end_line, text, sha256, id, "
                        "document_id) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (*values, record.id),
                    )

            chunks_by_section = {chunk.section_id: chunk for chunk in parsed.chunks}
            for section_id in unchanged:
                chunk = chunks_by_section[section_id]
                connection.execute(
                    "UPDATE chunks SET start_line=?, end_line=? WHERE id=?",
                    (chunk.start_line, chunk.end_line, chunk.id),
                )
                row = connection.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE chunk_id=? AND chunk_sha256=?",
                    (chunk.id, chunk.sha256),
                ).fetchone()
                embeddings_retained += int(row[0])
            for section_id in added | changed:
                chunk = chunks_by_section[section_id]
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
                    (chunk.id, chunk.section_id, incoming[section_id].heading, chunk.text),
                )
            connection.execute(
                "UPDATE documents SET current_sha256=?, indexed_at=? WHERE id=?",
                (parsed.sha256, now, record.id),
            )
            index_version = self._increment_index_version(connection)
            return SyncStats(
                sections_added=len(added),
                sections_changed=len(changed),
                sections_unchanged=len(unchanged),
                sections_removed=len(removed),
                chunks_added=len(added),
                chunks_changed=len(changed),
                chunks_unchanged=len(unchanged),
                chunks_removed=chunks_removed,
                embeddings_retained=embeddings_retained,
                index_version=index_version,
            )

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

    def _source_item(self, row: sqlite3.Row) -> SourceItem:
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

    def get_section(self, section_id: str) -> SourceItem:
        row = self.connection.execute(
            "SELECT s.*, d.path, d.authority, d.priority, d.current_sha256 "
            "FROM sections s JOIN documents d ON d.id=s.document_id WHERE s.id=?",
            (section_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"section not found: {section_id}")
        return self._source_item(row)

    def structural_search(self, terms: tuple[str, ...], limit: int = 10) -> list[SearchHit]:
        """Find direct heading, identifier, section-mark, and path matches."""
        rows_by_section: dict[str, sqlite3.Row] = {}
        matched: dict[str, set[str]] = {}
        for term in terms:
            rows = self.connection.execute(
                "SELECT s.*, d.path, d.authority, d.priority, d.current_sha256, "
                "CASE WHEN s.heading=? COLLATE NOCASE THEN 0 "
                "WHEN s.heading LIKE ? ESCAPE '\\' THEN 1 ELSE 2 END AS structural_rank "
                "FROM sections s JOIN documents d ON d.id=s.document_id "
                "WHERE s.heading=? COLLATE NOCASE OR s.heading LIKE ? ESCAPE '\\' "
                "OR instr(s.text, ?) > 0 OR d.path=? ORDER BY structural_rank, d.path, s.ordinal "
                "LIMIT ?",
                (term, self._like_prefix(term), term, self._like_prefix(term), term, term, 100),
            ).fetchall()
            for row in rows:
                identifier = str(row["id"])
                previous = rows_by_section.get(identifier)
                if previous is None or int(row["structural_rank"]) < int(
                    previous["structural_rank"]
                ):
                    rows_by_section[identifier] = row
                matched.setdefault(identifier, set()).add(term)
        ordered = sorted(
            rows_by_section.values(),
            key=lambda row: (
                int(row["structural_rank"]),
                str(row["path"]),
                int(row["ordinal"]),
                str(row["id"]),
            ),
        )
        return [
            SearchHit(
                source=self._source_item(row),
                score=float(100 - int(row["structural_rank"])),
                channels=("structural",),
                matched_terms=tuple(sorted(matched[str(row["id"])])),
            )
            for row in ordered[:limit]
        ]

    @staticmethod
    def _like_prefix(term: str) -> str:
        escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        return escaped + "%"

    def semantic_search(
        self,
        query_vector: NDArray[np.float32],
        model: str,
        limit: int = 10,
    ) -> list[SearchHit]:
        chunk_ids, matrix = self.load_embeddings(model)
        if not chunk_ids:
            return []
        scores = cosine_scores(query_vector, matrix)
        ranked = sorted(
            range(len(chunk_ids)),
            key=lambda index: (-float(scores[index]), chunk_ids[index]),
        )
        hits: list[SearchHit] = []
        seen: set[str] = set()
        for index in ranked:
            row = self.connection.execute(
                "SELECT s.*, d.path, d.authority, d.priority, d.current_sha256 "
                "FROM chunks c JOIN sections s ON s.id=c.section_id "
                "JOIN documents d ON d.id=s.document_id WHERE c.id=?",
                (chunk_ids[index],),
            ).fetchone()
            if row is None or str(row["id"]) in seen:
                continue
            seen.add(str(row["id"]))
            hits.append(
                SearchHit(
                    source=self._source_item(row),
                    score=float(scores[index]),
                    channels=("semantic",),
                )
            )
            if len(hits) >= limit:
                break
        return hits

    def lexical_search(self, query: str, limit: int = 10) -> list[SearchHit]:
        """Run parameterized FTS5/BM25 with deterministic exact and authority boosts."""
        terms = re.findall(r"[\w][\w.@/-]*", query, flags=re.UNICODE)
        if not terms or limit < 1:
            return []
        expression = " OR ".join(f'"{term.replace(chr(34), chr(34) * 2)}"' for term in terms)
        rows = self.connection.execute(
            "SELECT s.*, d.path, d.authority, d.priority, d.current_sha256, "
            "bm25(chunks_fts, 0.0, 0.0, 8.0, 1.0) AS lexical_rank "
            "FROM chunks_fts JOIN sections s ON s.id=chunks_fts.section_id "
            "JOIN documents d ON d.id=s.document_id WHERE chunks_fts MATCH ? "
            "ORDER BY lexical_rank, d.path, s.ordinal LIMIT ?",
            (expression, min(limit * 8, 400)),
        ).fetchall()
        # Exact case-sensitive identifier fallback handles tokenizer edge cases such as @1.
        exact_rows = self.connection.execute(
            "SELECT s.*, d.path, d.authority, d.priority, d.current_sha256, "
            "0.0 AS lexical_rank FROM sections s JOIN documents d ON d.id=s.document_id "
            "WHERE instr(s.heading, ?) > 0 OR instr(s.text, ?) > 0 "
            "ORDER BY d.path, s.ordinal LIMIT ?",
            (query, query, min(limit * 4, 200)),
        ).fetchall()
        by_section: dict[str, sqlite3.Row] = {}
        for row in [*rows, *exact_rows]:
            by_section.setdefault(str(row["id"]), row)

        query_folded = query.strip().casefold()
        identifier = re.compile(rf"(?<![\w]){re.escape(query.strip())}(?![\w])")
        hits: list[SearchHit] = []
        for row in by_section.values():
            authority = Authority(row["authority"])
            exact_heading = str(row["heading"]).strip().casefold() == query_folded
            exact_identifier = bool(query.strip()) and bool(identifier.search(str(row["text"])))
            authority_score = (
                -1_000_000.0 if authority is Authority.GENERATED else int(authority) * 1_000
            )
            score = (
                -float(row["lexical_rank"])
                + authority_score
                + int(row["priority"]) * 2
                + (10_000 if exact_heading else 0)
                + (5_000 if exact_identifier else 0)
            )
            matched = tuple(
                term for term in terms if term.casefold() in str(row["text"]).casefold()
            )
            hits.append(
                SearchHit(
                    source=self._source_item(row),
                    score=score,
                    channels=("bm25",),
                    matched_terms=matched,
                )
            )
        hits.sort(
            key=lambda hit: (
                -hit.score,
                -int(hit.source.provenance.authority),
                -hit.source.provenance.priority,
                hit.source.provenance.document_path,
                hit.source.provenance.start_line,
                hit.source.provenance.section_id,
            )
        )
        return hits[:limit]

    def graph_sections(self) -> list[GraphSection]:
        rows = self.connection.execute(
            "SELECT id, parent_id, heading, text FROM sections ORDER BY document_id, ordinal"
        ).fetchall()
        return [
            GraphSection(
                id=str(row["id"]),
                parent_id=str(row["parent_id"]) if row["parent_id"] is not None else None,
                heading=str(row["heading"]),
                text=str(row["text"]),
            )
            for row in rows
        ]

    def replace_graph(self, graph: ExtractedGraph) -> None:
        with self.connection:
            self.connection.execute('DELETE FROM "references"')
            self.connection.execute("DELETE FROM symbols")
            self.connection.executemany(
                'INSERT INTO "references"(source_section_id, target_section_id, edge_type, '
                "label, resolved) VALUES(?, ?, ?, ?, ?)",
                (
                    (
                        edge.source_section_id,
                        edge.target_section_id,
                        edge.edge_type.value,
                        edge.label,
                        int(edge.resolved),
                    )
                    for edge in graph.edges
                ),
            )
            self.connection.executemany(
                "INSERT INTO symbols(symbol, section_id, kind) VALUES(?, ?, ?)",
                ((item.symbol, item.section_id, item.kind) for item in graph.symbols),
            )

    def get_references(
        self,
        section_id: str,
        *,
        edge_types: tuple[EdgeType, ...] | None = None,
        incoming: bool = False,
    ) -> list[ReferenceResult]:
        column = "target_section_id" if incoming else "source_section_id"
        parameters: list[object] = [section_id]
        sql = f'SELECT * FROM "references" WHERE {column}=?'  # noqa: S608 - fixed column
        if edge_types:
            sql += " AND edge_type IN (" + ",".join("?" for _ in edge_types) + ")"
            parameters.extend(edge.value for edge in edge_types)
        sql += " ORDER BY edge_type, label, source_section_id, target_section_id"
        rows = self.connection.execute(sql, parameters).fetchall()
        results: list[ReferenceResult] = []
        for row in rows:
            edge = ReferenceRecord(
                source_section_id=row["source_section_id"],
                target_section_id=row["target_section_id"],
                edge_type=EdgeType(row["edge_type"]),
                label=row["label"],
                resolved=bool(row["resolved"]),
            )
            results.append(
                ReferenceResult(
                    edge=edge,
                    source=self.get_section(edge.source_section_id),
                    target=(
                        self.get_section(edge.target_section_id)
                        if edge.target_section_id is not None
                        else None
                    ),
                )
            )
        return results

    def find_symbol(self, symbol: str, limit: int = 20) -> list[SymbolResult]:
        rows = self.connection.execute(
            "SELECT symbol, section_id, kind FROM symbols WHERE symbol=? COLLATE NOCASE "
            "ORDER BY symbol, section_id LIMIT ?",
            (symbol, limit),
        ).fetchall()
        return [
            SymbolResult(
                symbol=str(row["symbol"]),
                kind=str(row["kind"]),
                source=self.get_section(str(row["section_id"])),
            )
            for row in rows
        ]

    def replace_checkpoints(self, checkpoints: tuple[CheckpointMetadata, ...]) -> None:
        with self.connection:
            self.connection.execute("DELETE FROM checkpoint_metadata")
            self.connection.executemany(
                "INSERT INTO checkpoint_metadata(section_id, checkpoint_id, title, fields_json) "
                "VALUES(?, ?, ?, ?)",
                (
                    (
                        checkpoint.root_section_id,
                        checkpoint.checkpoint_id,
                        checkpoint.title,
                        checkpoint.model_dump_json(),
                    )
                    for checkpoint in checkpoints
                ),
            )

    def get_checkpoint_metadata(self, checkpoint_id: str) -> CheckpointMetadata:
        row = self.connection.execute(
            "SELECT fields_json FROM checkpoint_metadata WHERE checkpoint_id=? COLLATE NOCASE",
            (checkpoint_id,),
        ).fetchone()
        if row is None:
            raise KeyError(f"checkpoint not found: {checkpoint_id}")
        return CheckpointMetadata.model_validate_json(row[0])

    def list_checkpoint_metadata(self) -> list[CheckpointMetadata]:
        rows = self.connection.execute(
            "SELECT fields_json FROM checkpoint_metadata ORDER BY checkpoint_id"
        ).fetchall()
        return [CheckpointMetadata.model_validate_json(row[0]) for row in rows]

    def chunks_needing_embeddings(self, model: str) -> list[tuple[str, str, str]]:
        rows = self.connection.execute(
            "SELECT c.id, c.text, c.sha256 FROM chunks c LEFT JOIN embeddings e "
            "ON e.chunk_id=c.id WHERE e.chunk_id IS NULL OR e.model<>? "
            "OR e.chunk_sha256<>c.sha256 ORDER BY c.id",
            (model,),
        ).fetchall()
        return [(str(row[0]), str(row[1]), str(row[2])) for row in rows]

    def save_embeddings(
        self,
        model: str,
        dimensions: int,
        rows: list[tuple[str, str, NDArray[np.float32]]],
    ) -> int:
        with self.connection:
            self.connection.executemany(
                "INSERT INTO embeddings(chunk_id, model, dimensions, vector, chunk_sha256) "
                "VALUES(?, ?, ?, ?, ?) ON CONFLICT(chunk_id) DO UPDATE SET "
                "model=excluded.model, dimensions=excluded.dimensions, vector=excluded.vector, "
                "chunk_sha256=excluded.chunk_sha256",
                (
                    (
                        chunk_id,
                        model,
                        dimensions,
                        np.asarray(vector, dtype="<f4").tobytes(),
                        chunk_hash,
                    )
                    for chunk_id, chunk_hash, vector in rows
                ),
            )
            self.connection.execute(
                "INSERT INTO index_metadata(key, value) VALUES('embedding_model', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (model,),
            )
            self.connection.execute(
                "INSERT INTO index_metadata(key, value) VALUES('embedding_dimensions', ?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (str(dimensions),),
            )
            return self._increment_index_version(self.connection)

    def load_embeddings(self, model: str) -> tuple[list[str], NDArray[np.float32]]:
        rows = self.connection.execute(
            "SELECT chunk_id, dimensions, vector FROM embeddings WHERE model=? ORDER BY chunk_id",
            (model,),
        ).fetchall()
        if not rows:
            return [], np.empty((0, 0), dtype=np.float32)
        dimensions = int(rows[0]["dimensions"])
        vectors = [np.frombuffer(row["vector"], dtype="<f4", count=dimensions) for row in rows]
        return [str(row["chunk_id"]) for row in rows], np.stack(vectors)

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
