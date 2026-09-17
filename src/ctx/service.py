"""Shared application service used by every CLI and MCP adapter."""

from __future__ import annotations

from pathlib import Path

from ctx.config import (
    ConfigError,
    DocumentConfig,
    database_path,
    load_config,
    read_source,
)
from ctx.models import IndexStatus, SourceItem, SyncStats
from ctx.parser import PARSER_VERSION, parse_markdown, sha256_text
from ctx.store import SQLiteStore


class StaleIndexError(RuntimeError):
    """Raised rather than returning source whose indexed hash is no longer current."""

    code = "STALE_INDEX"

    def __init__(self, path: str):
        super().__init__(f"STALE_INDEX: source changed since indexing: {path}")
        self.path = path


class ContextEngine:
    """Workspace-scoped indexing and retrieval facade."""

    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        self.config = load_config(self.root)
        self.store = SQLiteStore(database_path(self.root))

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> ContextEngine:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _configured(self, path: str) -> DocumentConfig:
        for document in self.config.documents:
            if document.path == path:
                return document
        raise StaleIndexError(path)

    def sync_workspace(self) -> SyncStats:
        """Synchronize configured sources, performing zero parse/index work when unchanged."""
        self.config = load_config(self.root)
        source_text: dict[str, str] = {}
        source_hash: dict[str, str] = {}
        for document in self.config.documents:
            text = read_source(self.root, document, self.config.limits)
            source_text[document.path] = text
            source_hash[document.path] = sha256_text(text)

        existing = {document.path: document for document in self.store.list_documents()}
        configured_paths = {document.path for document in self.config.documents}
        missing_old = [item for path, item in existing.items() if path not in configured_paths]
        renamed = 0
        # A rename is accepted only for a unique same-hash absent source, avoiding guesses.
        for document in self.config.documents:
            if document.path in existing:
                continue
            candidates = [item for item in missing_old if item.sha256 == source_hash[document.path]]
            if len(candidates) == 1:
                previous = candidates[0]
                record = self.store.rename_document(previous.path, document.path)
                existing.pop(previous.path)
                existing[document.path] = record
                missing_old.remove(previous)
                renamed += 1

        totals = {
            "documents_added": 0,
            "documents_changed": 0,
            "documents_unchanged": 0,
            "documents_removed": 0,
            "documents_renamed": renamed,
            "sections_added": 0,
            "sections_changed": 0,
            "sections_unchanged": 0,
            "sections_removed": 0,
            "chunks_added": 0,
            "chunks_changed": 0,
            "chunks_unchanged": 0,
            "chunks_removed": 0,
            "embeddings_retained": 0,
        }
        for document in self.config.documents:
            was_known = document.path in existing
            record = self.store.register_document(
                document.path, document.authority, document.priority
            )
            parser_current = self.store.document_parser_version(record.id) == PARSER_VERSION
            if record.sha256 == source_hash[document.path] and parser_current:
                totals["documents_unchanged"] += 1
                continue
            parsed = parse_markdown(source_text[document.path], record.id)
            stats = self.store.sync_document(record, parsed)
            totals["documents_changed" if was_known else "documents_added"] += 1
            for field in (
                "sections_added",
                "sections_changed",
                "sections_unchanged",
                "sections_removed",
                "chunks_added",
                "chunks_changed",
                "chunks_unchanged",
                "chunks_removed",
                "embeddings_retained",
            ):
                totals[field] += getattr(stats, field)

        for record in self.store.list_documents():
            if record.path not in configured_paths:
                self.store.remove_document(record.path)
                totals["documents_removed"] += 1

        return SyncStats(**totals, index_version=self.store.index_version())

    def index_workspace(self) -> SyncStats:
        """Index configured documents; equivalent to safe incremental sync."""
        return self.sync_workspace()

    def status(self) -> IndexStatus:
        self.config = load_config(self.root)
        indexed = self.store.list_documents()
        indexed_by_path = {item.path: item for item in indexed}
        stale: list[str] = []
        missing: list[str] = []
        for document in self.config.documents:
            try:
                text = read_source(self.root, document, self.config.limits)
            except ConfigError:
                missing.append(document.path)
                continue
            record = indexed_by_path.get(document.path)
            if record is None:
                missing.append(document.path)
            elif record.sha256 != sha256_text(text):
                stale.append(document.path)
        return IndexStatus(
            index_version=self.store.index_version(),
            configured_documents=len(self.config.documents),
            indexed_documents=len(indexed),
            stale_documents=tuple(stale),
            missing_documents=tuple(missing),
            parser_version=PARSER_VERSION,
            embedding_model=self.store.get_metadata("embedding_model") or "none",
        )

    def get_section(self, section_id: str, *, auto_sync: bool = False) -> SourceItem:
        item = self.store.get_section(section_id)
        document = self._configured(item.provenance.document_path)
        current = read_source(self.root, document, self.config.limits)
        if sha256_text(current) != item.provenance.document_sha256:
            if auto_sync:
                self.sync_workspace()
                return self.get_section(section_id, auto_sync=False)
            raise StaleIndexError(document.path)
        return item
