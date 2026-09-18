"""Shared context engine and factory used identically by CLI and MCP."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TypeVar

import numpy as np
from numpy.typing import NDArray

from ctx.checkpoints import (
    CHECKPOINT_VERSION,
    CheckpointContext,
    CheckpointResult,
    NamedError,
    SecurityContextItem,
    load_generated_artifacts,
    recognize_checkpoints,
)
from ctx.config import ConfigError, DocumentConfig, database_path, load_config, read_source
from ctx.embeddings import EmbeddingProvider, FastEmbedProvider
from ctx.graph import GRAPH_VERSION, GraphSection, extract_graph
from ctx.heading import HEADING_NORMALIZATION_VERSION
from ctx.models import (
    Authority,
    ContextPack,
    DocumentRecord,
    EdgeType,
    FilterSet,
    IndexStatus,
    OutlineEntry,
    ReferenceResult,
    SearchHit,
    SourceExcerpt,
    SourceItem,
    StatusCategory,
    StatusReason,
    SymbolResult,
    SyncStats,
)
from ctx.parser import (
    CHUNKER_VERSION,
    EMBEDDING_TEXT_VERSION,
    PARSER_VERSION,
    parse_markdown,
    sha256_text,
)
from ctx.retrieval import RETRIEVAL_VERSION, classify_query, fuse_ranked
from ctx.store import SCHEMA_VERSION, SQLiteStore, document_id

T = TypeVar("T")


class StaleIndexError(RuntimeError):
    code = "STALE_INDEX"

    def __init__(self, path: str):
        super().__init__(f"STALE_INDEX: source changed since indexing: {path}")
        self.path = path


class AmbiguousCheckpointError(RuntimeError):
    code = "AMBIGUOUS_CHECKPOINT"

    def __init__(self, checkpoint_id: str, candidates: list[dict[str, object]]):
        self.checkpoint_id = checkpoint_id
        self.candidates = candidates
        super().__init__(
            f"AMBIGUOUS_CHECKPOINT: {checkpoint_id} has equal candidates: "
            + ", ".join(str(item["document_path"]) for item in candidates)
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code,
            "checkpoint_id": self.checkpoint_id,
            "candidates": self.candidates,
        }


class ConcurrentGenerationError(RuntimeError):
    code = "INDEX_GENERATION_CHANGED"


class ContextEngine:
    """Workspace-scoped facade; adapters must construct it via ``create_context_engine``."""

    def __init__(
        self,
        root: Path,
        *,
        embedder: EmbeddingProvider | None = None,
        read_only: bool = False,
    ):
        self.root = root.resolve(strict=True)
        self.config = load_config(self.root)
        self.store = SQLiteStore(database_path(self.root), read_only=read_only)
        self.embedder = embedder
        self._query_cache: OrderedDict[tuple[str, str, int], NDArray[np.float32]] = OrderedDict()
        self._cache_lock = threading.Lock()

    @property
    def active_channels(self) -> tuple[str, ...]:
        return ("structural", "lexical", "semantic") if self.embedder else ("structural", "lexical")

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

    def _behavior_metadata(self, artifact_fingerprint: str) -> dict[str, str]:
        return {
            "schema_version": str(SCHEMA_VERSION),
            "parser_version": PARSER_VERSION,
            "chunker_version": CHUNKER_VERSION,
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
            "heading_normalization_version": HEADING_NORMALIZATION_VERSION,
            "graph_version": GRAPH_VERSION,
            "checkpoint_version": CHECKPOINT_VERSION,
            "retrieval_version": RETRIEVAL_VERSION,
            "checkpoint_artifacts": artifact_fingerprint,
            "embedding_model": self.embedder.identity if self.embedder else "none",
            "embedding_dimensions": str(self.embedder.dimensions if self.embedder else 0),
        }

    def _behavior_fingerprint(self, metadata: dict[str, str], records: list[DocumentRecord]) -> str:
        payload = {
            "algorithms": metadata,
            "documents": [
                {
                    "id": item.id,
                    "path": item.path,
                    "authority": int(item.authority),
                    "priority": item.priority,
                    "sha256": item.sha256,
                }
                for item in records
            ],
        }
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()

    def _plan_documents(
        self, source_hash: dict[str, str]
    ) -> tuple[list[DocumentRecord], int, int, int]:
        existing = self.store.list_documents()
        by_path = {item.path: item for item in existing}
        configured_paths = {item.path for item in self.config.documents}
        missing_old = [item for item in existing if item.path not in configured_paths]
        new_configs = [item for item in self.config.documents if item.path not in by_path]
        old_by_hash: dict[str, list[DocumentRecord]] = {}
        new_by_hash: dict[str, list[DocumentConfig]] = {}
        for old_item in missing_old:
            if old_item.sha256:
                old_by_hash.setdefault(old_item.sha256, []).append(old_item)
        for new_item in new_configs:
            new_by_hash.setdefault(source_hash[new_item.path], []).append(new_item)
        rename_by_path: dict[str, DocumentRecord] = {}
        for digest, new_items in new_by_hash.items():
            old_items = old_by_hash.get(digest, [])
            if len(old_items) == 1 and len(new_items) == 1:
                rename_by_path[new_items[0].path] = old_items[0]

        records: list[DocumentRecord] = []
        renamed = 0
        added = 0
        for config in self.config.documents:
            old = by_path.get(config.path)
            if old is None:
                old = rename_by_path.get(config.path)
                if old:
                    renamed += 1
                else:
                    added += 1
            records.append(
                DocumentRecord(
                    id=old.id if old else document_id(),
                    path=config.path,
                    authority=config.authority,
                    priority=config.priority,
                    sha256=source_hash[config.path],
                    indexed_at=old.indexed_at if old else None,
                )
            )
        retained_old = {item.id for item in records}
        removed = sum(item.id not in retained_old for item in existing)
        return records, added, removed, renamed

    def sync_workspace(self) -> SyncStats:
        """Prepare and commit one generation while multi-query readers wait."""
        with self.store.consistent_read():
            return self._sync_workspace_unlocked()

    def _sync_workspace_unlocked(self) -> SyncStats:
        """Prepare all expensive work, then commit one complete logical generation."""
        self.config = load_config(self.root)
        source_text: dict[str, str] = {}
        source_hash: dict[str, str] = {}
        for document in self.config.documents:
            text = read_source(self.root, document, self.config.limits)
            source_text[document.path] = text
            source_hash[document.path] = sha256_text(text)

        previous_records = self.store.list_documents()
        records, documents_added, documents_removed, documents_renamed = self._plan_documents(
            source_hash
        )
        old_documents = {item.id: item for item in previous_records}
        active_ids = {item.id for item in records}
        artifacts = load_generated_artifacts(
            self.root,
            self.config.limits.max_file_bytes,
            tuple(records),
            deleted_document_ids=set(old_documents) - active_ids,
        )
        artifact_fingerprint = hashlib.sha256(
            "".join(f"{key}:{value.sha256}" for key, value in sorted(artifacts.items())).encode()
        ).hexdigest()
        behavior_metadata = self._behavior_metadata(artifact_fingerprint)
        fingerprint = self._behavior_fingerprint(behavior_metadata, records)
        previous_fingerprint = self.store.get_metadata("behavior_fingerprint")

        source_or_metadata_changed = previous_fingerprint != fingerprint
        missing_vectors = False
        if self.embedder and self.store.section_ids():
            expected = self.store.chunk_count()
            present = self.store.connection.execute(
                "SELECT COUNT(*) FROM embeddings WHERE embedding_identity=?",
                (self.embedder.identity,),
            ).fetchone()[0]
            missing_vectors = int(present) != expected
        if not source_or_metadata_changed and not missing_vectors:
            return SyncStats(
                documents_unchanged=len(records),
                sections_unchanged=len(self.store.section_ids()),
                chunks_unchanged=self.store.chunk_count(),
                embeddings_retained=(self.store.chunk_count() if self.embedder is not None else 0),
                index_generation=self.store.index_generation(),
            )

        old_section_rows = {
            str(row["id"]): (str(row["sha256"]), str(row["document_id"]))
            for row in self.store.connection.execute(
                "SELECT id,sha256,document_id FROM sections"
            ).fetchall()
        }
        old_chunk_rows = {
            str(row["id"]): (str(row["source_sha256"]), str(row["document_id"]))
            for row in self.store.connection.execute(
                "SELECT c.id,c.source_sha256,s.document_id FROM search_chunks c "
                "JOIN sections s ON s.id=c.section_id"
            ).fetchall()
        }
        parsed = {
            record.id: parse_markdown(
                source_text[record.path],
                record.id,
                embedding_token_counter=(self.embedder.count_tokens if self.embedder else None),
            )
            for record in records
        }
        graph_sections = [
            GraphSection(
                id=section.id,
                parent_id=section.parent_id,
                heading=section.heading,
                text=section.text,
                document_id=record.id,
                authority=record.authority,
                priority=record.priority,
                ordinal=section.ordinal,
            )
            for record in records
            for section in parsed[record.id].sections
        ]
        graph = extract_graph(graph_sections)
        checkpoints = recognize_checkpoints(graph_sections, artifacts)

        embedding_rows: list[tuple[str, str, str, object]] = []
        retained = 0
        created = 0
        source_changed_ids = {
            record.id
            for record in records
            if record.id not in old_documents or old_documents[record.id].sha256 != record.sha256
        }
        if self.embedder is not None:
            existing_vectors = self.store.existing_embedding_vectors(self.embedder.identity)
            pending: list[tuple[str, str, str, str]] = []
            for record in records:
                for chunk in parsed[record.id].chunks:
                    key = (chunk.id, chunk.source_sha256, chunk.embedding_sha256)
                    vector = existing_vectors.get(key)
                    if vector is None:
                        pending.append(
                            (
                                chunk.id,
                                chunk.source_sha256,
                                chunk.embedding_sha256,
                                chunk.embedding_text,
                            )
                        )
                    else:
                        embedding_rows.append((*key, vector))
                        if record.id in source_changed_ids:
                            retained += 1
            if pending:
                matrix = self.embedder.embed_documents([item[3] for item in pending])
                if matrix.shape != (len(pending), self.embedder.dimensions):
                    raise RuntimeError("embedding backend returned unexpected dimensions")
                embedding_rows.extend(
                    (item[0], item[1], item[2], matrix[index]) for index, item in enumerate(pending)
                )
                created = len(pending)

        incoming_section_rows = {
            section.id: (section.sha256, record.id)
            for record in records
            for section in parsed[record.id].sections
        }
        incoming_chunk_rows = {
            chunk.id: (chunk.source_sha256, record.id)
            for record in records
            for chunk in parsed[record.id].chunks
        }
        changed_documents = sum(
            item.id in old_documents
            and (
                old_documents[item.id].sha256 != item.sha256
                or old_documents[item.id].authority != item.authority
                or old_documents[item.id].priority != item.priority
            )
            for item in records
        )
        documents_unchanged = len(records) - documents_added - changed_documents
        incoming_document_ids = {item.id for item in records}
        affected_document_ids = {
            item.id
            for item in records
            if item.id not in old_documents or old_documents[item.id].sha256 != item.sha256
        } | {old_id for old_id in old_documents if old_id not in incoming_document_ids}
        old_section_hashes = {
            key: digest
            for key, (digest, owner) in old_section_rows.items()
            if owner in affected_document_ids
        }
        incoming_sections = {
            key: digest
            for key, (digest, owner) in incoming_section_rows.items()
            if owner in affected_document_ids
        }
        old_chunk_hashes = {
            key: digest
            for key, (digest, owner) in old_chunk_rows.items()
            if owner in affected_document_ids
        }
        incoming_chunks = {
            key: digest
            for key, (digest, owner) in incoming_chunk_rows.items()
            if owner in affected_document_ids
        }
        behavior_metadata["behavior_fingerprint"] = fingerprint
        generation = self.store.apply_workspace(
            records,
            parsed,
            graph,
            checkpoints,
            embedding_rows,  # type: ignore[arg-type]
            embedding_identity=self.embedder.identity if self.embedder else "none",
            embedding_dimensions=self.embedder.dimensions if self.embedder else 0,
            embedding_metadata=self.embedder.metadata if self.embedder else {},
            behavior_metadata=behavior_metadata,
            reason="ATOMIC_WORKSPACE_SYNC",
            behavior_fingerprint=fingerprint,
        )
        with self._cache_lock:
            self._query_cache.clear()
        return SyncStats(
            documents_added=documents_added,
            documents_changed=changed_documents,
            documents_unchanged=max(0, documents_unchanged),
            documents_removed=documents_removed,
            documents_renamed=documents_renamed,
            sections_added=len(set(incoming_sections) - set(old_section_hashes)),
            sections_changed=sum(
                incoming_sections[key] != old_section_hashes[key]
                for key in set(incoming_sections) & set(old_section_hashes)
            ),
            sections_unchanged=sum(
                incoming_sections[key] == old_section_hashes[key]
                for key in set(incoming_sections) & set(old_section_hashes)
            ),
            sections_removed=len(set(old_section_hashes) - set(incoming_sections)),
            chunks_added=len(set(incoming_chunks) - set(old_chunk_hashes)),
            chunks_changed=sum(
                incoming_chunks[key] != old_chunk_hashes[key]
                for key in set(incoming_chunks) & set(old_chunk_hashes)
            ),
            chunks_unchanged=sum(
                incoming_chunks[key] == old_chunk_hashes[key]
                for key in set(incoming_chunks) & set(old_chunk_hashes)
            ),
            chunks_removed=len(set(old_chunk_hashes) - set(incoming_chunks)),
            embeddings_retained=retained,
            embeddings_created=created,
            index_generation=generation,
        )

    def index_workspace(self) -> SyncStats:
        return self.sync_workspace()

    def status(self) -> IndexStatus:
        self.config = load_config(self.root)
        indexed = self.store.list_documents()
        indexed_by_path = {item.path: item for item in indexed}
        database_schema = int(self.store.connection.execute("PRAGMA user_version").fetchone()[0])
        reasons: list[StatusReason] = []
        if database_schema != SCHEMA_VERSION:
            reasons.append(
                StatusReason(
                    category=StatusCategory.SCHEMA_STALE,
                    reason=(
                        f"database schema {database_schema}; runtime requires {SCHEMA_VERSION}"
                    ),
                )
            )
        stale: list[str] = []
        missing: list[str] = []
        for document in self.config.documents:
            try:
                text = read_source(self.root, document, self.config.limits)
            except ConfigError as error:
                missing.append(document.path)
                reasons.append(
                    StatusReason(
                        category=StatusCategory.MISSING_SOURCE,
                        path=document.path,
                        reason=str(error),
                    )
                )
                continue
            record = indexed_by_path.get(document.path)
            if record is None:
                missing.append(document.path)
                reasons.append(
                    StatusReason(
                        category=StatusCategory.METADATA_STALE,
                        path=document.path,
                        reason="configured document is not indexed",
                    )
                )
            elif record.sha256 != sha256_text(text):
                stale.append(document.path)
                reasons.append(
                    StatusReason(
                        category=StatusCategory.SOURCE_STALE,
                        path=document.path,
                        reason="source SHA-256 differs from committed generation",
                    )
                )
            elif record.authority != document.authority or record.priority != document.priority:
                reasons.append(
                    StatusReason(
                        category=StatusCategory.METADATA_STALE,
                        path=document.path,
                        reason="authority or priority differs from committed generation",
                    )
                )
        algorithm_checks = (
            ("parser_version", PARSER_VERSION, StatusCategory.PARSER_STALE),
            ("chunker_version", CHUNKER_VERSION, StatusCategory.PARSER_STALE),
            (
                "heading_normalization_version",
                HEADING_NORMALIZATION_VERSION,
                StatusCategory.PARSER_STALE,
            ),
            ("graph_version", GRAPH_VERSION, StatusCategory.GRAPH_STALE),
            ("checkpoint_version", CHECKPOINT_VERSION, StatusCategory.GRAPH_STALE),
            ("retrieval_version", RETRIEVAL_VERSION, StatusCategory.SCHEMA_STALE),
            ("embedding_text_version", EMBEDDING_TEXT_VERSION, StatusCategory.EMBEDDINGS_STALE),
        )
        for key, expected, category in algorithm_checks:
            actual = self.store.get_metadata(key)
            if indexed and actual != expected:
                reasons.append(
                    StatusReason(
                        category=category, reason=f"{key}: indexed={actual!r}, runtime={expected!r}"
                    )
                )
        if self.embedder and self.store.chunk_count():
            vector_count = int(
                self.store.connection.execute(
                    "SELECT COUNT(*) FROM embeddings WHERE embedding_identity=?",
                    (self.embedder.identity,),
                ).fetchone()[0]
            )
            if vector_count != self.store.chunk_count():
                reasons.append(
                    StatusReason(
                        category=StatusCategory.EMBEDDINGS_STALE,
                        reason=f"missing vectors: {vector_count}/{self.store.chunk_count()}",
                    )
                )
        category = reasons[0].category if reasons else StatusCategory.CLEAN
        return IndexStatus(
            index_generation=self.store.index_generation(),
            configured_documents=len(self.config.documents),
            indexed_documents=len(indexed),
            category=category,
            reasons=tuple(reasons),
            stale_documents=tuple(stale),
            missing_documents=tuple(missing),
            schema_version_db=database_schema,
            parser_version=PARSER_VERSION,
            chunker_version=CHUNKER_VERSION,
            graph_version=GRAPH_VERSION,
            checkpoint_version=CHECKPOINT_VERSION,
            retrieval_version=RETRIEVAL_VERSION,
            embedding_identity=self.store.get_metadata("embedding_identity") or "none",
            active_channels=self.active_channels,
        )

    def _validate_hits(self, hits: list[SearchHit]) -> list[SearchHit]:
        if not hits:
            return hits
        generation = hits[0].index_generation
        if any(hit.index_generation != generation for hit in hits):
            raise ConcurrentGenerationError("mixed index generations in retrieval result")
        if self.store.index_generation() != generation:
            raise ConcurrentGenerationError("index generation changed during retrieval")
        checked: set[str] = set()
        for hit in hits:
            path = hit.source.provenance.document_path
            if path not in checked:
                document = self._configured(path)
                current = read_source(self.root, document, self.config.limits)
                if sha256_text(current) != hit.source.provenance.document_sha256:
                    raise StaleIndexError(path)
                checked.add(path)
        return hits

    def document_outline(self, path: str) -> list[OutlineEntry]:
        entries = self.store.document_outline(path)
        document = self._configured(path)
        current = read_source(self.root, document, self.config.limits)
        record = self.store.get_document_by_path(path)
        if record.sha256 != sha256_text(current):
            raise StaleIndexError(path)
        return entries

    def get_lines(self, path: str, start_line: int, end_line: int) -> list[SourceItem]:
        if start_line < 1 or end_line < start_line:
            raise ValueError("invalid one-based line range")
        if end_line - start_line + 1 > self.config.limits.max_line_results:
            raise ValueError(
                "line range exceeds configured maximum of "
                f"{self.config.limits.max_line_results} lines"
            )
        document = self._configured(path)
        current = read_source(self.root, document, self.config.limits)
        record = self.store.get_document_by_path(path)
        if record.sha256 != sha256_text(current):
            raise StaleIndexError(path)
        lines = current.splitlines(keepends=True)
        if end_line > len(lines):
            raise ValueError("line range exceeds document length")
        line_offsets = [0]
        for line in lines:
            line_offsets.append(line_offsets[-1] + len(line))
        results: list[SourceItem] = []
        for item in self.store.document_sections(path):
            start = max(start_line, item.provenance.start_line)
            end = min(end_line, item.provenance.end_line)
            if start > end:
                continue
            text = "".join(lines[start - 1 : end])
            start_offset = line_offsets[start - 1]
            end_offset = line_offsets[end]
            provenance = item.provenance.model_copy(
                update={
                    "start_line": start,
                    "end_line": end,
                    "start_column": 0,
                    "end_column": len(lines[end - 1]),
                    "start_offset": start_offset,
                    "end_offset": end_offset,
                    "range_sha256": sha256_text(text),
                }
            )
            results.append(
                SourceExcerpt(text=text, provenance=provenance, reason="exact line range")
            )
        return results

    @staticmethod
    def _filters(
        *,
        documents: set[str] | None = None,
        authority_floor: Authority | None = None,
        authorities: set[Authority] | None = None,
        exclude_documents: set[str] | None = None,
        heading_prefix: tuple[str, ...] | None = None,
        scope: str | None = None,
    ) -> FilterSet:
        return FilterSet(
            documents=frozenset(documents) if documents else None,
            authority_floor=authority_floor,
            authorities=frozenset(authorities) if authorities else None,
            exclude_documents=frozenset(exclude_documents or ()),
            heading_prefix=heading_prefix,
            scope=scope,
        )

    def search_exact(
        self,
        query: str,
        *,
        limit: int = 10,
        include_full_section: bool = False,
        **filter_args: object,
    ) -> list[SearchHit]:
        with self.store.consistent_read():
            return self._search_exact_unlocked(
                query,
                limit=limit,
                include_full_section=include_full_section,
                **filter_args,
            )

    def _search_exact_unlocked(
        self,
        query: str,
        *,
        limit: int = 10,
        include_full_section: bool = False,
        **filter_args: object,
    ) -> list[SearchHit]:
        self._validate_query(query, exact=True)
        bounded = min(max(limit, 1), self.config.limits.max_results)
        filters = self._filters(**filter_args)  # type: ignore[arg-type]
        for _ in range(2):
            hits = self.store.structural_search(
                (query.strip(),),
                bounded,
                filters=filters,
                include_full_section=include_full_section,
            )
            try:
                return self._validate_hits(hits)
            except ConcurrentGenerationError:
                continue
        raise ConcurrentGenerationError("index changed repeatedly during exact search")

    def _validate_query(self, query: str, *, exact: bool = False) -> None:
        if not query.strip():
            if exact:
                raise ValueError("exact query is empty")
            return
        if len(query) > self.config.limits.max_query_chars:
            raise ValueError("query exceeds configured max_query_chars")

    def search_lexical(
        self,
        query: str,
        *,
        limit: int = 10,
        auto_sync: bool = False,
        include_full_section: bool = False,
        **filter_args: object,
    ) -> list[SearchHit]:
        with self.store.consistent_read():
            return self._search_lexical_unlocked(
                query,
                limit=limit,
                auto_sync=auto_sync,
                include_full_section=include_full_section,
                **filter_args,
            )

    def _search_lexical_unlocked(
        self,
        query: str,
        *,
        limit: int = 10,
        auto_sync: bool = False,
        include_full_section: bool = False,
        **filter_args: object,
    ) -> list[SearchHit]:
        self._validate_query(query)
        if not query.strip():
            return []
        if auto_sync:
            self.sync_workspace()
        bounded = min(max(limit, 1), self.config.limits.max_results)
        filters = self._filters(**filter_args)  # type: ignore[arg-type]
        for _ in range(2):
            hits = self.store.lexical_search(
                query, bounded, filters=filters, include_full_section=include_full_section
            )
            try:
                return self._validate_hits(hits)
            except ConcurrentGenerationError:
                continue
        raise ConcurrentGenerationError("index changed repeatedly during lexical search")

    def _query_embedding(self, query: str) -> NDArray[np.float32]:
        if self.embedder is None:
            raise RuntimeError("semantic channel is disabled")
        key = (self.embedder.identity, query, self.store.index_generation())
        with self._cache_lock:
            cached = self._query_cache.get(key)
            if cached is not None:
                self._query_cache.move_to_end(key)
                return cached
        value = self.embedder.embed_query(query)
        value.setflags(write=False)
        with self._cache_lock:
            self._query_cache[key] = value
            self._query_cache.move_to_end(key)
            while len(self._query_cache) > 128:
                self._query_cache.popitem(last=False)
        return value

    def search_semantic(
        self,
        query: str,
        *,
        limit: int = 10,
        include_full_section: bool = False,
        **filter_args: object,
    ) -> list[SearchHit]:
        """Run only the installed verified-local semantic channel."""
        self._validate_query(query)
        if not query.strip():
            return []
        if self.embedder is None:
            return []
        bounded = min(max(limit, 1), self.config.limits.max_results)
        filters = self._filters(**filter_args)  # type: ignore[arg-type]
        with self.store.consistent_read():
            hits = self.store.semantic_search(
                self._query_embedding(query),
                self.embedder.identity,
                bounded,
                filters=filters,
                include_full_section=include_full_section,
            )
            return self._validate_hits(hits)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        auto_sync: bool = False,
        include_full_section: bool = False,
        **filter_args: object,
    ) -> list[SearchHit]:
        with self.store.consistent_read():
            return self._search_unlocked(
                query,
                limit=limit,
                auto_sync=auto_sync,
                include_full_section=include_full_section,
                **filter_args,
            )

    def _search_unlocked(
        self,
        query: str,
        *,
        limit: int = 10,
        auto_sync: bool = False,
        include_full_section: bool = False,
        **filter_args: object,
    ) -> list[SearchHit]:
        self._validate_query(query)
        if not query.strip():
            return []
        if auto_sync:
            self.sync_workspace()
        bounded = min(max(limit, 1), self.config.limits.max_results)
        candidates = min(max(bounded * 4, 20), self.config.limits.max_results)
        filters = self._filters(**filter_args)  # type: ignore[arg-type]
        classification = classify_query(query)
        structural_terms = tuple(dict.fromkeys((query.strip(), *classification.structural_terms)))
        for _ in range(2):
            start_generation = self.store.index_generation()
            structural = self.store.structural_search(
                structural_terms,
                candidates,
                filters=filters,
                include_full_section=include_full_section,
            )
            lexical = self.store.lexical_search(
                query, candidates, filters=filters, include_full_section=include_full_section
            )
            channels: list[tuple[str, list[SearchHit]]] = [
                ("structural", structural),
                ("bm25", lexical),
            ]
            if self.embedder is not None:
                semantic = self.store.semantic_search(
                    self._query_embedding(query),
                    self.embedder.identity,
                    candidates,
                    filters=filters,
                    include_full_section=include_full_section,
                )
                channels.append(("semantic", semantic))
            if self.store.index_generation() != start_generation:
                continue
            fused = fuse_ranked(query, classification, channels, bounded)
            try:
                return self._validate_hits(fused)
            except ConcurrentGenerationError:
                continue
        raise ConcurrentGenerationError("index changed repeatedly during hybrid search")

    def get_checkpoint(
        self, checkpoint_id: str, *, document: str | None = None
    ) -> CheckpointResult:
        normalized = checkpoint_id.strip().upper()
        if not re.fullmatch(r"CP-\d+", normalized):
            raise ValueError("checkpoint_id must have form CP-N")
        candidates = self.store.checkpoint_candidates(normalized, document)
        if not candidates:
            raise KeyError(f"checkpoint not found: {normalized}")
        top_authority = candidates[0][1].authority
        top_priority = candidates[0][1].priority
        tied = [
            pair
            for pair in candidates
            if pair[1].authority == top_authority and pair[1].priority == top_priority
        ]
        if len(tied) > 1:
            raise AmbiguousCheckpointError(
                normalized,
                [
                    {
                        "document_id": record.id,
                        "document_path": record.path,
                        "authority": record.authority.name,
                        "priority": record.priority,
                        "root_section_id": metadata.root_section_id,
                    }
                    for metadata, record in tied
                ],
            )
        metadata = candidates[0][0]
        sources = tuple(self.get_section(section_id) for section_id in metadata.section_ids)
        return CheckpointResult(metadata=metadata, sources=sources)

    def get_checkpoint_context(
        self,
        checkpoint_id: str,
        *,
        document: str | None = None,
        token_budget: int = 15_000,
        allow_required_budget_expansion: bool = False,
    ) -> CheckpointContext:
        checkpoint = self.get_checkpoint(checkpoint_id, document=document)
        dependencies: list[ReferenceResult] = []
        references: list[ReferenceResult] = []
        interfaces: list[SourceItem] = []
        security: dict[str, SecurityContextItem] = {}
        for section_id in checkpoint.metadata.section_ids:
            for result in self.store.get_references(section_id):
                if result.edge.edge_type is EdgeType.DEPENDS_ON:
                    dependencies.append(result)
                    if result.target is not None and re.search(
                        r"(?i)\b(?:interface|model|type|class)\b", result.target.text
                    ):
                        interfaces.append(result.target)
                elif result.edge.edge_type in {EdgeType.REFERENCES, EdgeType.RELATED_SECTION}:
                    references.append(result)
                elif result.edge.edge_type is EdgeType.USES_TYPE and result.target is not None:
                    interfaces.append(result.target)
                if result.target and "security" in result.target.text.casefold():
                    security[result.target.provenance.section_id] = SecurityContextItem(
                        source=result.target,
                        applicability="directly_applicable",
                        reason=(
                            f"checkpoint graph edge {result.edge.edge_type.value}: "
                            f"{result.edge.label}"
                        ),
                    )
        exact_text = "\n".join(source.text for source in checkpoint.sources)
        error_matches = set(
            re.findall(
                r"\b(?:[A-Z][A-Za-z0-9]*(?:Error|Failure|Timeout|Exception)|"
                r"[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+)\b",
                exact_text,
            )
        )
        error_evidence = tuple(
            NamedError(symbol=value, confidence=1.0, source="checkpoint exact source")
            for value in sorted(error_matches)
        )
        # Normative global security is evidence-backed and precedes optional semantic fallback.
        for hit in self.search(
            "security",
            limit=10,
            documents={checkpoint.metadata.document_id},
            authority_floor=Authority.REFERENCE,
        ):
            if "security" in hit.source.text.casefold():
                security.setdefault(
                    hit.source.provenance.section_id,
                    SecurityContextItem(
                        source=hit.source,
                        applicability="global",
                        reason="same-document normative security source",
                    ),
                )
        if not security:
            for hit in self.search("security", limit=3, authority_floor=Authority.NORMATIVE):
                security[hit.source.provenance.section_id] = SecurityContextItem(
                    source=hit.source,
                    applicability="semantic_candidate",
                    reason="semantic fallback; inspect applicability",
                )
        fields = checkpoint.metadata.fields
        pack = self.get_context_pack(
            f"Implement {checkpoint.metadata.checkpoint_id} — {checkpoint.metadata.title}",
            token_budget,
            allow_required_budget_expansion=allow_required_budget_expansion,
            _checkpoint_document=checkpoint.metadata.document_id,
        )
        unique_interfaces = {source.provenance.section_id: source for source in interfaces}
        security_context = tuple(security.values())
        return CheckpointContext(
            checkpoint=checkpoint,
            dependencies=tuple(dependencies),
            references=tuple(references),
            interfaces_models=tuple(unique_interfaces.values()),
            named_errors=tuple(sorted(error_matches)),
            error_evidence=error_evidence,
            security_rules=tuple(item.source for item in security_context),
            security_context=security_context,
            acceptance_criteria=fields.get("tests_acceptance_criteria", ()),
            verification_commands=fields.get("verify", ()),
            out_of_scope=fields.get("out_of_scope", ()),
            context_pack=pack,
        )

    def get_context_pack(
        self,
        task: str,
        token_budget: int,
        *,
        allow_required_budget_expansion: bool = False,
        _checkpoint_document: str | None = None,
        documents: set[str] | None = None,
        authority_floor: Authority | None = None,
        authorities: set[Authority] | None = None,
        exclude_documents: set[str] | None = None,
        heading_prefix: tuple[str, ...] | None = None,
        scope: str | None = None,
    ) -> ContextPack:
        from ctx.context_pack import build_context_pack

        if len(task) > self.config.limits.max_query_chars:
            raise ValueError("task exceeds configured max_query_chars")
        if token_budget > self.config.limits.max_token_budget:
            raise ValueError("token_budget exceeds configured max_token_budget")
        with self.store.consistent_read():
            return build_context_pack(
                self,
                task,
                token_budget,
                filters=self._filters(
                    documents=documents,
                    authority_floor=authority_floor,
                    authorities=authorities,
                    exclude_documents=exclude_documents,
                    heading_prefix=heading_prefix,
                    scope=scope,
                ),
                allow_required_budget_expansion=allow_required_budget_expansion,
                checkpoint_document=_checkpoint_document,
            )

    def get_references(self, section_id: str, *, incoming: bool = False) -> list[ReferenceResult]:
        results = self.store.get_references(section_id, incoming=incoming)
        hits = [
            SearchHit(
                source=result.source,
                score=0,
                channels=("graph",),
                index_generation=result.source.provenance.index_generation,
            )
            for result in results
        ]
        hits.extend(
            SearchHit(
                source=result.target,
                score=0,
                channels=("graph",),
                index_generation=result.target.provenance.index_generation,
            )
            for result in results
            if result.target is not None
        )
        self._validate_hits(hits)
        return results

    def get_dependencies(self, section_id: str) -> list[ReferenceResult]:
        results = self.store.get_references(section_id, edge_types=(EdgeType.DEPENDS_ON,))
        self.get_references(section_id)
        return results

    def find_symbol(
        self,
        symbol: str,
        *,
        limit: int = 20,
        documents: set[str] | None = None,
        authority_floor: Authority | None = None,
        authorities: set[Authority] | None = None,
        exclude_documents: set[str] | None = None,
        heading_prefix: tuple[str, ...] | None = None,
        scope: str | None = None,
    ) -> list[SymbolResult]:
        if len(symbol) > self.config.limits.max_query_chars:
            raise ValueError("symbol exceeds configured max_query_chars")
        results = self.store.find_symbol(
            symbol,
            min(limit, self.config.limits.max_results),
            filters=self._filters(
                documents=documents,
                authority_floor=authority_floor,
                authorities=authorities,
                exclude_documents=exclude_documents,
                heading_prefix=heading_prefix,
                scope=scope,
            ),
        )
        self._validate_hits(
            [
                SearchHit(
                    source=result.source,
                    score=0,
                    channels=("symbol",),
                    index_generation=result.source.provenance.index_generation,
                )
                for result in results
            ]
        )
        return results

    def get_section(self, section_id: str, *, auto_sync: bool = False) -> SourceItem:
        item = self.store.get_section(section_id)
        document = self._configured(item.provenance.document_path)
        current = read_source(self.root, document, self.config.limits)
        if sha256_text(current) != item.provenance.document_sha256:
            if auto_sync:
                self.sync_workspace()
                return self.get_section(section_id, auto_sync=False)
            raise StaleIndexError(document.path)
        if item.provenance.index_generation != self.store.index_generation():
            return self.get_section(section_id, auto_sync=auto_sync)
        return item


def create_context_engine(
    root: Path,
    *,
    embedder: EmbeddingProvider | None = None,
    embeddings_enabled: bool = True,
    model_dir: Path | None = None,
    read_only: bool = False,
) -> ContextEngine:
    """Single engine factory for CLI/MCP with explicit verified-local fallback semantics."""
    resolved = root.resolve(strict=True)
    if embedder is not None:
        return ContextEngine(resolved, embedder=embedder, read_only=read_only)
    config = load_config(resolved)
    if not embeddings_enabled or config.embedding.backend == "disabled":
        return ContextEngine(resolved, embedder=None, read_only=read_only)
    configured_dir = Path(config.embedding.model_dir) if config.embedding.model_dir else model_dir
    try:
        provider = FastEmbedProvider(
            config.embedding.model,
            revision=config.embedding.revision,
            cache_dir=configured_dir,
        )
    except RuntimeError:
        provider = None
    return ContextEngine(resolved, embedder=provider, read_only=read_only)
