"""Shared application service used by every CLI and MCP adapter."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ctx.checkpoints import (
    CheckpointContext,
    CheckpointResult,
    load_generated_artifacts,
    recognize_checkpoints,
)
from ctx.config import (
    ConfigError,
    DocumentConfig,
    database_path,
    load_config,
    read_source,
)
from ctx.embeddings import EmbeddingProvider
from ctx.graph import extract_graph
from ctx.models import (
    Authority,
    ContextPack,
    EdgeType,
    IndexStatus,
    ReferenceResult,
    SearchHit,
    SourceItem,
    SymbolResult,
    SyncStats,
)
from ctx.parser import PARSER_VERSION, parse_markdown, sha256_text
from ctx.retrieval import classify_query, fuse_ranked
from ctx.store import SQLiteStore


class StaleIndexError(RuntimeError):
    """Raised rather than returning source whose indexed hash is no longer current."""

    code = "STALE_INDEX"

    def __init__(self, path: str):
        super().__init__(f"STALE_INDEX: source changed since indexing: {path}")
        self.path = path


class ContextEngine:
    """Workspace-scoped indexing and retrieval facade."""

    def __init__(self, root: Path, *, embedder: EmbeddingProvider | None = None):
        self.root = root.resolve(strict=True)
        self.config = load_config(self.root)
        self.store = SQLiteStore(database_path(self.root))
        self.embedder = embedder

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
            "embeddings_created": 0,
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

        graph_changed = any(
            totals[field]
            for field in (
                "documents_added",
                "documents_changed",
                "documents_removed",
            )
        )
        if graph_changed:
            self.store.replace_graph(extract_graph(self.store.graph_sections()))

        artifacts = load_generated_artifacts(self.root, self.config.limits.max_file_bytes)
        artifact_fingerprint = hashlib.sha256(
            "".join(
                f"{key}:{artifact.sha256}" for key, artifact in sorted(artifacts.items())
            ).encode()
        ).hexdigest()
        artifacts_changed = self.store.get_metadata("checkpoint_artifacts") != artifact_fingerprint
        if graph_changed or artifacts_changed:
            checkpoints = recognize_checkpoints(self.store.graph_sections(), artifacts)
            self.store.replace_checkpoints(checkpoints)
            self.store.set_metadata("checkpoint_artifacts", artifact_fingerprint)
            if artifacts_changed and not graph_changed:
                self.store.touch_index()

        if self.embedder is not None:
            pending = self.store.chunks_needing_embeddings(self.embedder.identity)
            if pending:
                matrix = self.embedder.embed_documents([row[1] for row in pending])
                if matrix.shape != (len(pending), self.embedder.dimensions):
                    raise RuntimeError("embedding backend returned unexpected dimensions")
                rows = [
                    (chunk_id, chunk_hash, matrix[index])
                    for index, (chunk_id, _text, chunk_hash) in enumerate(pending)
                ]
                self.store.save_embeddings(self.embedder.identity, self.embedder.dimensions, rows)
                totals["embeddings_created"] = len(rows)

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

    def _validate_hits(self, hits: list[SearchHit]) -> list[SearchHit]:
        # Validate all documents before returning any possibly stale source text.
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

    def document_outline(self, path: str) -> list[SourceItem]:
        items = self.store.document_sections(path)
        self._validate_hits(
            [SearchHit(source=item, score=0, channels=("outline",)) for item in items]
        )
        return items

    def get_lines(self, path: str, start_line: int, end_line: int) -> list[SourceItem]:
        if start_line < 1 or end_line < start_line:
            raise ValueError("invalid one-based line range")
        if end_line - start_line + 1 > 2_000:
            raise ValueError("line range exceeds safe maximum of 2000 lines")
        document = self._configured(path)
        current = read_source(self.root, document, self.config.limits)
        record = self.store.get_document_by_path(path)
        if record.sha256 != sha256_text(current):
            raise StaleIndexError(path)
        lines = current.splitlines(keepends=True)
        if end_line > len(lines):
            raise ValueError("line range exceeds document length")
        results: list[SourceItem] = []
        for item in self.store.document_sections(path):
            start = max(start_line, item.provenance.start_line)
            end = min(end_line, item.provenance.end_line)
            if start > end:
                continue
            text = "".join(lines[start - 1 : end])
            provenance = item.provenance.model_copy(update={"start_line": start, "end_line": end})
            results.append(SourceItem(text=text, provenance=provenance, reason="exact line range"))
        return results

    def search_exact(self, query: str, *, limit: int = 10) -> list[SearchHit]:
        if not query.strip() or len(query) > self.config.limits.max_query_chars:
            raise ValueError("exact query is empty or exceeds max_query_chars")
        bounded = min(max(limit, 1), self.config.limits.max_results)
        return self._validate_hits(self.store.structural_search((query.strip(),), bounded))

    def search_lexical(
        self, query: str, *, limit: int = 10, auto_sync: bool = False
    ) -> list[SearchHit]:
        if not query.strip():
            return []
        if len(query) > self.config.limits.max_query_chars:
            raise ValueError("query exceeds configured max_query_chars")
        bounded = min(max(limit, 1), self.config.limits.max_results)
        if auto_sync:
            self.sync_workspace()
        return self._validate_hits(self.store.lexical_search(query, bounded))

    def search(self, query: str, *, limit: int = 10, auto_sync: bool = False) -> list[SearchHit]:
        """Run structural, BM25, and available local semantic retrieval, then fuse."""
        if not query.strip():
            return []
        if len(query) > self.config.limits.max_query_chars:
            raise ValueError("query exceeds configured max_query_chars")
        if auto_sync:
            self.sync_workspace()
        bounded = min(max(limit, 1), self.config.limits.max_results)
        candidates = min(max(bounded * 4, 20), self.config.limits.max_results)
        classification = classify_query(query)
        structural_terms = tuple(dict.fromkeys((query.strip(), *classification.structural_terms)))
        structural = self.store.structural_search(structural_terms, candidates)
        lexical = self.store.lexical_search(query, candidates)
        channels: list[tuple[str, list[SearchHit]]] = [
            ("structural", structural),
            ("bm25", lexical),
        ]
        if self.embedder is not None:
            semantic = self.store.semantic_search(
                self.embedder.embed_query(query), self.embedder.identity, candidates
            )
            channels.append(("semantic", semantic))
        fused = fuse_ranked(query, classification, channels, bounded)
        return self._validate_hits(fused)

    def get_checkpoint(self, checkpoint_id: str) -> CheckpointResult:
        normalized = checkpoint_id.strip().upper()
        if not re.fullmatch(r"CP-\d+", normalized):
            raise ValueError("checkpoint_id must have form CP-N")
        metadata = self.store.get_checkpoint_metadata(normalized)
        sources = tuple(self.get_section(section_id) for section_id in metadata.section_ids)
        return CheckpointResult(metadata=metadata, sources=sources)

    def get_checkpoint_context(
        self, checkpoint_id: str, *, token_budget: int = 7_000
    ) -> CheckpointContext:
        checkpoint = self.get_checkpoint(checkpoint_id)
        dependencies: list[ReferenceResult] = []
        references: list[ReferenceResult] = []
        interfaces: list[SourceItem] = []
        for section_id in checkpoint.metadata.section_ids:
            for result in self.store.get_references(section_id):
                if result.edge.edge_type is EdgeType.DEPENDS_ON:
                    dependencies.append(result)
                elif result.edge.edge_type in {
                    EdgeType.REFERENCES,
                    EdgeType.RELATED_SECTION,
                }:
                    references.append(result)
                elif result.edge.edge_type is EdgeType.USES_TYPE and result.target is not None:
                    interfaces.append(result.target)
        exact_text = "\n".join(source.text for source in checkpoint.sources)
        named_errors = tuple(
            sorted(set(re.findall(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b", exact_text)))
        )
        security_rules = tuple(
            hit.source
            for hit in self.search("security constraint", limit=5)
            if "security" in hit.source.text.casefold()
        )
        fields = checkpoint.metadata.fields
        pack = self.get_context_pack(
            f"Implement {checkpoint.metadata.checkpoint_id} — {checkpoint.metadata.title}",
            token_budget,
        )
        unique_interfaces = {source.provenance.section_id: source for source in interfaces}
        return CheckpointContext(
            checkpoint=checkpoint,
            dependencies=tuple(dependencies),
            references=tuple(references),
            interfaces_models=tuple(unique_interfaces.values()),
            named_errors=named_errors,
            security_rules=security_rules,
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
        documents: set[str] | None = None,
        authority_floor: Authority | None = None,
    ) -> ContextPack:
        from ctx.context_pack import build_context_pack

        if len(task) > self.config.limits.max_query_chars:
            raise ValueError("task exceeds configured max_query_chars")
        return build_context_pack(
            self,
            task,
            token_budget,
            documents=documents,
            authority_floor=authority_floor,
        )

    def get_references(self, section_id: str, *, incoming: bool = False) -> list[ReferenceResult]:
        results = self.store.get_references(section_id, incoming=incoming)
        hits = [SearchHit(source=result.source, score=0, channels=("graph",)) for result in results]
        hits.extend(
            SearchHit(source=result.target, score=0, channels=("graph",))
            for result in results
            if result.target is not None
        )
        self._validate_hits(hits)
        return results

    def get_dependencies(self, section_id: str) -> list[ReferenceResult]:
        results = self.store.get_references(section_id, edge_types=(EdgeType.DEPENDS_ON,))
        self.get_references(section_id)  # validates all source-bearing graph responses
        return results

    def find_symbol(self, symbol: str, *, limit: int = 20) -> list[SymbolResult]:
        if len(symbol) > self.config.limits.max_query_chars:
            raise ValueError("symbol exceeds configured max_query_chars")
        results = self.store.find_symbol(symbol, min(limit, self.config.limits.max_results))
        self._validate_hits(
            [SearchHit(source=result.source, score=0, channels=("symbol",)) for result in results]
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
        return item
