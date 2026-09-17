"""Typed domain models shared by all ctx adapters and services."""

from __future__ import annotations

from enum import IntEnum, StrEnum

from pydantic import BaseModel, ConfigDict, Field


class Authority(IntEnum):
    """Source authority; larger values rank higher.

    GENERATED is intentionally lowest so generated navigation material cannot outrank source.
    """

    GENERATED = 0
    INFORMAL = 1
    HISTORICAL = 2
    REFERENCE = 3
    NORMATIVE = 4


class EdgeType(StrEnum):
    PARENT_OF = "PARENT_OF"
    CHILD_OF = "CHILD_OF"
    REFERENCES = "REFERENCES"
    DEPENDS_ON = "DEPENDS_ON"
    USES_TYPE = "USES_TYPE"
    RELATED_SECTION = "RELATED_SECTION"


class SourceRange(BaseModel):
    """An exact, inclusive, one-based range from an authoritative source file."""

    model_config = ConfigDict(frozen=True)

    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    text: str
    sha256: str


class Section(BaseModel):
    """Authoritative structural unit represented by exact source text."""

    model_config = ConfigDict(frozen=True)

    id: str
    document_key: str
    ordinal: int = Field(ge=0)
    level: int = Field(ge=0, le=6)
    heading: str
    heading_path: tuple[str, ...]
    parent_id: str | None
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    text: str
    sha256: str


class Chunk(BaseModel):
    """Search-only text derived from one complete authoritative section."""

    model_config = ConfigDict(frozen=True)

    id: str
    section_id: str
    ordinal: int = Field(ge=0)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    text: str
    sha256: str


class DocumentRecord(BaseModel):
    """Configured source document stored in the index."""

    model_config = ConfigDict(frozen=True)

    id: str
    path: str
    authority: Authority
    priority: int
    sha256: str | None = None
    indexed_at: str | None = None


class Provenance(BaseModel):
    """Required provenance accompanying every source-bearing response."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    document_path: str
    document_sha256: str
    authority: Authority
    priority: int
    section_id: str
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    section_sha256: str
    index_version: int


class SyncStats(BaseModel):
    """Observable work performed by an index/sync operation."""

    model_config = ConfigDict(frozen=True)

    documents_added: int = 0
    documents_changed: int = 0
    documents_unchanged: int = 0
    documents_removed: int = 0
    documents_renamed: int = 0
    sections_added: int = 0
    sections_changed: int = 0
    sections_unchanged: int = 0
    sections_removed: int = 0
    chunks_added: int = 0
    chunks_changed: int = 0
    chunks_unchanged: int = 0
    chunks_removed: int = 0
    embeddings_retained: int = 0
    embeddings_created: int = 0
    index_version: int = 0


class IndexStatus(BaseModel):
    model_config = ConfigDict(frozen=True)

    index_version: int
    configured_documents: int
    indexed_documents: int
    stale_documents: tuple[str, ...]
    missing_documents: tuple[str, ...]
    parser_version: str
    embedding_model: str


class SourceItem(BaseModel):
    """Exact source section plus complete provenance."""

    model_config = ConfigDict(frozen=True)

    text: str
    provenance: Provenance
    score: float | None = None
    reason: str | None = None


class ReferenceRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    source_section_id: str
    target_section_id: str | None
    edge_type: EdgeType
    label: str
    resolved: bool


class ReferenceResult(BaseModel):
    """Traversable graph edge with exact source/target where available."""

    model_config = ConfigDict(frozen=True)

    edge: ReferenceRecord
    source: SourceItem
    target: SourceItem | None


class SymbolResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    symbol: str
    kind: str
    source: SourceItem


class SearchHit(BaseModel):
    """Ranked authoritative section returned by one or more retrieval channels."""

    model_config = ConfigDict(frozen=True)

    source: SourceItem
    score: float
    channels: tuple[str, ...]
    matched_terms: tuple[str, ...] = ()


class ParsedDocument(BaseModel):
    """Deterministic structural parse of a Markdown source."""

    model_config = ConfigDict(frozen=True)

    document_key: str
    source_text: str
    sha256: str
    front_matter: str | None
    sections: tuple[Section, ...]
    chunks: tuple[Chunk, ...]
    parser_version: str
