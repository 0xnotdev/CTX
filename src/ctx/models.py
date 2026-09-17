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


class SourceItem(BaseModel):
    """Exact source section plus complete provenance."""

    model_config = ConfigDict(frozen=True)

    text: str
    provenance: Provenance
    score: float | None = None
    reason: str | None = None


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
