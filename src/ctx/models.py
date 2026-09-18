"""Strict domain models shared by the CLI, MCP adapter, and durable store.

Source-bearing types deliberately distinguish complete sections from exact excerpts.  Derived
chunks, rankings, graph records, and metadata are navigation aids and never replace source.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Default for external and durable boundaries."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class Authority(IntEnum):
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


class ResolutionStatus(StrEnum):
    RESOLVED = "RESOLVED"
    UNRESOLVED = "UNRESOLVED"
    AMBIGUOUS = "AMBIGUOUS"


class ReferenceOrigin(StrEnum):
    PROSE = "PROSE"
    HEADING = "HEADING"
    CODE = "CODE"
    INLINE_CODE = "INLINE_CODE"
    LINK = "LINK"
    CHECKPOINT_FIELD = "CHECKPOINT_FIELD"


class StatusCategory(StrEnum):
    CLEAN = "CLEAN"
    SOURCE_STALE = "SOURCE_STALE"
    METADATA_STALE = "METADATA_STALE"
    PARSER_STALE = "PARSER_STALE"
    EMBEDDINGS_STALE = "EMBEDDINGS_STALE"
    GRAPH_STALE = "GRAPH_STALE"
    SCHEMA_STALE = "SCHEMA_STALE"
    MISSING_SOURCE = "MISSING_SOURCE"


class BudgetMethod(StrEnum):
    STRICT_BYTE_UPPER_BOUND = "STRICT_BYTE_UPPER_BOUND"
    APPROXIMATE_GENERIC = "APPROXIMATE_GENERIC"
    MODEL_SPECIFIC = "MODEL_SPECIFIC"


class CompletenessStatus(StrEnum):
    COMPLETE = "COMPLETE"
    PARTIAL = "PARTIAL"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTING = "CONFLICTING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class ContextMode(StrEnum):
    STANDARD = "STANDARD"
    STRICT_AGENT = "STRICT_AGENT"


class RetrievalMode(StrEnum):
    HYBRID_SEMANTIC = "HYBRID_SEMANTIC"
    LEXICAL_ONLY = "LEXICAL_ONLY"


class CoverageCategory(StrEnum):
    PRIMARY = "primary"
    DEPENDENCIES = "dependencies"
    ARCHITECTURE = "architecture"
    DECISIONS = "decisions"
    CURRENT_STATE = "current_state"
    SECURITY = "security"
    ACCEPTANCE = "acceptance"
    VERIFICATION = "verification"
    TESTING = "testing"
    NORMATIVE = "normative"
    CHECKPOINT_DESCENDANTS = "checkpoint_descendants"


class CoverageStatus(StrEnum):
    COVERED = "COVERED"
    OMITTED = "OMITTED"
    AMBIGUOUS = "AMBIGUOUS"
    CONFLICTING = "CONFLICTING"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class SourceRange(StrictModel):
    """Unambiguous exact contiguous source range.

    Lines are inclusive and one-based. Columns are zero-based Unicode code-point offsets on the
    boundary lines. Absolute offsets are Unicode code-point offsets in the complete document.
    """

    schema_version: Literal[1] = 1
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    start_column: int = Field(default=0, ge=0)
    end_column: int | None = Field(default=None, ge=0)
    start_offset: int = Field(default=0, ge=0)
    end_offset: int = Field(default=0, ge=0)
    sha256: str

    @model_validator(mode="after")
    def valid_range(self) -> SourceRange:
        if self.end_line < self.start_line or self.end_offset < self.start_offset:
            raise ValueError("source range ends before it starts")
        return self


class Section(StrictModel):
    """Complete authoritative Markdown section."""

    schema_version: Literal[2] = 2
    id: str
    document_key: str
    ordinal: int = Field(ge=0)
    level: int = Field(ge=0, le=6)
    heading: str
    heading_path: tuple[str, ...]
    parent_id: str | None
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    start_offset: int = Field(default=0, ge=0)
    end_offset: int = Field(default=0, ge=0)
    text: str
    sha256: str


class SearchChunk(StrictModel):
    """Bounded search unit retaining exact source separately from embedding text."""

    schema_version: Literal[2] = 2
    id: str
    section_id: str
    ordinal: int = Field(ge=0)
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    start_column: int = Field(default=0, ge=0)
    end_column: int | None = Field(default=None, ge=0)
    start_offset: int = Field(default=0, ge=0)
    end_offset: int = Field(default=0, ge=0)
    source_text: str
    source_sha256: str
    embedding_text: str
    embedding_sha256: str
    chunker_version: str
    token_estimate: int = Field(ge=0)

    # V0 compatibility for internal callers. New durable/API fields remain explicit.
    @property
    def text(self) -> str:
        return self.source_text

    @property
    def sha256(self) -> str:
        return self.source_sha256


Chunk = SearchChunk


class DocumentRecord(StrictModel):
    schema_version: Literal[2] = 2
    id: str
    path: str
    authority: Authority
    priority: int
    sha256: str | None = None
    indexed_at: str | None = None


class Provenance(StrictModel):
    """Provenance for a full section or an exact returned subset."""

    schema_version: Literal[2] = 2
    document_id: str
    document_path: str
    document_sha256: str
    authority: Authority
    priority: int
    section_id: str
    heading_path: tuple[str, ...]
    section_start_line: int
    section_end_line: int
    start_line: int
    end_line: int
    start_column: int = 0
    end_column: int | None = None
    start_offset: int = 0
    end_offset: int = 0
    section_sha256: str
    range_sha256: str
    index_generation: int

    @property
    def index_version(self) -> int:
        """Compatibility name retained for V0 callers."""
        return self.index_generation


class SourceRef(StrictModel):
    """Compact source identity; intentionally contains no source text."""

    schema_version: Literal[1] = 1
    document_id: str
    document_path: str
    section_id: str
    heading_path: tuple[str, ...]
    start_line: int
    end_line: int
    start_column: int = 0
    end_column: int | None = None
    section_sha256: str
    range_sha256: str
    authority: Authority
    priority: int
    index_generation: int


class SourceItem(StrictModel):
    """Exact authoritative source; use ``source_type`` to distinguish extent."""

    schema_version: Literal[2] = 2
    source_type: Literal["section", "excerpt"] = "section"
    text: str
    provenance: Provenance
    score: float | None = None
    reason: str | None = None

    @property
    def ref(self) -> SourceRef:
        p = self.provenance
        return SourceRef(
            document_id=p.document_id,
            document_path=p.document_path,
            section_id=p.section_id,
            heading_path=p.heading_path,
            start_line=p.start_line,
            end_line=p.end_line,
            start_column=p.start_column,
            end_column=p.end_column,
            section_sha256=p.section_sha256,
            range_sha256=p.range_sha256,
            authority=p.authority,
            priority=p.priority,
            index_generation=p.index_generation,
        )


class SourceSection(SourceItem):
    source_type: Literal["section"] = "section"


class SourceExcerpt(SourceItem):
    source_type: Literal["excerpt"] = "excerpt"


class OutlineEntry(StrictModel):
    schema_version: Literal[1] = 1
    section_id: str
    heading: str
    heading_path: tuple[str, ...]
    level: int
    ordinal: int
    start_line: int
    end_line: int
    section_sha256: str
    document_id: str
    document_path: str
    document_sha256: str
    authority: Authority
    priority: int
    index_generation: int

    @property
    def provenance(self) -> Provenance:
        """V0 compatibility without loading section text from SQLite."""
        return Provenance(
            document_id=self.document_id,
            document_path=self.document_path,
            document_sha256=self.document_sha256,
            authority=self.authority,
            priority=self.priority,
            section_id=self.section_id,
            heading_path=self.heading_path,
            section_start_line=self.start_line,
            section_end_line=self.end_line,
            start_line=self.start_line,
            end_line=self.end_line,
            section_sha256=self.section_sha256,
            range_sha256=self.section_sha256,
            index_generation=self.index_generation,
        )


class SyncStats(StrictModel):
    schema_version: Literal[2] = 2
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
    index_generation: int = 0

    @property
    def index_version(self) -> int:
        return self.index_generation


class StatusReason(StrictModel):
    category: StatusCategory
    path: str | None = None
    reason: str


class IndexStatus(StrictModel):
    schema_version: Literal[3] = 3
    index_generation: int
    configured_documents: int
    indexed_documents: int
    category: StatusCategory
    reasons: tuple[StatusReason, ...] = ()
    stale_documents: tuple[str, ...] = ()
    missing_documents: tuple[str, ...] = ()
    schema_version_db: int
    parser_version: str
    chunker_version: str
    graph_version: str
    checkpoint_version: str
    retrieval_version: str
    embedding_identity: str
    retrieval_mode: RetrievalMode
    active_channels: tuple[str, ...]

    @property
    def index_version(self) -> int:
        return self.index_generation

    @property
    def embedding_model(self) -> str:
        return self.embedding_identity


class FilterSet(StrictModel):
    documents: frozenset[str] | None = None
    authority_floor: Authority | None = None
    authorities: frozenset[Authority] | None = None
    exclude_documents: frozenset[str] = frozenset()
    heading_prefix: tuple[str, ...] | None = None
    scope: str | None = None


class ContextPackItem(StrictModel):
    schema_version: Literal[2] = 2
    source: SourceItem
    reason: str
    category: str
    relevance: float
    confidence: float = Field(ge=0, le=1)
    estimated_tokens: int
    section_sha256: str
    range_sha256: str
    index_generation: int


class PossibleConflict(StrictModel):
    schema_version: Literal[2] = 2
    label: Literal["POSSIBLE_CONFLICT"] = "POSSIBLE_CONFLICT"
    identifier: str
    reason: str
    sources: tuple[SourceRef, SourceRef]


class OmittedRequiredEvidence(StrictModel):
    schema_version: Literal[1] = 1
    category: CoverageCategory
    reason: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    document_id: str | None = None
    document_path: str | None = None
    section_id: str | None = None
    checkpoint_id: str | None = None
    dependency: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    range_sha256: str | None = None


class AmbiguousEvidence(StrictModel):
    schema_version: Literal[1] = 1
    category: CoverageCategory
    label: str
    reason: str
    source: SourceRef
    candidates: tuple[SourceRef, ...] = ()


class CategoryCoverage(StrictModel):
    schema_version: Literal[1] = 1
    category: CoverageCategory
    status: CoverageStatus
    required: bool
    evidence: tuple[SourceRef, ...] = ()
    omitted_count: int = 0
    notes: tuple[str, ...] = ()


class ContextPack(StrictModel):
    schema_version: Literal[4] = 4
    task: str
    requested_token_budget: int
    token_budget: int
    budget_expanded: bool = False
    budget_method: BudgetMethod
    budget_counter_identity: str
    budget_safety_margin: int
    content_tokens: int
    metadata_tokens: int
    serialized_estimated_tokens: int
    estimated_tokens: int
    token_count_method: str
    items: tuple[ContextPackItem, ...]
    completeness_status: CompletenessStatus
    category_coverage: tuple[CategoryCoverage, ...]
    omitted_required_evidence: tuple[OmittedRequiredEvidence, ...] = ()
    ambiguous_evidence: tuple[AmbiguousEvidence, ...] = ()
    omitted_relevant_sections: tuple[str, ...]
    possible_conflicts: tuple[PossibleConflict, ...] = ()
    index_generation: int
    retrieval_metadata: dict[str, str | int | bool | list[str]]

    @property
    def index_version(self) -> int:
        return self.index_generation


class ReferenceRecord(StrictModel):
    schema_version: Literal[2] = 2
    source_section_id: str
    target_section_id: str | None
    candidate_target_ids: tuple[str, ...] = ()
    edge_type: EdgeType
    label: str
    status: ResolutionStatus
    reason: str
    evidence: str
    origin: ReferenceOrigin

    @property
    def resolved(self) -> bool:
        return self.status is ResolutionStatus.RESOLVED


class ReferenceResult(StrictModel):
    schema_version: Literal[2] = 2
    edge: ReferenceRecord
    source: SourceItem
    target: SourceItem | None
    candidates: tuple[SourceRef, ...] = ()


class SymbolResult(StrictModel):
    schema_version: Literal[2] = 2
    symbol: str
    kind: str
    confidence: float = Field(default=1.0, ge=0, le=1)
    origin: ReferenceOrigin = ReferenceOrigin.PROSE
    source: SourceItem


class SearchHit(StrictModel):
    schema_version: Literal[2] = 2
    source: SourceItem
    score: float
    channels: tuple[str, ...]
    matched_terms: tuple[str, ...] = ()
    chunk_id: str | None = None
    match_start_line: int | None = None
    match_end_line: int | None = None
    index_generation: int


class ParsedDocument(StrictModel):
    schema_version: Literal[2] = 2
    document_key: str
    source_text: str
    sha256: str
    front_matter: str | None
    sections: tuple[Section, ...]
    chunks: tuple[SearchChunk, ...]
    parser_version: str
    chunker_version: str
