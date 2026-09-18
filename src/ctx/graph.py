"""Syntax-aware deterministic cross-reference graph extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ctx.heading import canonical_heading
from ctx.models import (
    Authority,
    EdgeType,
    ReferenceOrigin,
    ReferenceRecord,
    ResolutionStatus,
)

GRAPH_VERSION = "ctx-graph:4"


@dataclass(frozen=True)
class GraphSection:
    id: str
    parent_id: str | None
    heading: str
    text: str
    document_id: str = ""
    authority: Authority = Authority.REFERENCE
    priority: int = 0
    ordinal: int = 0


@dataclass(frozen=True)
class ExtractedSymbol:
    symbol: str
    section_id: str
    kind: str
    origin: ReferenceOrigin = ReferenceOrigin.PROSE
    confidence: float = 1.0


@dataclass(frozen=True)
class ExtractedGraph:
    edges: tuple[ReferenceRecord, ...]
    symbols: tuple[ExtractedSymbol, ...]


_CP = re.compile(r"\bCP-\d+\b", re.IGNORECASE)
_SECTION = re.compile(r"(?:§\s*\d+(?:\.\d+)*|(?:section|chapter)\s+\d+(?:\.\d+)*)", re.I)
_CAMEL = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}(?:Error|Failure|Timeout|Exception)?\b")
_ERROR = re.compile(r"\b[A-Z][A-Za-z0-9]*(?:Error|Failure|Timeout|Exception)\b")
_CONSTANT = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+(?:\d+)?\b")
_DOTTED = re.compile(r"\b(?:[A-Za-z_]\w*\.)+[A-Za-z_]\w*(?::[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)?\b")
_NAMESPACE = re.compile(r"\b[A-Za-z_]\w*(?:::[A-Za-z_]\w*)+\b")
_KEBAB = re.compile(r"(?<!-)\b[a-z][a-z0-9]*(?:-[a-z0-9]+)+\b(?!-)")
_VERSIONED = re.compile(r"\b[A-Za-z_][\w.-]*@\d+\b")
_CLI_FLAG = re.compile(r"(?<!\w)--[a-z0-9][a-z0-9-]*\b")
_FILE_SYMBOL = re.compile(r"\b[\w./-]+\.py:[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?\b")
_DEPENDENCY = re.compile(r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?Dependencies?(?:\*\*)?\s*:\s*(.*)$")
_LINK = re.compile(r"\[[^]]+\]\(([^)#]*)#([^)]+)\)")
_DECLARATION = re.compile(r"(?i)\b(?:interface|model|type|class|error|exception)\s+([\w.:-]+)")


def _symbol_kind(symbol: str) -> str:
    if _CP.fullmatch(symbol):
        return "checkpoint"
    if _ERROR.fullmatch(symbol) or _CONSTANT.fullmatch(symbol):
        return "error_or_code"
    if _CLI_FLAG.fullmatch(symbol):
        return "cli_flag"
    if _FILE_SYMBOL.fullmatch(symbol):
        return "file_symbol"
    if "::" in symbol:
        return "namespace_symbol"
    if "." in symbol or "@" in symbol:
        return "qualified_identifier"
    if "-" in symbol:
        return "kebab_identifier"
    return "type"


def _origin(
    text: str, offset: int, *, heading: bool = False, dependency: bool = False
) -> ReferenceOrigin:
    if heading:
        return ReferenceOrigin.HEADING
    if dependency:
        return ReferenceOrigin.CHECKPOINT_FIELD
    prefix = text[:offset]
    line = text[
        prefix.rfind("\n") + 1 : text.find("\n", offset) if "\n" in text[offset:] else len(text)
    ]
    if prefix.count("```") % 2 or prefix.count("~~~") % 2:
        return ReferenceOrigin.CODE
    relative = offset - (prefix.rfind("\n") + 1)
    if line[:relative].count("`") % 2:
        return ReferenceOrigin.INLINE_CODE
    return ReferenceOrigin.PROSE


def _symbols(text: str) -> list[tuple[str, int]]:
    found: dict[tuple[str, int], None] = {}
    for pattern in (
        _CP,
        _FILE_SYMBOL,
        _NAMESPACE,
        _DOTTED,
        _ERROR,
        _CAMEL,
        _CONSTANT,
        _VERSIONED,
        _CLI_FLAG,
        _KEBAB,
    ):
        for match in pattern.finditer(text):
            found[(match.group(0), match.start())] = None
    return sorted(found, key=lambda item: (item[1], item[0]))


def _choose(
    source: GraphSection, candidates: list[GraphSection]
) -> tuple[str | None, tuple[str, ...], ResolutionStatus, str]:
    candidates = list({item.id: item for item in candidates if item.id != source.id}.values())
    if not candidates:
        return None, (), ResolutionStatus.UNRESOLVED, "no matching declaration"
    same_document = [item for item in candidates if item.document_id == source.document_id]
    pool = same_document or candidates
    authority = max(item.authority for item in pool)
    pool = [item for item in pool if item.authority == authority]
    priority = max(item.priority for item in pool)
    pool = [item for item in pool if item.priority == priority]
    ordered = tuple(item.id for item in sorted(pool, key=lambda item: (item.ordinal, item.id)))
    if len(ordered) == 1:
        reason = "same document" if same_document else "highest authority then priority"
        return ordered[0], ordered, ResolutionStatus.RESOLVED, reason
    return None, ordered, ResolutionStatus.AMBIGUOUS, "equal document/authority/priority candidates"


def extract_graph(sections: list[GraphSection]) -> ExtractedGraph:
    """Extract one resolved edge, or one explicit unresolved/ambiguous edge, per reference."""
    by_cp: dict[str, list[GraphSection]] = {}
    by_mark: dict[str, list[GraphSection]] = {}
    by_slug: dict[str, list[GraphSection]] = {}
    symbol_owners: dict[str, list[GraphSection]] = {}
    symbols: dict[tuple[str, str, str, ReferenceOrigin], ExtractedSymbol] = {}

    for section in sections:
        heading_key = canonical_heading(section.heading)
        by_slug.setdefault(heading_key, []).append(section)
        checkpoint_heading = re.match(r"^cp-(\d+)(?:-|$)", heading_key)
        if checkpoint_heading is not None:
            by_cp.setdefault(f"CP-{checkpoint_heading.group(1)}", []).append(section)
        for match in _SECTION.finditer(section.heading):
            by_mark.setdefault(re.sub(r"\s+", "", match.group(0)).casefold(), []).append(section)
        declared = {match.group(1).casefold() for match in _DECLARATION.finditer(section.text)}
        for symbol, offset in _symbols(section.text):
            origin = _origin(section.text, offset, heading=symbol in section.heading)
            confidence = (
                1.0
                if origin in {ReferenceOrigin.HEADING, ReferenceOrigin.CHECKPOINT_FIELD}
                else 0.8
            )
            item = ExtractedSymbol(symbol, section.id, _symbol_kind(symbol), origin, confidence)
            symbols[(symbol, section.id, item.kind, origin)] = item
            if symbol.casefold() in declared or symbol in section.heading:
                symbol_owners.setdefault(symbol.casefold(), []).append(section)

    edges: dict[tuple[str, EdgeType, str, ReferenceOrigin], ReferenceRecord] = {}

    def add(
        source: GraphSection,
        candidates: list[GraphSection] | None,
        edge_type: EdgeType,
        label: str,
        evidence: str,
        origin: ReferenceOrigin,
        *,
        direct_target: str | None = None,
        unresolved_reason: str | None = None,
    ) -> None:
        target: str | None
        candidate_ids: tuple[str, ...]
        status: ResolutionStatus
        reason: str
        if direct_target is not None:
            target, candidate_ids, status, reason = (
                direct_target,
                (direct_target,),
                ResolutionStatus.RESOLVED,
                "structural AST relationship",
            )
        else:
            target, candidate_ids, status, reason = _choose(source, candidates or [])
            if status is ResolutionStatus.UNRESOLVED and unresolved_reason is not None:
                reason = unresolved_reason
        edges[(source.id, edge_type, label, origin)] = ReferenceRecord(
            source_section_id=source.id,
            target_section_id=target,
            candidate_target_ids=candidate_ids,
            edge_type=edge_type,
            label=label,
            status=status,
            reason=reason,
            evidence=evidence[:500],
            origin=origin,
        )

    by_id = {section.id: section for section in sections}
    for section in sections:
        if section.parent_id is not None and section.parent_id in by_id:
            parent = by_id[section.parent_id]
            add(
                parent,
                None,
                EdgeType.PARENT_OF,
                section.heading,
                section.heading,
                ReferenceOrigin.HEADING,
                direct_target=section.id,
            )
            add(
                section,
                None,
                EdgeType.CHILD_OF,
                parent.heading,
                parent.heading,
                ReferenceOrigin.HEADING,
                direct_target=parent.id,
            )

        own_cp = re.match(r"^cp-(\d+)(?:-|$)", canonical_heading(section.heading))
        own_cp_value = f"CP-{own_cp.group(1)}" if own_cp else None
        for match in _CP.finditer(section.text):
            label = match.group(0).upper()
            if label != own_cp_value:
                add(
                    section,
                    by_cp.get(label),
                    EdgeType.REFERENCES,
                    label,
                    match.group(0),
                    _origin(section.text, match.start()),
                )
        own_mark = _SECTION.search(section.heading)
        own_mark_value = re.sub(r"\s+", "", own_mark.group(0)).casefold() if own_mark else None
        for match in _SECTION.finditer(section.text):
            key = re.sub(r"\s+", "", match.group(0)).casefold()
            if key != own_mark_value:
                add(
                    section,
                    by_mark.get(key),
                    EdgeType.REFERENCES,
                    match.group(0),
                    match.group(0),
                    _origin(section.text, match.start()),
                )
        for match in _LINK.finditer(section.text):
            destination, anchor = match.groups()
            external = bool(destination)
            add(
                section,
                [] if external else by_slug.get(canonical_heading(anchor)),
                EdgeType.RELATED_SECTION,
                f"{destination}#{anchor}",
                match.group(0),
                ReferenceOrigin.LINK,
                unresolved_reason=(
                    "external anchor syntax is not guaranteed by the CTX internal scheme"
                    if external
                    else None
                ),
            )

        dependency_values: list[str] = []
        for match in _DEPENDENCY.finditer(section.text):
            value = match.group(1)
            if not value.strip():
                tail = section.text[match.end() :]
                lines: list[str] = []
                for line in tail.splitlines():
                    if re.match(r"\s*(?:[-*+]\s+|\d+[.)]\s+)", line):
                        lines.append(line)
                    elif lines:
                        break
                value = "\n".join(lines)
            dependency_values.append(value)
        if canonical_heading(section.heading) in {"dependency", "dependencies"}:
            dependency_values.append("\n".join(section.text.splitlines()[1:]))
        for dependency in dependency_values:
            labels = [*_CP.findall(dependency), *_SECTION.findall(dependency)]
            labels.extend(symbol for symbol, _ in _symbols(dependency))
            for label in sorted(set(labels), key=str.casefold):
                if _CP.fullmatch(label):
                    targets = by_cp.get(label.upper())
                elif _SECTION.fullmatch(label):
                    targets = by_mark.get(re.sub(r"\s+", "", label).casefold())
                else:
                    targets = symbol_owners.get(label.casefold())
                add(
                    section,
                    targets,
                    EdgeType.DEPENDS_ON,
                    label,
                    dependency,
                    ReferenceOrigin.CHECKPOINT_FIELD,
                )

        for symbol, offset in _symbols(section.text):
            if _CAMEL.fullmatch(symbol) or _ERROR.fullmatch(symbol) or _DOTTED.fullmatch(symbol):
                owners = symbol_owners.get(symbol.casefold())
                if owners and any(owner.id != section.id for owner in owners):
                    add(
                        section,
                        owners,
                        EdgeType.USES_TYPE,
                        symbol,
                        symbol,
                        _origin(section.text, offset),
                    )

    return ExtractedGraph(
        edges=tuple(
            sorted(
                edges.values(),
                key=lambda edge: (
                    edge.source_section_id,
                    edge.edge_type.value,
                    edge.label,
                    edge.origin.value,
                ),
            )
        ),
        symbols=tuple(
            sorted(
                symbols.values(),
                key=lambda item: (item.symbol.casefold(), item.section_id, item.origin.value),
            )
        ),
    )
