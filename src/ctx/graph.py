"""Deterministic cross-reference graph and symbol extraction without an LLM."""

from __future__ import annotations

import re
from dataclasses import dataclass

from ctx.models import EdgeType, ReferenceRecord


@dataclass(frozen=True)
class GraphSection:
    id: str
    parent_id: str | None
    heading: str
    text: str


@dataclass(frozen=True)
class ExtractedSymbol:
    symbol: str
    section_id: str
    kind: str


@dataclass(frozen=True)
class ExtractedGraph:
    edges: tuple[ReferenceRecord, ...]
    symbols: tuple[ExtractedSymbol, ...]


_CP = re.compile(r"\bCP-\d+\b", re.IGNORECASE)
_SECTION = re.compile(r"§\s*\d+(?:\.\d+)*")
_CAMEL = re.compile(r"\b(?:[A-Z][a-z0-9]+){2,}\b")
_CONSTANT = re.compile(r"\b[A-Z][A-Z0-9]+(?:_[A-Z0-9]+)+\b")
_DOTTED = re.compile(r"\b[A-Za-z_][\w-]*(?:\.[\w-]+)+(?:@\d+)?\b")
_VERSIONED = re.compile(r"\b[A-Za-z_][\w.-]*@\d+\b")
_DEPENDENCY = re.compile(r"(?im)^\s*(?:[-*]\s*)?(?:\*\*)?Dependencies?(?:\*\*)?\s*:\s*(.+)$")
_LINK = re.compile(r"\[[^]]+\]\(#([^)]+)\)")


def _slug(value: str) -> str:
    return re.sub(r"[^\w-]+", "-", value.casefold()).strip("-")


def _symbol_kind(symbol: str) -> str:
    if _CP.fullmatch(symbol):
        return "checkpoint"
    if _CONSTANT.fullmatch(symbol):
        return "constant_or_error"
    if "." in symbol or "@" in symbol:
        return "qualified_identifier"
    return "type"


def _symbols(text: str) -> set[str]:
    found: set[str] = set()
    for pattern in (_CP, _CAMEL, _CONSTANT, _DOTTED, _VERSIONED):
        found.update(match.group(0) for match in pattern.finditer(text))
    return found


def extract_graph(sections: list[GraphSection]) -> ExtractedGraph:
    """Extract resolvable and explicit unresolved edges from exact section text."""
    by_cp: dict[str, list[str]] = {}
    by_mark: dict[str, list[str]] = {}
    by_slug: dict[str, list[str]] = {}
    symbol_owners: dict[str, list[str]] = {}
    symbols: set[tuple[str, str, str]] = set()

    for section in sections:
        by_slug.setdefault(_slug(section.heading), []).append(section.id)
        cp_heading = _CP.search(section.heading)
        if cp_heading:
            by_cp.setdefault(cp_heading.group(0).upper(), []).append(section.id)
        mark_heading = _SECTION.search(section.heading)
        if mark_heading:
            by_mark.setdefault(mark_heading.group(0).replace(" ", ""), []).append(section.id)
        for symbol in _symbols(section.text):
            symbols.add((symbol, section.id, _symbol_kind(symbol)))
            if symbol in section.heading or re.search(
                rf"(?i)\b(?:interface|model|type|class|error)\s+{re.escape(symbol)}\b",
                section.text,
            ):
                symbol_owners.setdefault(symbol.casefold(), []).append(section.id)

    edges: set[tuple[str, str | None, EdgeType, str, bool]] = set()

    def add(
        source: str,
        targets: list[str] | None,
        edge_type: EdgeType,
        label: str,
    ) -> None:
        valid = sorted(target for target in (targets or []) if target != source)
        if valid:
            for target in valid:
                edges.add((source, target, edge_type, label, True))
        elif not targets:
            edges.add((source, None, edge_type, label, False))
        # A label resolving only to itself is intentionally not an edge.

    for section in sections:
        if section.parent_id is not None:
            add(section.parent_id, [section.id], EdgeType.PARENT_OF, section.heading)
            add(section.id, [section.parent_id], EdgeType.CHILD_OF, section.heading)

        own_cp = _CP.search(section.heading)
        own_cp_value = own_cp.group(0).upper() if own_cp else None
        for match in _CP.finditer(section.text):
            label = match.group(0).upper()
            if label != own_cp_value:
                add(section.id, by_cp.get(label), EdgeType.REFERENCES, label)
        own_mark = _SECTION.search(section.heading)
        own_mark_value = own_mark.group(0).replace(" ", "") if own_mark else None
        for match in _SECTION.finditer(section.text):
            label = match.group(0).replace(" ", "")
            if label != own_mark_value:
                add(section.id, by_mark.get(label), EdgeType.REFERENCES, label)
        for anchor in _LINK.findall(section.text):
            add(section.id, by_slug.get(anchor.casefold()), EdgeType.RELATED_SECTION, f"#{anchor}")

        for dependency in _DEPENDENCY.findall(section.text):
            labels = [*_CP.findall(dependency), *_SECTION.findall(dependency)]
            labels.extend(_symbols(dependency))
            for label in sorted(set(labels)):
                if _CP.fullmatch(label):
                    targets = by_cp.get(label.upper())
                elif _SECTION.fullmatch(label):
                    targets = by_mark.get(label.replace(" ", ""))
                else:
                    targets = symbol_owners.get(label.casefold())
                add(section.id, targets, EdgeType.DEPENDS_ON, label)

        for symbol in sorted(_symbols(section.text)):
            if _CAMEL.fullmatch(symbol):
                owners = symbol_owners.get(symbol.casefold())
                if owners and any(owner != section.id for owner in owners):
                    add(section.id, owners, EdgeType.USES_TYPE, symbol)

    records = tuple(
        ReferenceRecord(
            source_section_id=source,
            target_section_id=target,
            edge_type=edge_type,
            label=label,
            resolved=resolved,
        )
        for source, target, edge_type, label, resolved in sorted(
            edges,
            key=lambda edge: (
                edge[0],
                edge[2].value,
                edge[3],
                edge[1] or "",
            ),
        )
    )
    extracted_symbols = tuple(
        ExtractedSymbol(symbol=symbol, section_id=section_id, kind=kind)
        for symbol, section_id, kind in sorted(symbols)
    )
    return ExtractedGraph(edges=records, symbols=extracted_symbols)
