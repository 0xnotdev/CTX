"""Deterministic Markdown parsing and bounded AST-aware search chunking."""

from __future__ import annotations

import bisect
import hashlib
import math
import re
from collections import defaultdict
from collections.abc import Callable
from pathlib import PurePosixPath

from markdown_it import MarkdownIt
from markdown_it.token import Token

from ctx.heading import canonical_heading
from ctx.models import ParsedDocument, SearchChunk, Section

PARSER_VERSION = "markdown-it-py:4/ctx-sections:3"
CHUNKER_VERSION = "ctx-semantic-windows:3"
EMBEDDING_TEXT_VERSION = "heading-path-prefix:2"
# The default BGE runtime truncates at 512 model tokens.  Leave 64 tokens of headroom and aim
# for a useful 320-token passage with 15% overlap.  FastEmbed supplies exact model token counts;
# the no-model path uses the deterministic approximation documented by
# ``approximate_embedding_tokens`` below.
EMBEDDING_WINDOW_TARGET_TOKENS = 320
EMBEDDING_WINDOW_MAX_TOKENS = 448
EMBEDDING_WINDOW_OVERLAP_TOKENS = 48


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _front_matter(lines: list[str]) -> str | None:
    if not lines or lines[0].rstrip("\r\n") != "---":
        return None
    for index in range(1, len(lines)):
        if lines[index].rstrip("\r\n") in {"---", "..."}:
            return "".join(lines[: index + 1])
    return None


def _heading_tokens(tokens: list[Token]) -> list[tuple[int, int, str]]:
    headings: list[tuple[int, int, str]] = []
    for index, token in enumerate(tokens):
        if token.type != "heading_open" or token.map is None:
            continue
        heading = ""
        if index + 1 < len(tokens) and tokens[index + 1].type == "inline":
            heading = tokens[index + 1].content.strip()
        headings.append((token.map[0], int(token.tag[1]), heading))
    return headings


def _section_id(document_key: str, path: tuple[str, ...], occurrence: int) -> str:
    canonical_path = tuple(canonical_heading(part) for part in path)
    readable = "/".join(canonical_path)
    canonical_identity = "\0".join(canonical_path)
    identity = f"{document_key}\0{canonical_identity}\0{occurrence}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    suffix = f"-{occurrence}" if occurrence > 1 else ""
    return f"sec:{readable}{suffix}:{digest}"


def approximate_embedding_tokens(text: str) -> int:
    """Deterministic no-model estimate used only to size derived search windows.

    The estimate is ``ceil(UTF-8 bytes / 4) + 2`` (the conventional generic text ratio plus
    special-token allowance).  It is not represented as an exact tokenizer bound.  When the
    verified FastEmbed runtime is active, its own ``token_count`` replaces this function.
    """
    return math.ceil(len(text.encode("utf-8")) / 4) + 2 if text else 0


def _largest_fitting_end(
    text: str,
    start: int,
    prefix: str,
    maximum: int,
    count_tokens: Callable[[str], int],
) -> int:
    """Find a deterministic maximal character boundary within the token budget."""
    low, high = start + 1, len(text)
    best = start
    while low <= high:
        middle = (low + high) // 2
        candidate = f"{prefix}\n\n{text[start:middle]}" if prefix else text[start:middle]
        if count_tokens(candidate) <= maximum:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return max(start + 1, best)


def _preferred_end(text: str, start: int, hard_end: int) -> int:
    """Prefer a nearby line/sentence/word boundary without making windows tiny."""
    if hard_end >= len(text):
        return len(text)
    window = text[start:hard_end]
    # Keep a fence opener out of the preceding prose window. Oversized fenced blocks may still
    # be split into exact interior windows, as they cannot fit in one model input.
    marker = re.compile(r"(?m)^[ \t]{0,3}(?:```|~~~)")
    inside_fence = len(marker.findall(text[:start])) % 2 == 1
    starts_at_fence = marker.match(text[start:]) is not None
    fence = marker.search(window[1:]) if not inside_fence and not starts_at_fence else None
    if fence is not None:
        return start + 1 + fence.start()
    floor = max(1, int(len(window) * 0.82))
    matches = list(re.finditer(r"(?:\n|(?<=[.!?])\s+|\s+)", window, flags=re.UNICODE))
    cuts = [match.end() for match in matches if match.end() >= floor]
    return start + cuts[-1] if cuts else hard_end


def _overlap_start(text: str, start: int, end: int, count_tokens: Callable[[str], int]) -> int:
    """Choose at most the configured token overlap while guaranteeing forward progress."""
    low, high = start + 1, end
    best = end
    while low <= high:
        middle = (low + high) // 2
        if count_tokens(text[middle:end]) <= EMBEDDING_WINDOW_OVERLAP_TOKENS:
            best = middle
            high = middle - 1
        else:
            low = middle + 1
    # Align to the next source boundary.  This may reduce, never increase, overlap.
    match = re.search(r"(?:^|\s+)", text[best:end], flags=re.UNICODE)
    aligned = best + match.end() if match and best + match.end() < end else best
    return min(end - 1, max(start + 1, aligned))


def _window_spans(
    text: str, prefix: str, count_tokens: Callable[[str], int]
) -> list[tuple[int, int]]:
    """Create exact, overlapping semantic-window ranges over one authoritative section."""
    if not text:
        return []
    # A pathological heading must not force one-character windows.  It is still authoritative
    # source and is indexed as source text; only its optional navigation prefix is omitted.
    window_prefix = prefix if count_tokens(f"{prefix}\n\n") < EMBEDDING_WINDOW_TARGET_TOKENS else ""
    windows: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        hard_end = _largest_fitting_end(
            text, start, window_prefix, EMBEDDING_WINDOW_TARGET_TOKENS, count_tokens
        )
        end = _preferred_end(text, start, hard_end)
        candidate = f"{window_prefix}\n\n{text[start:end]}" if window_prefix else text[start:end]
        if count_tokens(candidate) > EMBEDDING_WINDOW_MAX_TOKENS:
            end = _largest_fitting_end(
                text, start, window_prefix, EMBEDDING_WINDOW_MAX_TOKENS, count_tokens
            )
        windows.append((start, end))
        if end >= len(text):
            break
        if text[end:].startswith(("```", "~~~")):
            start = end  # a structural fence boundary is more valuable than overlap here
        else:
            next_start = _overlap_start(text, start, end, count_tokens)
            start = end if next_start <= start else next_start
    return windows


def _line_position(
    line_starts: list[int], start: int, end: int
) -> tuple[int, int, int, int | None]:
    start_index = max(0, bisect.bisect_right(line_starts, start) - 1)
    probe = max(start, end - 1)
    end_index = max(0, bisect.bisect_right(line_starts, probe) - 1)
    return (
        start_index + 1,
        end_index + 1,
        start - line_starts[start_index],
        end - line_starts[end_index],
    )


def _search_chunks(
    section: Section,
    document_text: str,
    line_starts: list[int],
    count_tokens: Callable[[str], int],
) -> list[SearchChunk]:
    prefix = " > ".join(section.heading_path)
    chunks: list[SearchChunk] = []
    spans = _window_spans(section.text, prefix, count_tokens)
    for ordinal, (relative_start, relative_end) in enumerate(spans):
        source_text = section.text[relative_start:relative_end]
        absolute_start = section.start_offset + relative_start
        absolute_end = section.start_offset + relative_end
        start_line, end_line, start_column, end_column = _line_position(
            line_starts, absolute_start, absolute_end
        )
        candidate = f"{prefix}\n\n{source_text}" if prefix else source_text
        # Pathological headings can consume the complete input allowance.  They remain present
        # in authoritative source, while navigation text falls back to the exact source window.
        embedding_text = (
            candidate if count_tokens(candidate) <= EMBEDDING_WINDOW_MAX_TOKENS else source_text
        )
        if count_tokens(embedding_text) > EMBEDDING_WINDOW_MAX_TOKENS:
            embedding_text = ""
        source_hash = sha256_text(source_text)
        chunks.append(
            SearchChunk(
                id=f"chk:{section.id}:{ordinal}:{source_hash[:12]}",
                section_id=section.id,
                ordinal=ordinal,
                start_line=start_line,
                end_line=end_line,
                start_column=start_column,
                end_column=end_column,
                start_offset=absolute_start,
                end_offset=absolute_end,
                source_text=source_text,
                source_sha256=source_hash,
                embedding_text=embedding_text,
                embedding_sha256=sha256_text(embedding_text),
                chunker_version=CHUNKER_VERSION,
                token_estimate=count_tokens(embedding_text),
            )
        )
    return chunks


def parse_markdown(
    source_text: str,
    document_key: str = "document.md",
    *,
    embedding_token_counter: Callable[[str], int] | None = None,
) -> ParsedDocument:
    """Parse exact structural sections and bounded search chunks.

    Section boundaries come only from Markdown AST heading maps. Chunks never alter section text:
    each is an exact contiguous source range with separate embedding text.
    """
    key = PurePosixPath(document_key).as_posix()
    lines = source_text.splitlines(keepends=True)
    if not lines and source_text:
        lines = [source_text]
    line_starts = [0]
    for line in lines:
        line_starts.append(line_starts[-1] + len(line))

    markdown = MarkdownIt("commonmark", {"html": False}).enable("table")
    front_matter = _front_matter(lines)
    parse_text = source_text
    if front_matter is not None:
        parse_text = (
            "".join(
                "\n" if line.endswith("\n") else ""
                for line in front_matter.splitlines(keepends=True)
            )
            + source_text[len(front_matter) :]
        )
    headings = _heading_tokens(markdown.parse(parse_text))
    starts = [item[0] for item in headings]
    raw: list[tuple[int, int, str]] = []
    if lines and (not starts or starts[0] > 0):
        raw.append((0, 0, "[preamble]"))
    raw.extend(headings)

    sections: list[Section] = []
    stack: list[Section] = []
    path_counts: defaultdict[tuple[str, ...], int] = defaultdict(int)
    for ordinal, (start_zero, level, heading) in enumerate(raw):
        next_start = raw[ordinal + 1][0] if ordinal + 1 < len(raw) else len(lines)
        next_start = max(start_zero + 1, next_start)
        start_offset = line_starts[min(start_zero, len(lines))]
        end_offset = line_starts[min(next_start, len(lines))]
        text = source_text[start_offset:end_offset]
        path: tuple[str, ...]
        if level == 0:
            stack.clear()
            path = (heading,)
            parent_id = None
        else:
            while stack and stack[-1].level >= level:
                stack.pop()
            parent_id = stack[-1].id if stack else None
            path = (*stack[-1].heading_path, heading) if stack else (heading,)
        canonical_path = tuple(canonical_heading(part) for part in path)
        path_counts[canonical_path] += 1
        identifier = _section_id(key, path, path_counts[canonical_path])
        section = Section(
            id=identifier,
            document_key=key,
            ordinal=ordinal,
            level=level,
            heading=heading,
            heading_path=path,
            parent_id=parent_id,
            start_line=start_zero + 1,
            end_line=max(start_zero + 1, next_start),
            start_offset=start_offset,
            end_offset=end_offset,
            text=text,
            sha256=sha256_text(text),
        )
        sections.append(section)
        if level > 0:
            stack.append(section)

    count_tokens = embedding_token_counter or approximate_embedding_tokens
    chunks = [
        chunk
        for section in sections
        for chunk in _search_chunks(section, source_text, line_starts, count_tokens)
    ]
    return ParsedDocument(
        document_key=key,
        source_text=source_text,
        sha256=sha256_text(source_text),
        front_matter=front_matter,
        sections=tuple(sections),
        chunks=tuple(chunks),
        parser_version=PARSER_VERSION,
        chunker_version=CHUNKER_VERSION,
    )
