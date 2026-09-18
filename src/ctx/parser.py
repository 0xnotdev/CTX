"""Deterministic Markdown parsing and bounded AST-aware search chunking."""

from __future__ import annotations

import bisect
import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import PurePosixPath

from markdown_it import MarkdownIt
from markdown_it.token import Token

from ctx.models import ParsedDocument, SearchChunk, Section

PARSER_VERSION = "markdown-it-py:4/ctx-sections:2"
CHUNKER_VERSION = "ctx-ast-byte-safe:2"
EMBEDDING_TEXT_VERSION = "heading-path-prefix:1"
# A byte is an upper bound on byte-fallback tokenizer tokens. Keeping embedding text below 480
# therefore stays under the common 512-token BGE limit without relying on model truncation.
CHUNK_TARGET_BYTES = 384
CHUNK_MAX_BYTES = 448


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
    """Internal stable slug, not a claim of GitHub/CommonMark anchor compatibility."""
    normalized = unicodedata.normalize("NFKC", value).casefold()
    slug = re.sub(r"[^\w.-]+", "-", normalized, flags=re.UNICODE).strip("-._")
    return slug or "section"


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
    readable = "/".join(_slug(part) for part in path)
    identity = f"{document_key}\0{'/'.join(path)}\0{occurrence}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:16]
    suffix = f"-{occurrence}" if occurrence > 1 else ""
    return f"sec:{readable}{suffix}:{digest}"


def _byte_len(text: str) -> int:
    return len(text.encode("utf-8"))


def _bounded_split(text: str, maximum: int = CHUNK_TARGET_BYTES) -> list[tuple[int, int]]:
    """Split text at line/sentence/word boundaries with a hard UTF-8 byte ceiling."""
    if not text:
        return []
    result: list[tuple[int, int]] = []
    start = 0
    size = len(text)
    preferred = re.compile(r"(?:\n|(?<=[.!?])\s+|\s+)", re.UNICODE)
    while start < size:
        byte_count = 0
        hard_end = start
        while hard_end < size:
            encoded = text[hard_end].encode("utf-8")
            if byte_count + len(encoded) > maximum:
                break
            byte_count += len(encoded)
            hard_end += 1
        if hard_end == size:
            result.append((start, size))
            break
        if hard_end == start:  # impossible for positive maximum, defensive for odd encodings
            hard_end += 1
        window = text[start:hard_end]
        cuts = [match.end() for match in preferred.finditer(window)]
        end = start + cuts[-1] if cuts and cuts[-1] >= len(window) // 2 else hard_end
        result.append((start, end))
        start = end
    return result


def _block_spans(text: str, lines: list[str], markdown: MarkdownIt) -> list[tuple[int, int]]:
    """Return contiguous block extents, preserving whitespace between AST blocks."""
    line_offsets = [0]
    for line in lines:
        line_offsets.append(line_offsets[-1] + len(line))
    mapped: list[tuple[int, int]] = []
    for token in markdown.parse(text):
        if token.level != 0 or token.map is None:
            continue
        start, end = token.map
        if start < end <= len(lines):
            mapped.append((line_offsets[start], line_offsets[end]))
    mapped.sort()
    nonoverlap: list[tuple[int, int]] = []
    cursor = 0
    for start, end in mapped:
        if end <= cursor:
            continue
        start = max(start, cursor)
        if start > cursor:
            start = cursor  # attach blank/inter-block bytes to the following source block
        nonoverlap.append((start, end))
        cursor = end
    if cursor < len(text):
        nonoverlap.append((cursor, len(text)))
    return nonoverlap or ([(0, len(text))] if text else [])


def _chunk_spans(text: str, markdown: MarkdownIt) -> list[tuple[int, int]]:
    lines = text.splitlines(keepends=True)
    blocks = _block_spans(text, lines, markdown)
    atoms: list[tuple[int, int]] = []
    for start, end in blocks:
        block = text[start:end]
        if _byte_len(block) <= CHUNK_TARGET_BYTES:
            atoms.append((start, end))
        else:
            atoms.extend((start + left, start + right) for left, right in _bounded_split(block))

    chunks: list[tuple[int, int]] = []
    current_start: int | None = None
    current_end = 0
    for start, end in atoms:
        if current_start is None:
            current_start, current_end = start, end
            continue
        proposed = text[current_start:end]
        if start == current_end and _byte_len(proposed) <= CHUNK_TARGET_BYTES:
            current_end = end
        else:
            chunks.append((current_start, current_end))
            current_start, current_end = start, end
    if current_start is not None:
        chunks.append((current_start, current_end))

    # Every emitted source chunk and its optional prefix is independently bounded.
    result: list[tuple[int, int]] = []
    for start, end in chunks:
        if _byte_len(text[start:end]) <= CHUNK_MAX_BYTES:
            result.append((start, end))
        else:
            result.extend(
                (start + left, start + right) for left, right in _bounded_split(text[start:end])
            )
    return result


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
    markdown: MarkdownIt,
) -> list[SearchChunk]:
    prefix = " > ".join(section.heading_path)
    chunks: list[SearchChunk] = []
    for ordinal, (relative_start, relative_end) in enumerate(_chunk_spans(section.text, markdown)):
        source_text = section.text[relative_start:relative_end]
        absolute_start = section.start_offset + relative_start
        absolute_end = section.start_offset + relative_end
        start_line, end_line, start_column, end_column = _line_position(
            line_starts, absolute_start, absolute_end
        )
        candidate = f"{prefix}\n\n{source_text}" if prefix else source_text
        embedding_text = candidate if _byte_len(candidate) <= CHUNK_MAX_BYTES else source_text
        # A pathological heading can itself exceed the model limit; source remains exact and
        # embedding text is still hard-bounded without inventing/truncating authoritative text.
        if _byte_len(embedding_text) > CHUNK_MAX_BYTES:
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
                token_estimate=_byte_len(embedding_text),
            )
        )
    return chunks


def parse_markdown(source_text: str, document_key: str = "document.md") -> ParsedDocument:
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
        path_counts[path] += 1
        identifier = _section_id(key, path, path_counts[path])
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

    chunks = [
        chunk
        for section in sections
        for chunk in _search_chunks(section, source_text, line_starts, markdown)
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
