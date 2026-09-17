"""Deterministic Markdown AST-to-section parser.

The parser uses markdown-it token source maps to locate headings. It slices only at heading
boundaries reported by the AST; it never tokenizes source text into arbitrary windows. Each
section therefore contains a heading and all of its direct body through the next heading.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections import defaultdict
from pathlib import PurePosixPath

from markdown_it import MarkdownIt
from markdown_it.token import Token

from ctx.models import Chunk, ParsedDocument, Section

PARSER_VERSION = "markdown-it-py:4/ctx-sections:1"


def sha256_text(text: str) -> str:
    """Return a stable SHA-256 over exact UTF-8 source text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _slug(value: str) -> str:
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
        level = int(token.tag[1])
        heading = ""
        if index + 1 < len(tokens) and tokens[index + 1].type == "inline":
            heading = tokens[index + 1].content.strip()
        headings.append((token.map[0], level, heading))
    return headings


def _section_id(document_key: str, path: tuple[str, ...], occurrence: int) -> str:
    readable = "/".join(_slug(part) for part in path)
    identity = f"{document_key}\0{'/'.join(path)}\0{occurrence}"
    digest = hashlib.sha256(identity.encode()).hexdigest()[:12]
    suffix = f"-{occurrence}" if occurrence > 1 else ""
    return f"sec:{readable}{suffix}:{digest}"


def parse_markdown(source_text: str, document_key: str = "document.md") -> ParsedDocument:
    """Parse Markdown into exact, ordered sections and search-only chunks.

    Line ranges are one-based and inclusive. Content before the first heading (including YAML
    front matter) becomes a level-zero ``[preamble]`` section. Empty input has no sections.
    Duplicate heading paths receive deterministic occurrence suffixes.
    """
    key = PurePosixPath(document_key).as_posix()
    lines = source_text.splitlines(keepends=True)
    if not lines and source_text:
        lines = [source_text]

    markdown = MarkdownIt("commonmark", {"html": False}).enable("table")
    front_matter = _front_matter(lines)
    # CommonMark otherwise interprets YAML delimiters as setext/thematic syntax. Blank only
    # the closed front-matter prefix for AST navigation while retaining every original byte.
    parse_text = source_text
    if front_matter is not None:
        parse_text = "".join(
            "\n" if line.endswith("\n") else "" for line in front_matter.splitlines(keepends=True)
        )
        parse_text += source_text[len(front_matter) :]
    headings = _heading_tokens(markdown.parse(parse_text))
    starts = [item[0] for item in headings]
    raw: list[tuple[int, int, str]] = []
    if lines and (not starts or starts[0] > 0):
        raw.append((0, 0, "[preamble]"))
    raw.extend(headings)

    sections: list[Section] = []
    chunks: list[Chunk] = []
    stack: list[Section] = []
    path_counts: defaultdict[tuple[str, ...], int] = defaultdict(int)

    for ordinal, (start_zero, level, heading) in enumerate(raw):
        next_start = raw[ordinal + 1][0] if ordinal + 1 < len(raw) else len(lines)
        # A parser map may point at EOF for malformed input; retain a valid exact range.
        next_start = max(start_zero + 1, next_start)
        text = "".join(lines[start_zero:next_start])
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
        section_id = _section_id(key, path, path_counts[path])
        section = Section(
            id=section_id,
            document_key=key,
            ordinal=ordinal,
            level=level,
            heading=heading,
            heading_path=path,
            parent_id=parent_id,
            start_line=start_zero + 1,
            end_line=max(start_zero + 1, next_start),
            text=text,
            sha256=sha256_text(text),
        )
        sections.append(section)
        if level > 0:
            stack.append(section)
        chunk_hash = sha256_text(text)
        chunks.append(
            Chunk(
                id=f"chk:{section_id}:{chunk_hash[:12]}",
                section_id=section_id,
                ordinal=0,
                start_line=section.start_line,
                end_line=section.end_line,
                text=text,
                sha256=chunk_hash,
            )
        )

    return ParsedDocument(
        document_key=key,
        source_text=source_text,
        sha256=sha256_text(source_text),
        front_matter=front_matter,
        sections=tuple(sections),
        chunks=tuple(chunks),
        parser_version=PARSER_VERSION,
    )
