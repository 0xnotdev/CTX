"""One canonical CTX-internal heading identity scheme.

This is deliberately not advertised as GitHub, CommonMark, or any renderer's anchor algorithm.
It exists only to make CTX parsing, indexing, and deterministic graph resolution agree.
"""

from __future__ import annotations

import json
import unicodedata

HEADING_NORMALIZATION_VERSION = "ctx-heading-nfkc-casefold-separators:1"


def canonical_heading(value: str) -> str:
    """Return the canonical CTX key for heading or local-anchor text.

    Unicode is NFKC-normalized and case-folded. Unicode letters, decimal/numeric characters,
    and combining marks are retained. Every run of whitespace, punctuation, symbols (including
    periods, underscores, hyphens, parentheses, slashes, and emoji), or controls becomes one
    ASCII hyphen. Empty keys become ``section``.
    """
    normalized = unicodedata.normalize("NFKC", value).casefold()
    result: list[str] = []
    separated = True
    for character in normalized:
        category = unicodedata.category(character)
        retained = character.isalnum() or category.startswith("M")
        if retained:
            result.append(character)
            separated = False
        elif not separated and result:
            result.append("-")
            separated = True
    canonical = "".join(result).strip("-")
    return canonical or "section"


def canonical_heading_path_json(value: str) -> str:
    """Canonical slash-joined key for a stored JSON heading path."""
    try:
        parts = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return canonical_heading(value)
    if not isinstance(parts, list):
        return canonical_heading(value)
    return "/".join(canonical_heading(str(part)) for part in parts)
