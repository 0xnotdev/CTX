from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from ctx.config import add_document_config, initialize_workspace
from ctx.models import Authority
from ctx.parser import (
    EMBEDDING_WINDOW_MAX_TOKENS,
    EMBEDDING_WINDOW_OVERLAP_TOKENS,
    EMBEDDING_WINDOW_TARGET_TOKENS,
    parse_markdown,
    sha256_text,
)
from ctx.service import ContextEngine


class _ConceptEmbedding:
    """Small semantic fixture: synonym groups share dimensions; no network is involved."""

    dimensions = 8
    identity = "test/concept-embedding:1"
    metadata = {
        "provider": "test",
        "model_name": "concept",
        "revision": "1",
        "artifact_sha256": "fixture",
        "runtime_version": "1",
    }
    _concepts = {
        "automobile": 0,
        "car": 0,
        "engine": 1,
        "motor": 1,
        "repair": 2,
        "fix": 2,
        "security": 3,
    }

    def count_tokens(self, text: str) -> int:
        return len(re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)) + 2

    def _one(self, text: str) -> NDArray[np.float32]:
        value = np.zeros(self.dimensions, dtype=np.float32)
        for word in re.findall(r"\w+", text.casefold(), flags=re.UNICODE):
            dimension = self._concepts.get(word)
            if dimension is not None:
                value[dimension] += 1
        norm = float(np.linalg.norm(value))
        return value / norm if norm else value

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        return np.stack([self._one(text) for text in texts])

    def embed_query(self, query: str) -> NDArray[np.float32]:
        return self._one(query)


def test_semantic_windows_cross_old_boundary_overlap_and_preserve_authority() -> None:
    # The two halves were farther apart than the old 384-byte target but fit in one
    # model-sized semantic window.
    source = (
        "# Large requirement\n"
        + ("ordinary setup detail " * 20)
        + "CROSSING_ALPHA "
        + ("transition material " * 12)
        + "CROSSING_OMEGA\n"
        + ("followup evidence sentence. " * 120)
    )
    parsed = parse_markdown(source, "huge.md")

    assert len(parsed.sections) == 1
    assert parsed.sections[0].text == source
    assert "".join(section.text for section in parsed.sections) == source
    assert any(
        "CROSSING_ALPHA" in chunk.source_text and "CROSSING_OMEGA" in chunk.source_text
        for chunk in parsed.chunks
    )
    assert len(parsed.chunks) >= 2
    assert all(
        right.start_offset < left.end_offset
        for left, right in zip(parsed.chunks, parsed.chunks[1:], strict=False)
    )
    assert all(0 < chunk.token_estimate <= EMBEDDING_WINDOW_MAX_TOKENS for chunk in parsed.chunks)
    assert any(
        EMBEDDING_WINDOW_TARGET_TOKENS - EMBEDDING_WINDOW_OVERLAP_TOKENS
        <= chunk.token_estimate
        <= EMBEDDING_WINDOW_MAX_TOKENS
        for chunk in parsed.chunks[:-1]
    )


def test_low_lexical_overlap_semantic_query_hydrates_exact_source(tmp_path: Path) -> None:
    initialize_workspace(tmp_path)
    target = "# Maintenance\nThe automobile engine repair procedure preserves the warranty.\n"
    (tmp_path / "manual.md").write_text(
        target + "# Access control\nSecurity policy for credentials.\n", encoding="utf-8"
    )
    add_document_config(tmp_path, "manual.md", Authority.NORMATIVE)

    with ContextEngine(tmp_path, embedder=_ConceptEmbedding()) as engine:
        engine.sync_workspace()
        assert not engine.search_lexical("fix car motor", limit=5)
        hit = engine.search_semantic("fix car motor", limit=1)[0]
        assert hit.source.text == target
        provenance = hit.source.provenance
        assert provenance.range_sha256 == hashlib.sha256(target.encode()).hexdigest()
        assert provenance.section_sha256 == provenance.range_sha256
        current = (tmp_path / provenance.document_path).read_text(encoding="utf-8")
        assert current[provenance.start_offset : provenance.end_offset] == hit.source.text


def test_unicode_and_code_fence_window_ranges_are_exact() -> None:
    payload = "\n".join(f"print('行🙂 {index}')" for index in range(260))
    source = f"# Unicode／Code 🙂\nIntro café.\n```python\n{payload}\n```\n尾部 Ω.\n"
    parsed = parse_markdown(source, "unicode.md")
    assert len(parsed.chunks) > 2
    for ordinal, chunk in enumerate(parsed.chunks):
        assert chunk.ordinal == ordinal
        assert source[chunk.start_offset : chunk.end_offset] == chunk.source_text
        assert chunk.source_sha256 == sha256_text(chunk.source_text)
        assert chunk.id.endswith(chunk.source_sha256[:12])
        start_prefix = source[: chunk.start_offset]
        end_prefix = source[: chunk.end_offset]
        assert chunk.start_line == start_prefix.count("\n") + 1
        assert chunk.end_line == max(chunk.start_line, end_prefix.rstrip("\n").count("\n") + 1)
