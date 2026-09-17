"""Pluggable, local-only text embedding providers."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class EmbeddingProvider(Protocol):
    """Minimal interface implemented by real and deterministic test backends."""

    @property
    def identity(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def embed_query(self, query: str) -> NDArray[np.float32]: ...


class HashEmbedding:
    """Deterministic lightweight fixture backend; not the production semantic model."""

    def __init__(self, dimensions: int = 64):
        if dimensions < 8:
            raise ValueError("dimensions must be at least 8")
        self._dimensions = dimensions

    @property
    def identity(self) -> str:
        return f"ctx/hash-fixture:1:{self.dimensions}"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def _one(self, text: str) -> NDArray[np.float32]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        tokens = re.findall(r"[\w.@/-]+", text.casefold(), flags=re.UNICODE)
        for token in tokens:
            digest = hashlib.blake2b(token.encode(), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimensions
            sign = 1.0 if digest[8] & 1 else -1.0
            vector[index] += sign
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.stack([self._one(text) for text in texts]).astype(np.float32, copy=False)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        return self._one(query)


class FastEmbedProvider:
    """CPU ONNX provider. Network use is opt-in through ``allow_download``."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        cache_dir: Path | None = None,
        allow_download: bool = False,
    ):
        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("install ctx-context[embeddings] to use FastEmbed") from error
        kwargs: dict[str, object] = {"model_name": model_name}
        if cache_dir is not None:
            kwargs["cache_dir"] = str(cache_dir)
        # FastEmbed forwards local_files_only to its Hugging Face model downloader.
        kwargs["local_files_only"] = not allow_download
        try:
            self._model = TextEmbedding(**kwargs)
        except Exception as error:
            if not allow_download:
                raise RuntimeError(
                    f"embedding model {model_name!r} is not cached; explicitly download it "
                    "with `ctx index --download-model` while online"
                ) from error
            raise
        self._name = model_name
        self._dimensions = int(self._model.embedding_size)

    @property
    def identity(self) -> str:
        return f"fastembed:{self._name}:onnx"

    @property
    def dimensions(self) -> int:
        return self._dimensions

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors = list(self._model.passage_embed(texts))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        return np.asarray(next(iter(self._model.query_embed(query))), dtype=np.float32)


def cosine_scores(query: NDArray[np.float32], matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    """Cosine similarity with stable zero-vector behavior."""
    if matrix.size == 0:
        return np.empty(0, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(matrix, axis=1)
    denominator = row_norms * query_norm
    products = matrix @ query
    result: NDArray[np.float32] = np.divide(
        products,
        denominator,
        out=np.zeros_like(products, dtype=np.float32),
        where=denominator != 0,
    )
    return result
