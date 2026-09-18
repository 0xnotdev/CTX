"""Explicit, verified, local-only embedding providers and model lifecycle."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from collections.abc import Sequence
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray
from platformdirs import user_cache_path
from pydantic import Field

from ctx.models import StrictModel

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"
# Exact FastEmbed/Qdrant ONNX artifact revision tested for the default model.
DEFAULT_REVISION = "52398278842ec682c6f32300af41344b1c0b0bb2"
MODEL_MANIFEST_VERSION = 1


class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    @property
    def metadata(self) -> dict[str, str]: ...

    def count_tokens(self, text: str) -> int: ...

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]: ...

    def embed_query(self, query: str) -> NDArray[np.float32]: ...


class ModelFile(StrictModel):
    path: str
    size: int = Field(ge=0)
    sha256: str


class ModelManifest(StrictModel):
    schema_version: int = MODEL_MANIFEST_VERSION
    provider: str
    runtime: str
    runtime_version: str
    model_name: str
    revision: str
    dimensions: int = Field(ge=1)
    installed_at: str
    files: tuple[ModelFile, ...]
    artifact_sha256: str

    @property
    def identity(self) -> str:
        return (
            f"{self.provider}:{self.model_name}@{self.revision}:"
            f"{self.artifact_sha256}:{self.dimensions}:{self.runtime_version}"
        )


class HashEmbedding:
    """Deterministic mechanics fixture. It is never a production semantic claim."""

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

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "provider": "ctx",
            "model_name": "hash-fixture",
            "revision": "1",
            "artifact_sha256": "deterministic-test-only",
            "runtime_version": "1",
        }

    def _one(self, text: str) -> NDArray[np.float32]:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        for token in re.findall(r"[\w.@/:-]+", text.casefold(), flags=re.UNICODE):
            digest = hashlib.blake2b(token.encode(), digest_size=16).digest()
            index = int.from_bytes(digest[:8], "little") % self.dimensions
            vector[index] += 1.0 if digest[8] & 1 else -1.0
        norm = float(np.linalg.norm(vector))
        if norm:
            vector /= norm
        return vector

    def count_tokens(self, text: str) -> int:
        """Count the exact token units consumed by this mechanics-only fixture."""
        return len(re.findall(r"[\w.@/:-]+", text.casefold(), flags=re.UNICODE))

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.stack([self._one(text) for text in texts]).astype(np.float32, copy=False)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        return self._one(query)


def model_root(override: Path | None = None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    environment = os.environ.get("CTX_MODEL_DIR")
    if environment:
        return Path(environment).expanduser().resolve()
    return user_cache_path("ctx") / "models"


def model_key(model_name: str, revision: str = DEFAULT_REVISION) -> str:
    readable = re.sub(r"[^A-Za-z0-9._-]+", "--", model_name).strip("-")
    digest = hashlib.sha256(f"{model_name}\0{revision}".encode()).hexdigest()[:12]
    return f"{readable}-{digest}"


def installed_model_path(
    model_name: str = DEFAULT_MODEL,
    revision: str = DEFAULT_REVISION,
    *,
    root: Path | None = None,
) -> Path:
    return model_root(root) / model_key(model_name, revision)


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(directory: Path) -> tuple[ModelFile, ...]:
    rows: list[ModelFile] = []
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        transient = (
            ".locks" in relative.parts
            or path.name in {"manifest.json", "CACHEDIR.TAG", "files_metadata.json"}
            or path.suffix in {".lock", ".refs"}
        )
        if not path.is_file() or transient:
            continue
        rows.append(
            ModelFile(
                path=relative.as_posix(),
                size=path.stat().st_size,
                sha256=_hash_file(path),
            )
        )
    return tuple(rows)


def _detected_revision(directory: Path) -> str | None:
    revisions = {
        path.read_text(encoding="utf-8").strip()
        for path in directory.glob("models--*/refs/main")
        if path.is_file()
    }
    revisions.discard("")
    if len(revisions) > 1:
        raise RuntimeError("model cache contains multiple artifact revisions")
    return next(iter(revisions), None)


def _artifact_hash(files: Sequence[ModelFile]) -> str:
    canonical = json.dumps(
        [item.model_dump(mode="json") for item in files],
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def write_manifest(
    directory: Path,
    *,
    model_name: str,
    revision: str,
    dimensions: int,
    runtime_version: str,
) -> ModelManifest:
    files = _files(directory)
    if not files:
        raise RuntimeError("model installation contains no artifacts")
    manifest = ModelManifest(
        provider="fastembed",
        runtime="onnxruntime",
        runtime_version=runtime_version,
        model_name=model_name,
        revision=revision,
        dimensions=dimensions,
        installed_at=datetime.now(UTC).isoformat(),
        files=files,
        artifact_sha256=_artifact_hash(files),
    )
    target = directory / "manifest.json"
    temporary = directory / f".manifest.{os.getpid()}.tmp"
    temporary.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    os.chmod(temporary, 0o600)
    os.replace(temporary, target)
    return manifest


def verify_model(directory: Path) -> ModelManifest:
    target = directory / "manifest.json"
    try:
        manifest = ModelManifest.model_validate_json(target.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"invalid or missing model manifest: {target}") from error
    actual = _files(directory)
    if actual != manifest.files or _artifact_hash(actual) != manifest.artifact_sha256:
        raise RuntimeError(f"model artifact checksum mismatch: {directory}")
    return manifest


def install_model(
    source: Path,
    *,
    model_name: str = DEFAULT_MODEL,
    revision: str = DEFAULT_REVISION,
    root: Path | None = None,
    dimensions: int = 384,
) -> ModelManifest:
    """Install an existing local artifact directory; this function never uses the network."""
    source = source.resolve(strict=True)
    if not source.is_dir():
        raise RuntimeError("model install source must be a directory")
    destination = installed_model_path(model_name, revision, root=root)
    destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.install"
    if temporary.exists():
        shutil.rmtree(temporary)
    shutil.copytree(source, temporary)
    try:
        import fastembed

        runtime_version = str(getattr(fastembed, "__version__", "unknown"))
    except ImportError as error:
        shutil.rmtree(temporary)
        raise RuntimeError("install ctx-context[embeddings] to install a model") from error
    detected_revision = _detected_revision(temporary)
    if detected_revision is not None and detected_revision != revision:
        shutil.rmtree(temporary)
        raise RuntimeError(
            "local artifact revision does not match requested exact revision: "
            f"{detected_revision} != {revision}"
        )
    manifest = write_manifest(
        temporary,
        model_name=model_name,
        revision=revision,
        dimensions=dimensions,
        runtime_version=runtime_version,
    )
    if destination.exists():
        shutil.rmtree(destination)
    os.replace(temporary, destination)
    return manifest


def download_model(
    *,
    model_name: str = DEFAULT_MODEL,
    revision: str = DEFAULT_REVISION,
    root: Path | None = None,
) -> ModelManifest:
    """The sole network-capable operation in ctx; callers must invoke it explicitly."""
    destination = installed_model_path(model_name, revision, root=root)
    destination.mkdir(mode=0o700, parents=True, exist_ok=True)
    try:
        import fastembed
        from fastembed import TextEmbedding
    except ImportError as error:
        raise RuntimeError("install ctx-context[embeddings] to download a model") from error
    model = TextEmbedding(
        model_name=model_name,
        cache_dir=str(destination),
        local_files_only=False,
    )
    # Force artifact resolution while network use is explicitly allowed.
    dimensions = int(model.embedding_size)
    list(model.passage_embed(["ctx model verification"]))
    detected_revision = _detected_revision(destination)
    if detected_revision is not None and detected_revision != revision:
        raise RuntimeError(
            "downloaded FastEmbed artifact revision does not match requested exact revision: "
            f"{detected_revision} != {revision}"
        )
    return write_manifest(
        destination,
        model_name=model_name,
        revision=revision,
        dimensions=dimensions,
        runtime_version=str(getattr(fastembed, "__version__", "unknown")),
    )


class FastEmbedProvider:
    """Verified local ONNX provider. Construction never downloads or checks for updates."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        revision: str = DEFAULT_REVISION,
        cache_dir: Path | None = None,
        allow_download: bool = False,
    ):
        if allow_download:
            raise RuntimeError("network is permitted only by `ctx model download`")
        directory = installed_model_path(model_name, revision, root=cache_dir)
        manifest = verify_model(directory)
        if manifest.model_name != model_name or manifest.revision != revision:
            raise RuntimeError("installed model identity does not match requested model/revision")
        try:
            from fastembed import TextEmbedding
        except ImportError as error:  # pragma: no cover - optional dependency
            raise RuntimeError("install ctx-context[embeddings] to use FastEmbed") from error
        try:
            self._model = TextEmbedding(
                model_name=model_name,
                cache_dir=str(directory),
                local_files_only=True,
            )
        except Exception as error:
            raise RuntimeError(f"verified local model cannot be loaded: {directory}") from error
        self._manifest = manifest
        self._dimensions = int(self._model.embedding_size)
        if self._dimensions != manifest.dimensions:
            raise RuntimeError("installed model dimensions do not match manifest")

    @property
    def identity(self) -> str:
        return self._manifest.identity

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def metadata(self) -> dict[str, str]:
        return {
            "provider": self._manifest.provider,
            "model_name": self._manifest.model_name,
            "revision": self._manifest.revision,
            "artifact_sha256": self._manifest.artifact_sha256,
            "runtime_version": self._manifest.runtime_version,
        }

    def count_tokens(self, text: str) -> int:
        """Use the verified local runtime's tokenizer; no model/network call is involved."""
        return int(self._model.token_count(text))

    def embed_documents(self, texts: Sequence[str]) -> NDArray[np.float32]:
        vectors = list(self._model.passage_embed(texts))
        return np.asarray(vectors, dtype=np.float32)

    def embed_query(self, query: str) -> NDArray[np.float32]:
        return cast(
            NDArray[np.float32],
            np.asarray(next(iter(self._model.query_embed(query))), dtype=np.float32),
        )


@lru_cache(maxsize=1)
def runtime_versions() -> dict[str, str]:
    versions = {"numpy": np.__version__}
    try:
        import fastembed

        versions["fastembed"] = str(getattr(fastembed, "__version__", "unknown"))
    except ImportError:
        versions["fastembed"] = "not-installed"
    return versions


def cosine_scores(query: NDArray[np.float32], matrix: NDArray[np.float32]) -> NDArray[np.float32]:
    if matrix.size == 0:
        return np.empty(0, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    row_norms = np.linalg.norm(matrix, axis=1)
    denominator = row_norms * query_norm
    products = matrix @ query
    return cast(
        NDArray[np.float32],
        np.divide(
            products,
            denominator,
            out=np.zeros_like(products, dtype=np.float32),
            where=denominator != 0,
        ),
    )
