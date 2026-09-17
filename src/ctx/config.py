"""Workspace configuration and safe source-path handling."""

from __future__ import annotations

import json
import os
import tomllib
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from ctx.models import Authority

CONFIG_DIR = ".ctx"
CONFIG_FILE = "config.toml"
DB_FILE = "index.sqlite3"
DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024


class ConfigError(ValueError):
    """Raised when workspace configuration or a source path is unsafe."""


class DocumentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    authority: Authority = Authority.REFERENCE
    priority: int = Field(default=0, ge=-100, le=100)


class LimitsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    max_file_bytes: int = Field(default=DEFAULT_MAX_FILE_BYTES, ge=1)
    max_query_chars: int = Field(default=4_096, ge=1, le=1_000_000)
    max_results: int = Field(default=50, ge=1, le=1_000)
    max_response_chars: int = Field(default=1_000_000, ge=1_000)


class EmbeddingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: str = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"


class WorkspaceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int = 1
    documents: tuple[DocumentConfig, ...] = ()
    limits: LimitsConfig = LimitsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()


def config_path(root: Path) -> Path:
    return root / CONFIG_DIR / CONFIG_FILE


def database_path(root: Path) -> Path:
    return root / CONFIG_DIR / DB_FILE


def initialize_workspace(root: Path) -> WorkspaceConfig:
    """Create an empty local workspace without changing any source file."""
    resolved = root.resolve(strict=True)
    target = config_path(resolved)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if target.exists():
        return load_config(resolved)
    config = WorkspaceConfig()
    save_config(resolved, config)
    return config


def load_config(root: Path) -> WorkspaceConfig:
    target = config_path(root.resolve(strict=True))
    if not target.is_file():
        raise ConfigError(f"not a ctx workspace: {root} (run ctx init)")
    with target.open("rb") as handle:
        raw = tomllib.load(handle)
    documents = tuple(
        DocumentConfig(
            path=item["path"],
            authority=Authority[item.get("authority", "REFERENCE").upper()],
            priority=item.get("priority", 0),
        )
        for item in raw.get("documents", [])
    )
    return WorkspaceConfig(
        version=raw.get("version", 1),
        documents=documents,
        limits=LimitsConfig(**raw.get("limits", {})),
        embedding=EmbeddingConfig(**raw.get("embedding", {})),
    )


def save_config(root: Path, config: WorkspaceConfig) -> None:
    """Persist deterministic TOML using atomic replacement."""
    target = config_path(root.resolve(strict=True))
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    lines = [
        f"version = {config.version}",
        "",
        "[limits]",
        f"max_file_bytes = {config.limits.max_file_bytes}",
        f"max_query_chars = {config.limits.max_query_chars}",
        f"max_results = {config.limits.max_results}",
        f"max_response_chars = {config.limits.max_response_chars}",
        "",
        "[embedding]",
        f"backend = {json.dumps(config.embedding.backend)}",
        f"model = {json.dumps(config.embedding.model)}",
    ]
    for document in config.documents:
        lines.extend(
            [
                "",
                "[[documents]]",
                f"path = {json.dumps(document.path)}",
                f"authority = {json.dumps(document.authority.name)}",
                f"priority = {document.priority}",
            ]
        )
    temporary = target.with_suffix(".tmp")
    temporary.write_text("\n".join(lines) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def safe_source_path(root: Path, configured_path: str, max_bytes: int) -> Path:
    """Resolve and validate a read-only Markdown source inside the workspace root."""
    workspace = root.resolve(strict=True)
    candidate = Path(configured_path)
    if candidate.is_absolute():
        raise ConfigError("document path must be relative to the workspace")
    lexical = workspace / candidate
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(workspace)
    except (FileNotFoundError, ValueError) as error:
        raise ConfigError(
            f"document escapes workspace or does not exist: {configured_path}"
        ) from error
    if resolved.suffix.casefold() not in {".md", ".markdown"}:
        raise ConfigError(f"only Markdown files are supported: {configured_path}")
    if not resolved.is_file():
        raise ConfigError(f"document is not a regular file: {configured_path}")
    size = resolved.stat().st_size
    if size > max_bytes:
        raise ConfigError(f"document exceeds max_file_bytes ({size} > {max_bytes})")
    return resolved


def read_source(root: Path, document: DocumentConfig, limits: LimitsConfig) -> str:
    """Open a validated source read-only with strict UTF-8 decoding."""
    source = safe_source_path(root, document.path, limits.max_file_bytes)
    with source.open("r", encoding="utf-8", newline="") as handle:
        return handle.read(limits.max_file_bytes + 1)


def add_document_config(
    root: Path, path: str, authority: Authority, priority: int = 0
) -> WorkspaceConfig:
    config = load_config(root)
    source = safe_source_path(root, path, config.limits.max_file_bytes)
    relative = source.relative_to(root.resolve(strict=True)).as_posix()
    replacement = DocumentConfig(path=relative, authority=authority, priority=priority)
    documents = [item for item in config.documents if item.path != relative]
    documents.append(replacement)
    ordered = tuple(sorted(documents, key=lambda item: item.path))
    updated = config.model_copy(update={"documents": ordered})
    save_config(root, updated)
    return updated


def remove_document_config(root: Path, path: str) -> WorkspaceConfig:
    config = load_config(root)
    normalized = Path(path).as_posix()
    documents = tuple(item for item in config.documents if item.path != normalized)
    if len(documents) == len(config.documents):
        raise ConfigError(f"document is not configured: {path}")
    updated = config.model_copy(update={"documents": documents})
    save_config(root, updated)
    return updated
