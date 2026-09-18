"""Validated private workspace configuration and containment-aware source reads."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import tomllib
from contextlib import suppress
from pathlib import Path, PurePosixPath, PureWindowsPath

from pydantic import Field, ValidationError, model_validator

from ctx.embeddings import DEFAULT_MODEL, DEFAULT_REVISION
from ctx.models import Authority, StrictModel

CONFIG_DIR = ".ctx"
CONFIG_FILE = "config.toml"
DB_FILE = "index.sqlite3"
CONFIG_VERSION = 2
DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024


class ConfigError(ValueError):
    """Actionable malformed-config or unsafe-source error."""


class DocumentConfig(StrictModel):
    path: str
    authority: Authority = Authority.REFERENCE
    priority: int = Field(default=0, ge=-100, le=100)


class LimitsConfig(StrictModel):
    max_file_bytes: int = Field(default=DEFAULT_MAX_FILE_BYTES, ge=1)
    max_query_chars: int = Field(default=4_096, ge=1, le=1_000_000)
    max_results: int = Field(default=50, ge=1, le=1_000)
    max_response_chars: int = Field(default=1_000_000, ge=1_000, le=20_000_000)
    max_line_results: int = Field(default=2_000, ge=1, le=20_000)
    max_token_budget: int = Field(default=1_000_000, ge=64, le=2_000_000)


class EmbeddingConfig(StrictModel):
    backend: str = Field(default="fastembed", pattern="^(fastembed|disabled)$")
    model: str = DEFAULT_MODEL
    revision: str = DEFAULT_REVISION
    model_dir: str | None = None


class WorkspaceConfig(StrictModel):
    version: int = CONFIG_VERSION
    documents: tuple[DocumentConfig, ...] = ()
    limits: LimitsConfig = LimitsConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()

    @model_validator(mode="after")
    def validate_workspace(self) -> WorkspaceConfig:
        if self.version != CONFIG_VERSION:
            raise ValueError(
                f"unsupported config version {self.version}; supported version is {CONFIG_VERSION}"
            )
        seen: set[str] = set()
        for document in self.documents:
            normalized = normalize_configured_path(document.path)
            key = os.path.normcase(normalized).casefold()
            if key in seen:
                raise ValueError(f"duplicate normalized document path: {document.path}")
            seen.add(key)
        if not self.embedding.model.strip() or not self.embedding.revision.strip():
            raise ValueError("embedding model and revision must be non-empty")
        return self


def normalize_configured_path(value: str) -> str:
    if not value or "\x00" in value:
        raise ConfigError("document path must be a non-empty relative path")
    windows = PureWindowsPath(value)
    candidate = PurePosixPath(value.replace("\\", "/"))
    if windows.drive or windows.root or candidate.is_absolute():
        raise ConfigError("document path must be relative to the workspace")
    if any(part in {"", ".", ".."} for part in candidate.parts):
        raise ConfigError(f"document path contains unsafe components: {value}")
    return candidate.as_posix()


def config_path(root: Path) -> Path:
    return root / CONFIG_DIR / CONFIG_FILE


def database_path(root: Path) -> Path:
    return root / CONFIG_DIR / DB_FILE


def initialize_workspace(root: Path) -> WorkspaceConfig:
    resolved = root.resolve(strict=True)
    target = config_path(resolved)
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with suppress(OSError):
        target.parent.chmod(0o700)
    if target.exists():
        return load_config(resolved)
    config = WorkspaceConfig()
    save_config(resolved, config)
    return config


def _parse_authority(value: object, index: int) -> Authority:
    if not isinstance(value, str):
        raise ConfigError(f"documents[{index}].authority must be a string")
    try:
        return Authority[value.strip().upper()]
    except KeyError as error:
        choices = ", ".join(item.name.lower() for item in Authority)
        raise ConfigError(f"documents[{index}].authority must be one of: {choices}") from error


def load_config(root: Path) -> WorkspaceConfig:
    target = config_path(root.resolve(strict=True))
    if not target.is_file():
        raise ConfigError(f"not a ctx workspace: {root} (run ctx init)")
    try:
        with target.open("rb") as handle:
            raw = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"invalid {target}: {error}") from error
    if not isinstance(raw, dict):
        raise ConfigError(f"invalid {target}: top-level TOML table required")
    version = raw.get("version", 1)
    if not isinstance(version, int) or version > CONFIG_VERSION or version < 1:
        raise ConfigError(
            f"unsupported config version {version!r}; supported version is {CONFIG_VERSION}"
        )
    raw_documents = raw.get("documents", [])
    if not isinstance(raw_documents, list):
        raise ConfigError("documents must be an array of tables")
    documents: list[DocumentConfig] = []
    try:
        for index, item in enumerate(raw_documents):
            if not isinstance(item, dict) or "path" not in item:
                raise ConfigError(f"documents[{index}] requires path")
            path = normalize_configured_path(str(item["path"]))
            documents.append(
                DocumentConfig(
                    path=path,
                    authority=_parse_authority(item.get("authority", "REFERENCE"), index),
                    priority=item.get("priority", 0),
                )
            )
        return WorkspaceConfig(
            version=CONFIG_VERSION,
            documents=tuple(documents),
            limits=LimitsConfig.model_validate(raw.get("limits", {})),
            embedding=EmbeddingConfig.model_validate(raw.get("embedding", {})),
        )
    except (ValidationError, TypeError, ValueError) as error:
        if isinstance(error, ConfigError):
            raise
        raise ConfigError(f"invalid {target}: {error}") from error


def save_config(root: Path, config: WorkspaceConfig) -> None:
    """Write private config through a unique same-directory file and durable replace."""
    try:
        config = WorkspaceConfig.model_validate(config.model_dump())
    except ValidationError as error:
        raise ConfigError(f"cannot save invalid config: {error}") from error
    target = config_path(root.resolve(strict=True))
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with suppress(OSError):
        target.parent.chmod(0o700)
    lines = [
        f"version = {CONFIG_VERSION}",
        "",
        "[limits]",
        f"max_file_bytes = {config.limits.max_file_bytes}",
        f"max_query_chars = {config.limits.max_query_chars}",
        f"max_results = {config.limits.max_results}",
        f"max_response_chars = {config.limits.max_response_chars}",
        f"max_line_results = {config.limits.max_line_results}",
        f"max_token_budget = {config.limits.max_token_budget}",
        "",
        "[embedding]",
        f"backend = {json.dumps(config.embedding.backend)}",
        f"model = {json.dumps(config.embedding.model)}",
        f"revision = {json.dumps(config.embedding.revision)}",
    ]
    if config.embedding.model_dir is not None:
        lines.append(f"model_dir = {json.dumps(config.embedding.model_dir)}")
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
    content = ("\n".join(lines) + "\n").encode()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".config.", suffix=".tmp", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "wb", closefd=True) as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
        try:
            directory_fd = os.open(target.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        except OSError:
            pass  # Some supported filesystems/platforms do not permit directory fsync.
    finally:
        with suppress(FileNotFoundError):
            temporary.unlink()


def safe_source_path(root: Path, configured_path: str, max_bytes: int) -> Path:
    workspace = root.resolve(strict=True)
    normalized = normalize_configured_path(configured_path)
    lexical = workspace.joinpath(*PurePosixPath(normalized).parts)
    try:
        resolved = lexical.resolve(strict=True)
        resolved.relative_to(workspace)
    except (FileNotFoundError, OSError, ValueError) as error:
        raise ConfigError(
            f"document escapes workspace or does not exist: {configured_path}"
        ) from error
    if resolved.suffix.casefold() not in {".md", ".markdown"}:
        raise ConfigError(f"only Markdown files are supported: {configured_path}")
    try:
        metadata = resolved.stat()
    except OSError as error:
        raise ConfigError(f"cannot stat document: {configured_path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ConfigError(f"document is not a regular file: {configured_path}")
    if metadata.st_size > max_bytes:
        raise ConfigError(f"document exceeds max_file_bytes ({metadata.st_size} > {max_bytes})")
    return lexical


def read_source(root: Path, document: DocumentConfig, limits: LimitsConfig) -> str:
    """Read exact bytes through a no-follow descriptor and verify identity/size after read."""
    source = safe_source_path(root, document.path, limits.max_file_bytes)
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_BINARY", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(source, flags)
    except OSError as error:
        raise ConfigError(f"cannot safely open document {document.path}: {error}") from error
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ConfigError(f"document is not a regular file: {document.path}")
        chunks: list[bytes] = []
        total = 0
        while total <= limits.max_file_bytes:
            block = os.read(descriptor, min(1024 * 1024, limits.max_file_bytes + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
        raise ConfigError(f"document identity changed while reading: {document.path}")
    data = b"".join(chunks)
    if len(data) > limits.max_file_bytes or after.st_size > limits.max_file_bytes:
        raise ConfigError(
            f"document exceeds max_file_bytes after read ({max(len(data), after.st_size)} > "
            f"{limits.max_file_bytes})"
        )
    if len(data) != after.st_size:
        raise ConfigError(f"document changed size while reading: {document.path}")
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ConfigError(f"document is not valid UTF-8: {document.path}: {error}") from error


def add_document_config(
    root: Path, path: str, authority: Authority, priority: int = 0
) -> WorkspaceConfig:
    config = load_config(root)
    source = safe_source_path(root, path, config.limits.max_file_bytes)
    relative = source.resolve(strict=True).relative_to(root.resolve(strict=True)).as_posix()
    replacement = DocumentConfig(path=relative, authority=authority, priority=priority)
    key = os.path.normcase(relative).casefold()
    documents = [item for item in config.documents if os.path.normcase(item.path).casefold() != key]
    documents.append(replacement)
    updated = config.model_copy(
        update={"documents": tuple(sorted(documents, key=lambda item: item.path))}
    )
    # Revalidate model_copy because Pydantic deliberately does not validate updates by default.
    updated = WorkspaceConfig.model_validate(updated.model_dump())
    save_config(root, updated)
    return updated


def remove_document_config(root: Path, path: str) -> WorkspaceConfig:
    config = load_config(root)
    normalized = normalize_configured_path(path)
    key = os.path.normcase(normalized).casefold()
    documents = tuple(
        item for item in config.documents if os.path.normcase(item.path).casefold() != key
    )
    if len(documents) == len(config.documents):
        raise ConfigError(f"document is not configured: {path}")
    updated = WorkspaceConfig.model_validate(
        config.model_copy(update={"documents": documents}).model_dump()
    )
    save_config(root, updated)
    return updated
