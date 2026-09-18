"""Document-aware checkpoint recognition and exact-source context models."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pydantic import Field

from ctx.graph import GraphSection
from ctx.models import (
    Authority,
    ContextPack,
    DocumentRecord,
    ReferenceResult,
    SourceItem,
    StrictModel,
)

CHECKPOINT_VERSION = "ctx-checkpoints:3"
_CHECKPOINT = re.compile(r"^\s*(CP-\d+)\s*(?:[—–-]\s*)?(.*)$", re.IGNORECASE)
_FIELD_LINE = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?([A-Za-z][A-Za-z /_-]+?)(?:\*\*)?\s*:\s*(.*)$")
_LIST_LINE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)(.*)$")

_FIELD_NAMES = {
    "goal": "goal",
    "why": "why",
    "dependencies": "dependencies",
    "dependency": "dependencies",
    "exact scope": "exact_scope",
    "scope": "exact_scope",
    "files/modules": "files_modules",
    "files": "files_modules",
    "interfaces/models": "interfaces_models",
    "interfaces": "interfaces_models",
    "models": "interfaces_models",
    "cli behavior": "cli_behavior",
    "tests/acceptance criteria": "tests_acceptance_criteria",
    "tests": "tests_acceptance_criteria",
    "acceptance criteria": "tests_acceptance_criteria",
    "failure conditions": "failure_conditions",
    "out of scope": "out_of_scope",
    "artifacts": "artifacts",
    "verify": "verify",
    "security": "security",
    "security constraints": "security",
}


class GeneratedArtifact(StrictModel):
    schema_version: int = 2
    document_id: str
    checkpoint_id: str
    path: str
    authority: Authority = Authority.GENERATED
    sha256: str
    data: dict[str, Any]
    navigation_only: bool = True


class CheckpointMetadata(StrictModel):
    schema_version: int = 2
    document_id: str
    checkpoint_id: str
    title: str
    root_section_id: str
    section_ids: tuple[str, ...]
    fields: dict[str, tuple[str, ...]]
    generated_artifact: GeneratedArtifact | None = None


class CheckpointResult(StrictModel):
    schema_version: int = 2
    metadata: CheckpointMetadata
    sources: tuple[SourceItem, ...]


class SecurityContextItem(StrictModel):
    source: SourceItem
    applicability: str = Field(pattern="^(directly_applicable|global|semantic_candidate)$")
    reason: str


class NamedError(StrictModel):
    symbol: str
    confidence: float = Field(ge=0, le=1)
    source: str


class CheckpointContext(StrictModel):
    schema_version: int = 2
    checkpoint: CheckpointResult
    dependencies: tuple[ReferenceResult, ...]
    references: tuple[ReferenceResult, ...]
    interfaces_models: tuple[SourceItem, ...]
    named_errors: tuple[str, ...]
    error_evidence: tuple[NamedError, ...] = ()
    security_rules: tuple[SourceItem, ...]
    security_context: tuple[SecurityContextItem, ...] = ()
    acceptance_criteria: tuple[str, ...]
    verification_commands: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    context_pack: ContextPack


def _canonical_field(value: str) -> str | None:
    normalized = re.sub(r"\s+", " ", value.strip().casefold().replace("_", " "))
    return _FIELD_NAMES.get(normalized)


def _clean_values(text: str) -> list[str]:
    values: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        match = _LIST_LINE.match(line)
        values.append((match.group(1) if match else stripped).strip())
    return values


def _section_fields(section: GraphSection) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    lines = section.text.splitlines()
    heading_field = _canonical_field(section.heading)
    if heading_field:
        for value in _clean_values("\n".join(lines[1:])):
            result.setdefault(heading_field, []).append(value)

    index = 1
    while index < len(lines):
        match = _FIELD_LINE.match(lines[index])
        if not match:
            index += 1
            continue
        key = _canonical_field(match.group(1))
        if key is None:
            index += 1
            continue
        inline = match.group(2).strip()
        if inline:
            result.setdefault(key, []).append(inline)
            index += 1
            continue
        # An empty field value owns subsequent list/indented lines until another field or heading.
        index += 1
        continuation: list[str] = []
        while index < len(lines):
            if _FIELD_LINE.match(lines[index]) or re.match(r"^#{1,6}\s+", lines[index]):
                break
            if lines[index].strip():
                continuation.append(lines[index])
            index += 1
        for value in _clean_values("\n".join(continuation)):
            result.setdefault(key, []).append(value)
    return result


def recognize_checkpoints(
    sections: list[GraphSection],
    artifacts: dict[str, GeneratedArtifact] | None = None,
) -> tuple[CheckpointMetadata, ...]:
    """Recognize checkpoint roots and fields without crossing document boundaries."""
    by_parent: dict[str, list[GraphSection]] = {}
    for section in sections:
        if section.parent_id:
            by_parent.setdefault(section.parent_id, []).append(section)

    records: list[CheckpointMetadata] = []
    for root in sections:
        match = _CHECKPOINT.match(root.heading)
        if not match:
            continue
        checkpoint_id = match.group(1).upper()
        descendants: list[GraphSection] = []
        queue = list(by_parent.get(root.id, []))
        while queue:
            child = queue.pop(0)
            if child.document_id != root.document_id:
                continue
            descendants.append(child)
            queue[0:0] = by_parent.get(child.id, [])
        members = [root, *descendants]
        fields: dict[str, list[str]] = {}
        for section in members:
            for key, values in _section_fields(section).items():
                fields.setdefault(key, []).extend(values)
        artifact = (artifacts or {}).get(f"{root.document_id}:{checkpoint_id}")
        records.append(
            CheckpointMetadata(
                document_id=root.document_id,
                checkpoint_id=checkpoint_id,
                title=match.group(2).strip(),
                root_section_id=root.id,
                section_ids=tuple(section.id for section in members),
                fields={key: tuple(values) for key, values in sorted(fields.items())},
                generated_artifact=artifact,
            )
        )
    return tuple(records)


def _artifact_namespace(document_id: str) -> str:
    """Reversible, filesystem-safe persistent document identity (including on Windows)."""
    return quote(document_id, safe="")


def checkpoint_artifact_path(root: Path, document_id: str, checkpoint_id: str) -> Path:
    normalized = checkpoint_id.strip().upper()
    if not re.fullmatch(r"CP-\d+", normalized):
        raise ValueError("checkpoint_id must have form CP-N")
    return root / ".ctx" / "checkpoints" / _artifact_namespace(document_id) / f"{normalized}.json"


def _read_artifact(
    path: Path,
    *,
    root: Path,
    directory: Path,
    max_bytes: int,
    document_id: str,
    checkpoint_id: str,
) -> GeneratedArtifact | None:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(directory.resolve(strict=True))
        if not resolved.is_file() or resolved.stat().st_size > min(max_bytes, 1_000_000):
            return None
        raw = resolved.read_bytes()
        data = json.loads(raw)
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    provenance = data.get("provenance")
    if isinstance(provenance, dict):
        claimed = provenance.get("document_id")
        if claimed is not None and claimed != document_id:
            return None
    claimed = data.get("document_id")
    if claimed is not None and claimed != document_id:
        return None
    return GeneratedArtifact(
        document_id=document_id,
        checkpoint_id=checkpoint_id,
        path=resolved.relative_to(root).as_posix(),
        sha256=hashlib.sha256(raw).hexdigest(),
        data=data,
    )


def _legacy_document(data: dict[str, Any], documents: tuple[DocumentRecord, ...]) -> str | None:
    provenance = data.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    claimed_id = provenance.get("document_id", data.get("document_id"))
    claimed_path = provenance.get("document_path", data.get("document_path"))
    if isinstance(claimed_id, str):
        matches = [record for record in documents if record.id == claimed_id]
        return matches[0].id if len(matches) == 1 else None
    if isinstance(claimed_path, str):
        matches = [record for record in documents if record.path == claimed_path]
        return matches[0].id if len(matches) == 1 else None
    return None


def _cleanup_deleted_namespaces(root: Path, document_ids: set[str]) -> None:
    directory = root / ".ctx" / "checkpoints"
    for document_id in sorted(document_ids):
        path = directory / _artifact_namespace(document_id)
        try:
            if path.is_symlink():
                path.unlink()
            elif path.is_dir():
                shutil.rmtree(path)
        except OSError:
            # Cleanup is best effort; the namespace is absent from the active lookup either way.
            continue


def load_generated_artifacts(
    root: Path,
    max_bytes: int,
    documents: tuple[DocumentRecord, ...],
    *,
    deleted_document_ids: set[str] | None = None,
) -> dict[str, GeneratedArtifact]:
    """Load/migrate document-scoped generated aids; never reinterpret an unscoped CP label."""
    directory = root / ".ctx" / "checkpoints"
    if deleted_document_ids:
        _cleanup_deleted_namespaces(root, deleted_document_ids)
    if not directory.exists():
        return {}

    # Legacy artifacts migrate only when embedded provenance selects exactly one active document.
    for legacy in sorted(directory.glob("CP-*.json")):
        match = re.fullmatch(r"(CP-\d+)\.json", legacy.name, re.IGNORECASE)
        if match is None:
            continue
        try:
            raw = legacy.read_bytes()
            data = json.loads(raw)
        except (OSError, json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        document_id = _legacy_document(data, documents)
        if document_id is None:
            continue
        target = checkpoint_artifact_path(root, document_id, match.group(1))
        if target.exists():
            continue
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        try:
            os.replace(legacy, target)
        except OSError:
            continue

    artifacts: dict[str, GeneratedArtifact] = {}
    for record in sorted(documents, key=lambda item: item.id):
        namespace = directory / _artifact_namespace(record.id)
        if not namespace.is_dir() or namespace.is_symlink():
            continue
        for path in sorted(namespace.glob("CP-*.json")):
            match = re.fullmatch(r"(CP-\d+)\.json", path.name, re.IGNORECASE)
            if match is None:
                continue
            checkpoint_id = match.group(1).upper()
            artifact = _read_artifact(
                path,
                root=root,
                directory=directory,
                max_bytes=max_bytes,
                document_id=record.id,
                checkpoint_id=checkpoint_id,
            )
            if artifact is not None:
                artifacts[f"{record.id}:{checkpoint_id}"] = artifact
    return artifacts
