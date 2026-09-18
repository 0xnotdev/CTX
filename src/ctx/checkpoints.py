"""Document-aware checkpoint recognition and exact-source context models."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import Field

from ctx.graph import GraphSection
from ctx.models import Authority, ContextPack, ReferenceResult, SourceItem, StrictModel

CHECKPOINT_VERSION = "ctx-checkpoints:2"
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
    schema_version: int = 1
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
        if artifact is None:
            artifact = (artifacts or {}).get(checkpoint_id)
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


def load_generated_artifacts(root: Path, max_bytes: int) -> dict[str, GeneratedArtifact]:
    """Load bounded JSON checkpoint aids. They remain GENERATED and navigation-only."""
    directory = root / ".ctx" / "checkpoints"
    if not directory.exists():
        return {}
    artifacts: dict[str, GeneratedArtifact] = {}
    for path in sorted(directory.glob("CP-*.json")):
        try:
            resolved = path.resolve(strict=True)
            resolved.relative_to(directory.resolve(strict=True))
        except (FileNotFoundError, ValueError):
            continue
        if resolved.stat().st_size > min(max_bytes, 1_000_000):
            continue
        match = re.fullmatch(r"(CP-\d+)\.json", path.name, re.IGNORECASE)
        if not match:
            continue
        raw = resolved.read_bytes()
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        if not isinstance(data, dict):
            continue
        checkpoint_id = match.group(1).upper()
        artifacts[checkpoint_id] = GeneratedArtifact(
            path=resolved.relative_to(root).as_posix(),
            sha256=hashlib.sha256(raw).hexdigest(),
            data=data,
        )
    return artifacts
