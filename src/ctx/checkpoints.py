"""Checkpoint recognition and structured navigation metadata."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict

from ctx.graph import GraphSection
from ctx.models import Authority, ContextPack, ReferenceResult, SourceItem

_CHECKPOINT = re.compile(r"^\s*(CP-\d+)\s*(?:[—–-]\s*)?(.*)$", re.IGNORECASE)
_FIELD_LINE = re.compile(r"^\s*(?:[-*]\s*)?(?:\*\*)?([A-Za-z][A-Za-z /_-]+?)(?:\*\*)?\s*:\s*(.*)$")

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
}


class GeneratedArtifact(BaseModel):
    model_config = ConfigDict(frozen=True)

    path: str
    authority: Authority = Authority.GENERATED
    sha256: str
    data: dict[str, Any]
    navigation_only: bool = True


class CheckpointMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    checkpoint_id: str
    title: str
    root_section_id: str
    section_ids: tuple[str, ...]
    fields: dict[str, tuple[str, ...]]
    generated_artifact: GeneratedArtifact | None = None


class CheckpointResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: CheckpointMetadata
    sources: tuple[SourceItem, ...]


class CheckpointContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    checkpoint: CheckpointResult
    dependencies: tuple[ReferenceResult, ...]
    references: tuple[ReferenceResult, ...]
    interfaces_models: tuple[SourceItem, ...]
    named_errors: tuple[str, ...]
    security_rules: tuple[SourceItem, ...]
    acceptance_criteria: tuple[str, ...]
    verification_commands: tuple[str, ...]
    out_of_scope: tuple[str, ...]
    context_pack: ContextPack


def _canonical_field(value: str) -> str | None:
    normalized = re.sub(r"\s+", " ", value.strip().casefold().replace("_", " "))
    return _FIELD_NAMES.get(normalized)


def _body_without_heading(text: str) -> str:
    lines = text.splitlines()
    return "\n".join(lines[1:]).strip()


def recognize_checkpoints(
    sections: list[GraphSection],
    artifacts: dict[str, GeneratedArtifact] | None = None,
) -> tuple[CheckpointMetadata, ...]:
    """Recognize checkpoint roots, descendants, and canonical fields deterministically."""
    by_parent: dict[str, list[GraphSection]] = {}
    by_id = {section.id: section for section in sections}
    for section in sections:
        if section.parent_id:
            by_parent.setdefault(section.parent_id, []).append(section)

    records: list[CheckpointMetadata] = []
    for root in sections:
        match = _CHECKPOINT.match(root.heading)
        if not match:
            continue
        checkpoint_id = match.group(1).upper()
        title = match.group(2).strip()
        descendants: list[GraphSection] = []
        queue = list(by_parent.get(root.id, []))
        while queue:
            child = queue.pop(0)
            descendants.append(child)
            queue[0:0] = by_parent.get(child.id, [])
        members = [root, *descendants]
        fields: dict[str, list[str]] = {}
        for section in members:
            heading_field = _canonical_field(section.heading)
            if heading_field:
                value = _body_without_heading(section.text)
                if value:
                    fields.setdefault(heading_field, []).append(value)
            for line in section.text.splitlines()[1:]:
                field_match = _FIELD_LINE.match(line)
                if not field_match:
                    continue
                key = _canonical_field(field_match.group(1))
                value = field_match.group(2).strip()
                if key and value:
                    fields.setdefault(key, []).append(value)
        records.append(
            CheckpointMetadata(
                checkpoint_id=checkpoint_id,
                title=title,
                root_section_id=root.id,
                section_ids=tuple(section.id for section in members if section.id in by_id),
                fields={key: tuple(values) for key, values in sorted(fields.items())},
                generated_artifact=(artifacts or {}).get(checkpoint_id),
            )
        )
    return tuple(records)


def load_generated_artifacts(root: Path, max_bytes: int) -> dict[str, GeneratedArtifact]:
    """Load bounded `.ctx/checkpoints/CP-N.json` navigation artifacts as GENERATED only."""
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
