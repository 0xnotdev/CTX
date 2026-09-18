"""Hard multi-document retrieval evaluation; model use is explicit and local-only."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from ctx.config import add_document_config, initialize_workspace
from ctx.embeddings import FastEmbedProvider, HashEmbedding
from ctx.evaluation import PackCase, RetrievalCase, evaluate_report
from ctx.models import Authority
from ctx.service import ContextEngine


def run(production: bool, model_dir: Path | None) -> dict[str, object]:
    provider = FastEmbedProvider(cache_dir=model_dir) if production else HashEmbedding(64)
    root_parent = Path.cwd() / ".bench-tmp"
    root_parent.mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="ctx-eval-", dir=root_parent) as directory:
            root = Path(directory)
            initialize_workspace(root)
            filler = "Routine bookkeeping records are archived after processing. " * 1_500
            tail = (
                "Each experimental repetition reconstructs identical evidence from a fixed "
                "pseudorandom initialization and compares byte-level manifests before accepting "
                "the run."
            )
            documents = {
                "spec.md": (
                    "# RunManifest\nInterface model RunManifest stores seed and evidence hash.\n"
                    "# CP-2 — Fixtures\nPrepare deterministic fixtures.\n"
                    "# CP-14 — Repeated trials\n"
                    "## Dependencies\n- CP-2\n- RunManifest\n"
                    "## Security constraints\nNever execute inert Markdown or scripts.\n"
                    "## Tests/acceptance criteria\nThree manifests must be byte-identical.\n"
                    "## Verify\n`pytest tests/test_trials.py -q`\n"
                    f"# Deep reproducibility contract\n{filler}{tail}\n"
                ),
                "architecture.md": (
                    "# Context architecture\nSQLite generations atomically bind exact source, FTS, "
                    "vectors, graph, and checkpoints.\n"
                ),
                "research.md": (
                    "# Background research\nRemote vector servers are surveyed but not selected.\n"
                ),
                "old-spec.md": ("# CP-14 — Historical trials\nRunManifest must not be retained.\n"),
                "notes.md": (
                    "# Glossary\nrepeated randomized experiments prove identical results "
                    "multiple random trials validate consistent evidence confirms consistency "
                    "fixed randomness\n"
                ),
            }
            authority = {
                "spec.md": Authority.NORMATIVE,
                "architecture.md": Authority.NORMATIVE,
                "research.md": Authority.REFERENCE,
                "old-spec.md": Authority.HISTORICAL,
                "notes.md": Authority.INFORMAL,
            }
            for path, text in documents.items():
                (root / path).write_text(text, encoding="utf-8")
                add_document_config(root, path, authority[path])
            with ContextEngine(root, embedder=provider) as engine:
                engine.sync_workspace()
                sections = {
                    item.provenance.heading_path[-1]: item.provenance.section_id
                    for item in engine.store.document_sections("spec.md")
                }
                retrieval = [
                    RetrievalCase("checkpoint", "CP-14", sections["CP-14 — Repeated trials"]),
                    RetrievalCase(
                        "low-overlap-tail",
                        "What must happen before accepting rerun evidence?",
                        sections["Deep reproducibility contract"],
                    ),
                    RetrievalCase(
                        "lexical-decoy",
                        "How can repeated randomized experiments prove identical results?",
                        sections["Deep reproducibility contract"],
                    ),
                    RetrievalCase(
                        "model-paraphrase",
                        "Which record binds pseudorandom inputs to proof fingerprints?",
                        sections["RunManifest"],
                    ),
                ]
                required = frozenset(
                    sections[name]
                    for name in (
                        "CP-14 — Repeated trials",
                        "CP-2 — Fixtures",
                        "RunManifest",
                        "Security constraints",
                        "Tests/acceptance criteria",
                        "Verify",
                    )
                )
                packs = [
                    PackCase(
                        "Implement CP-14 with RunManifest security acceptance and verify",
                        7_000,
                        required,
                    )
                ]
                report = evaluate_report(engine, retrieval, packs)
                checkpoint = engine.get_checkpoint("CP-14")
                pack = engine.get_context_pack(packs[0].task, packs[0].token_budget)
                result = report.model_dump(mode="json")
                result.update(
                    {
                        "tier": "production-fastembed" if production else "deterministic-mechanics",
                        "checkpoint_accuracy": float(
                            checkpoint.metadata.root_section_id
                            == sections["CP-14 — Repeated trials"]
                        ),
                        "pack_selected_sections": len(pack.items),
                        "pack_serialized_tokens": pack.serialized_estimated_tokens,
                        "pack_budget": pack.token_budget,
                        "pack_required_ids": len(required),
                        "pack_required_found": len(
                            required & {item.source.provenance.section_id for item in pack.items}
                        ),
                    }
                )
                return result
    finally:
        shutil.rmtree(root_parent, ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--model-dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args.production, args.model_dir)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
