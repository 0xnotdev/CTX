"""Dated mechanics/production measurements; never downloads a model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import statistics
import tempfile
from collections.abc import Callable
from pathlib import Path
from time import perf_counter
from typing import Any

from ctx.config import add_document_config, database_path, initialize_workspace, safe_source_path
from ctx.embeddings import FastEmbedProvider, HashEmbedding
from ctx.models import Authority
from ctx.parser import parse_markdown
from ctx.service import ContextEngine


def timed(callable_: Callable[[], object], repeats: int) -> float:
    samples: list[float] = []
    for _ in range(repeats):
        start = perf_counter()
        callable_()
        samples.append((perf_counter() - start) * 1_000)
    return statistics.median(samples)


def generated(lines: int, prefix: str) -> str:
    values = []
    for number in range(1, lines + 1):
        if number % 200 == 1:
            values.append(f"# {prefix} section {number // 200}")
        elif number % 200 == 2:
            values.append(
                f"RunManifest network stabilization INVALID_EVIDENCE generated line {number}."
            )
        else:
            values.append(f"Generated local benchmark material {number}.")
    return "\n".join(values) + "\n"


def environment() -> dict[str, str]:
    packages = ("ctx-context", "fastembed", "mcp", "numpy", "onnxruntime", "pydantic")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return {
        "date": "2026-09-18",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "processor": platform.processor() or "not-reported",
        "cpu_count": str(os.cpu_count()),
        "dependencies": json.dumps(versions, sort_keys=True),
    }


def run(production: bool, model_dir: Path | None) -> dict[str, Any]:
    spec = generated(3_000, "spec")
    research = generated(20_000, "research")
    huge = "# Huge single section\n" + ("one enormous requirement sentence. " * 20_000)
    measurements: dict[str, Any] = {
        **environment(),
        "tier": "production-fastembed" if production else "deterministic-mechanics",
        "spec_lines": 3_000,
        "research_lines": 20_000,
        "parse_spec_ms": timed(lambda: parse_markdown(spec, "spec.md"), 3),
        "parse_research_ms": timed(lambda: parse_markdown(research, "research.md"), 3),
        "parse_huge_section_ms": timed(lambda: parse_markdown(huge, "huge.md"), 3),
    }
    if production:
        start = perf_counter()
        provider = FastEmbedProvider(cache_dir=model_dir)
        measurements["model_cold_load_ms"] = (perf_counter() - start) * 1_000
    else:
        provider = HashEmbedding(64)
    measurements["embedding_identity"] = provider.identity

    temporary_root = Path.cwd() / ".bench-tmp"
    temporary_root.mkdir(exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="ctx-benchmark-", dir=temporary_root) as directory:
            root = Path(directory)
            initialize_workspace(root)
            (root / "spec.md").write_text(spec, encoding="utf-8")
            (root / "research.md").write_text(research, encoding="utf-8")
            (root / "huge.md").write_text(huge, encoding="utf-8")
            add_document_config(root, "spec.md", Authority.NORMATIVE)
            add_document_config(root, "research.md", Authority.REFERENCE)
            add_document_config(root, "huge.md", Authority.NORMATIVE)
            with ContextEngine(root, embedder=provider) as engine:
                start = perf_counter()
                stats = engine.index_workspace()
                measurements["index_ms"] = (perf_counter() - start) * 1_000
                measurements["chunk_count"] = engine.store.chunk_count()
                measurements["embeddings_created"] = stats.embeddings_created
                measurements["exact_lookup_median_ms"] = timed(
                    lambda: engine.search_exact("RunManifest", limit=3), 50
                )
                measurements["fts_median_ms"] = timed(
                    lambda: engine.search_lexical("network stabilization", limit=5), 30
                )
                if production:
                    measurements["warm_semantic_median_ms"] = timed(
                        lambda: engine.search_semantic(
                            "recover equivalent evidence after repeated execution", limit=5
                        ),
                        10,
                    )
                measurements["warm_hybrid_median_ms"] = timed(
                    lambda: engine.search("network stabilization error", limit=5), 10
                )
                measurements["pack_median_ms"] = timed(
                    lambda: engine.get_context_pack("network stabilization RunManifest", 7_000),
                    5,
                )
                start = perf_counter()
                unchanged = engine.sync_workspace()
                measurements["unchanged_sync_ms"] = (perf_counter() - start) * 1_000
                measurements["unchanged_generation"] = unchanged.index_generation
                measurements["db_bytes"] = database_path(root).stat().st_size

            hundred = root / "hundred.md"
            with hundred.open("wb") as handle:
                handle.truncate(100 * 1024 * 1024)
            start = perf_counter()
            try:
                safe_source_path(root, "hundred.md", 25 * 1024 * 1024)
            except ValueError:
                measurements["100mb_default"] = "rejected_by_configured_25MiB_bound"
            measurements["100mb_bound_check_ms"] = (perf_counter() - start) * 1_000
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
    measurements["peak_rss_kib"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return measurements


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
