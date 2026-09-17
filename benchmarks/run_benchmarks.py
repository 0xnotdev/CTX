"""Reproducible local measurements; outputs observations, never performance promises."""

from __future__ import annotations

import json
import statistics
import tempfile
from pathlib import Path
from time import perf_counter

from ctx.config import add_document_config, initialize_workspace, safe_source_path
from ctx.embeddings import HashEmbedding
from ctx.models import Authority
from ctx.parser import parse_markdown
from ctx.service import ContextEngine


def timed(callable_, repeats: int) -> float:  # type: ignore[no-untyped-def]
    samples = []
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


def main() -> None:
    spec = generated(3_000, "spec")
    research = generated(20_000, "research")
    measurements: dict[str, float | int | str] = {
        "spec_lines": 3_000,
        "research_lines": 20_000,
        "parse_spec_ms": timed(lambda: parse_markdown(spec, "spec.md"), 3),
        "parse_research_ms": timed(lambda: parse_markdown(research, "research.md"), 3),
    }
    with tempfile.TemporaryDirectory(prefix="ctx-benchmark-") as directory:
        root = Path(directory)
        initialize_workspace(root)
        (root / "spec.md").write_text(spec, encoding="utf-8")
        (root / "research.md").write_text(research, encoding="utf-8")
        add_document_config(root, "spec.md", Authority.NORMATIVE)
        add_document_config(root, "research.md", Authority.REFERENCE)
        with ContextEngine(root, embedder=HashEmbedding(64)) as engine:
            start = perf_counter()
            engine.index_workspace()
            measurements["index_ms"] = (perf_counter() - start) * 1_000
            measurements["exact_lookup_median_ms"] = timed(
                lambda: engine.search_exact("RunManifest", limit=3), 100
            )
            measurements["fts_median_ms"] = timed(
                lambda: engine.search_lexical("network stabilization", limit=5), 50
            )
            measurements["warm_hybrid_median_ms"] = timed(
                lambda: engine.search("network stabilization error", limit=5), 20
            )
            measurements["pack_median_ms"] = timed(
                lambda: engine.get_context_pack("network stabilization RunManifest", 7_000),
                10,
            )

        # The V0 default is a deliberate 25 MiB/source safety bound. Measure predictable
        # rejection of a sparse 100 MiB source rather than pretending it is supported by default.
        hundred = root / "hundred.md"
        with hundred.open("wb") as handle:
            handle.truncate(100 * 1024 * 1024)
        start = perf_counter()
        try:
            safe_source_path(root, "hundred.md", 25 * 1024 * 1024)
        except ValueError:
            measurements["100mb_default"] = "rejected_by_configured_25MiB_bound"
        measurements["100mb_bound_check_ms"] = (perf_counter() - start) * 1_000

    print(json.dumps(measurements, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
