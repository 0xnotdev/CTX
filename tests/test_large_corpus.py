from pathlib import Path

from ctx.config import add_document_config, initialize_workspace
from ctx.models import Authority
from ctx.parser import parse_markdown
from ctx.service import ContextEngine


def _exact_lines(seed: list[str], count: int, prefix: str) -> str:
    lines = list(seed)
    while len(lines) < count:
        number = len(lines) + 1
        if number % 250 == 0:
            lines.append(f"## {prefix} duplicate")
        elif number % 251 == 0:
            lines.append(f"Generated technical evidence line {number} for RunManifest Δ.")
        else:
            lines.append(f"{prefix} generated line {number}.")
    return "\n".join(lines[:count]) + "\n"


def test_generated_large_synthetic_corpus_is_exact_and_searchable(tmp_path: Path) -> None:
    tiny = "\n".join(f"tiny line {line}" for line in range(1, 11)) + "\n"
    spec_seed = [
        "---",
        "title: Generated technical spec",
        "---",
        "# CP-14 — Generated reproducibility",
        "Dependencies: CP-2, RunManifest",
        "- list item one",
        "- list item two",
        "> quoted policy: scripts are inert",
        "| field | value |",
        "| --- | --- |",
        "| error | INVALID_EVIDENCE |",
        "```python",
        "# this is not a heading",
        "print('large fence')",
        "```",
        "# Unicode Δ café 安全",
        "A" * 20_000,
        "# Duplicate",
        "same body",
        "# Duplicate",
        "same body",
    ]
    research_seed = [
        "# Generated research",
        "This corpus is synthetic and contains no copied realistic material.",
        "# Network findings",
        "network.remove_ingress@1 supports proof.contract.compile.",
    ]
    spec = _exact_lines(spec_seed, 3_000, "spec")
    research = _exact_lines(research_seed, 20_000, "research")
    assert len(tiny.splitlines()) == 10
    assert len(spec.splitlines()) == 3_000
    assert len(research.splitlines()) == 20_000
    assert "".join(item.text for item in parse_markdown(spec, "spec.md").sections) == spec
    parsed_research = parse_markdown(research, "research.md")
    assert "".join(item.text for item in parsed_research.sections) == research

    initialize_workspace(tmp_path)
    (tmp_path / "tiny.md").write_text(tiny, encoding="utf-8")
    (tmp_path / "spec.md").write_text(spec, encoding="utf-8")
    (tmp_path / "research.md").write_text(research, encoding="utf-8")
    add_document_config(tmp_path, "tiny.md", Authority.INFORMAL)
    add_document_config(tmp_path, "spec.md", Authority.NORMATIVE)
    add_document_config(tmp_path, "research.md", Authority.REFERENCE)
    with ContextEngine(tmp_path) as engine:
        result = engine.index_workspace()
        assert result.documents_added == 3
        assert len(engine.store.list_documents()) == 3
        assert engine.search_exact("CP-14", limit=1)[0].source.provenance.document_path == "spec.md"
        network_hit = engine.search("network.remove_ingress@1", limit=1)[0]
        assert network_hit.source.provenance.document_path == "research.md"
