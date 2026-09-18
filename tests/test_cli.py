import json
from pathlib import Path

from typer.testing import CliRunner

from ctx import __version__
from ctx.cli import app

runner = CliRunner()


def test_help() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "provenance-rich" in result.stdout


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"ctx {__version__}"


def test_complete_cli_workflow(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spec.md").write_text(
        "# CP-14 — CLI workflow\nGoal: use RunManifest.\n# RunManifest\nInterface model.\n",
        encoding="utf-8",
    )
    assert runner.invoke(app, ["init", "--json"]).exit_code == 0
    added = runner.invoke(app, ["add", "spec.md", "--authority", "normative", "--json"])
    assert added.exit_code == 0, added.output
    indexed = runner.invoke(app, ["index", "--no-embeddings", "--json"])
    assert indexed.exit_code == 0, indexed.output
    assert json.loads(indexed.stdout)["documents_added"] == 1

    searched = runner.invoke(app, ["search", "CP-14", "--exact", "--json"])
    assert searched.exit_code == 0, searched.output
    hit = json.loads(searched.stdout)[0]
    section_id = hit["source"]["provenance"]["section_id"]
    exact = runner.invoke(app, ["section", section_id, "--json"])
    assert exact.exit_code == 0
    assert json.loads(exact.stdout)["text"].startswith("# CP-14")

    packed = runner.invoke(app, ["pack", "Implement CP-14", "--token-budget", "1000", "--json"])
    assert packed.exit_code == 0, packed.output
    assert json.loads(packed.stdout)["estimated_tokens"] <= 1000
    assert runner.invoke(app, ["status", "--json"]).exit_code == 0
    assert runner.invoke(app, ["docs", "--json"]).exit_code == 0
    assert runner.invoke(app, ["outline", "spec.md", "--json"]).exit_code == 0
    assert runner.invoke(app, ["find", "RunManifest", "--json"]).exit_code == 0


def test_doctor_is_read_only(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    (tmp_path / "spec.md").write_text("# A\ntext\n", encoding="utf-8")
    assert runner.invoke(app, ["init"]).exit_code == 0
    assert runner.invoke(app, ["add", "spec.md"]).exit_code == 0
    assert runner.invoke(app, ["index", "--no-embeddings"]).exit_code == 0
    database = tmp_path / ".ctx" / "index.sqlite3"
    before = database.stat().st_mtime_ns
    result = runner.invoke(app, ["doctor", "--json"])
    assert result.exit_code == 0, result.output
    payload = json.loads(result.stdout)
    assert payload["network_attempted"] is False
    assert payload["schema_version"] == 2
    assert database.stat().st_mtime_ns == before


def test_cli_reports_safe_path_errors(tmp_path: Path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.chdir(tmp_path)
    runner.invoke(app, ["init"])
    result = runner.invoke(app, ["add", "../escape.md"])
    assert result.exit_code == 2
    assert "error:" in result.stderr
