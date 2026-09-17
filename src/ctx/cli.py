"""Command-line adapter over the shared ctx application service."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

import typer
from pydantic import BaseModel
from rich.console import Console

from ctx import __version__
from ctx.config import (
    ConfigError,
    add_document_config,
    database_path,
    initialize_workspace,
    load_config,
    remove_document_config,
)
from ctx.embeddings import FastEmbedProvider
from ctx.mcp_server import run_mcp
from ctx.models import Authority
from ctx.service import ContextEngine
from ctx.store import SQLiteStore

app = typer.Typer(
    name="ctx",
    help="Index local Markdown and serve exact, provenance-rich source context.",
    no_args_is_help=True,
)
console = Console()


def version_callback(value: bool) -> None:
    """Print the package version and exit."""
    if value:
        typer.echo(f"ctx {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool | None,
        typer.Option("--version", callback=version_callback, is_eager=True, help="Show version."),
    ] = None,
) -> None:
    """Local context infrastructure; original Markdown remains authoritative."""


def _root(path: Path) -> Path:
    return path.resolve(strict=True)


def _authority(value: str) -> Authority:
    try:
        return Authority[value.strip().upper()]
    except KeyError as error:
        choices = ", ".join(item.name.lower() for item in Authority)
        raise typer.BadParameter(f"authority must be one of: {choices}") from error


def _data(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_data(item) for item in value]
    if isinstance(value, tuple):
        return [_data(item) for item in value]
    return value


def _emit(value: Any, json_output: bool) -> None:
    data = _data(value)
    encoded = json.dumps(data, ensure_ascii=False, indent=None if json_output else 2)
    if json_output:
        typer.echo(encoded)
    else:
        console.print_json(encoded)


def _embedder(root: Path, *, disabled: bool, download: bool) -> FastEmbedProvider | None:
    if disabled:
        return None
    config = load_config(root)
    try:
        return FastEmbedProvider(
            config.embedding.model,
            cache_dir=root / ".ctx" / "models",
            allow_download=download,
        )
    except RuntimeError as error:
        if download:
            raise
        typer.echo(
            f"warning: {error}; continuing with structural/FTS retrieval",
            err=True,
        )
        return None


def _fail(error: Exception) -> None:
    typer.echo(f"error: {error}", err=True)
    raise typer.Exit(code=2)


@app.command()
def init(
    root: Annotated[Path, typer.Argument(help="Workspace/repository root.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json", help="Emit machine JSON.")] = False,
) -> None:
    """Initialize `.ctx/config.toml` and the SQLite schema."""
    try:
        resolved = _root(root)
        config = initialize_workspace(resolved)
        with SQLiteStore(database_path(resolved)) as store:
            result = {
                "workspace": str(resolved),
                "config": str(resolved / ".ctx" / "config.toml"),
                "index_version": store.index_version(),
                "documents": len(config.documents),
            }
        _emit(result, json_output)
    except (ConfigError, OSError, RuntimeError) as error:
        _fail(error)


@app.command()
def add(
    path: Annotated[str, typer.Argument(help="Relative Markdown path.")],
    authority: Annotated[
        str, typer.Option(help="normative/reference/historical/informal/generated")
    ] = "reference",
    priority: Annotated[int, typer.Option(min=-100, max=100)] = 0,
    root: Annotated[Path, typer.Option(help="Workspace root.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Add/update one read-only Markdown source in workspace configuration."""
    try:
        config = add_document_config(_root(root), path, _authority(authority), priority)
        document = next(item for item in config.documents if item.path == Path(path).as_posix())
        _emit(document, json_output)
    except (ConfigError, OSError, StopIteration, typer.BadParameter) as error:
        _fail(error)


@app.command()
def remove(
    path: Annotated[str, typer.Argument(help="Configured relative path.")],
    root: Annotated[Path, typer.Option(help="Workspace root.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Remove a configured document and its derived index rows; source is untouched."""
    try:
        resolved = _root(root)
        remove_document_config(resolved, path)
        with SQLiteStore(database_path(resolved)) as store:
            removed = store.remove_document(Path(path).as_posix())
        _emit({"path": path, "removed_from_index": removed}, json_output)
    except (ConfigError, OSError) as error:
        _fail(error)


def _run_sync(root: Path, json_output: bool, no_embeddings: bool, download_model: bool) -> None:
    try:
        resolved = _root(root)
        provider = _embedder(resolved, disabled=no_embeddings, download=download_model)
        with ContextEngine(resolved, embedder=provider) as engine:
            _emit(engine.sync_workspace(), json_output)
    except (ConfigError, OSError, RuntimeError, ValueError) as error:
        _fail(error)


@app.command("index")
def index_command(
    root: Annotated[Path, typer.Option(help="Workspace root.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    no_embeddings: Annotated[bool, typer.Option(help="Skip local semantic embeddings.")] = False,
    download_model: Annotated[
        bool, typer.Option(help="Explicitly allow initial local model download.")
    ] = False,
) -> None:
    """Incrementally index all configured sources."""
    _run_sync(root, json_output, no_embeddings, download_model)


@app.command()
def sync(
    root: Annotated[Path, typer.Option(help="Workspace root.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    no_embeddings: Annotated[bool, typer.Option(help="Skip local semantic embeddings.")] = False,
    download_model: Annotated[bool, typer.Option(help="Explicitly download model.")] = False,
) -> None:
    """Synchronize only changed source material."""
    _run_sync(root, json_output, no_embeddings, download_model)


@app.command()
def status(
    root: Annotated[Path, typer.Option(help="Workspace root.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show index version, model, and stale/missing documents."""
    try:
        with ContextEngine(_root(root)) as engine:
            _emit(engine.status(), json_output)
    except (ConfigError, OSError) as error:
        _fail(error)


@app.command("docs")
def docs_command(
    root: Annotated[Path, typer.Option(help="Workspace root.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """List indexed documents with authority, priority, and source hash."""
    try:
        with ContextEngine(_root(root)) as engine:
            _emit(engine.store.list_documents(), json_output)
    except (ConfigError, OSError) as error:
        _fail(error)


@app.command()
def outline(
    path: Annotated[str, typer.Argument(help="Configured document path.")],
    root: Annotated[Path, typer.Option(help="Workspace root.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Show structural heading paths, IDs, hashes, and exact line ranges."""
    try:
        with ContextEngine(_root(root)) as engine:
            items = [
                {
                    "heading_path": item.provenance.heading_path,
                    "provenance": item.provenance.model_dump(mode="json"),
                }
                for item in engine.document_outline(path)
            ]
            _emit(items, json_output)
    except (ConfigError, OSError, KeyError, RuntimeError) as error:
        _fail(error)


@app.command()
def section(
    section_id: Annotated[str, typer.Argument(help="Stable section ID.")],
    root: Annotated[Path, typer.Option(help="Workspace root.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    auto_sync: Annotated[bool, typer.Option(help="Explicitly sync stale local source.")] = False,
) -> None:
    """Get one exact authoritative source section with provenance."""
    try:
        with ContextEngine(_root(root)) as engine:
            _emit(engine.get_section(section_id, auto_sync=auto_sync), json_output)
    except (ConfigError, OSError, KeyError, RuntimeError) as error:
        _fail(error)


@app.command()
def search(
    query: Annotated[str, typer.Argument(help="Task, heading, identifier, or phrase.")],
    root: Annotated[Path, typer.Option(help="Workspace root.")] = Path("."),
    limit: Annotated[int, typer.Option(min=1, max=100)] = 10,
    exact: Annotated[bool, typer.Option(help="Use structural exact matching only.")] = False,
    lexical: Annotated[bool, typer.Option(help="Use BM25 only.")] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Search bounded authoritative sections; never disconnected sentences."""
    try:
        resolved = _root(root)
        provider = None if exact or lexical else _embedder(resolved, disabled=False, download=False)
        with ContextEngine(resolved, embedder=provider) as engine:
            if exact:
                result = engine.search_exact(query, limit=limit)
            elif lexical:
                result = engine.search_lexical(query, limit=limit)
            else:
                result = engine.search(query, limit=limit)
            _emit(result, json_output)
    except (ConfigError, OSError, KeyError, RuntimeError, ValueError) as error:
        _fail(error)


@app.command()
def find(
    symbol: Annotated[str, typer.Argument(help="Exact symbol/type/error.")],
    root: Annotated[Path, typer.Option(help="Workspace root.")] = Path("."),
    limit: Annotated[int, typer.Option(min=1, max=100)] = 20,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Find deterministic symbol definitions/usages with provenance."""
    try:
        with ContextEngine(_root(root)) as engine:
            _emit(engine.find_symbol(symbol, limit=limit), json_output)
    except (ConfigError, OSError, RuntimeError, ValueError) as error:
        _fail(error)


@app.command()
def pack(
    task: Annotated[str, typer.Argument(help="Implementation task/checkpoint.")],
    token_budget: Annotated[int, typer.Option(min=64, max=1_000_000)] = 7_000,
    document: Annotated[list[str] | None, typer.Option("--document")] = None,
    authority_floor: Annotated[str | None, typer.Option()] = None,
    root: Annotated[Path, typer.Option(help="Workspace root.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Build a deliberate token-bounded exact-source context pack."""
    try:
        with ContextEngine(_root(root)) as engine:
            result = engine.get_context_pack(
                task,
                token_budget,
                documents=set(document) if document else None,
                authority_floor=_authority(authority_floor) if authority_floor else None,
            )
            _emit(result, json_output)
    except (ConfigError, OSError, RuntimeError, ValueError, typer.BadParameter) as error:
        _fail(error)


@app.command()
def mcp(
    root: Annotated[Path, typer.Option(help="Workspace root.")] = Path("."),
    no_embeddings: Annotated[bool, typer.Option(help="Disable semantic channel.")] = False,
) -> None:
    """Serve all shared operations over local MCP stdio."""
    try:
        resolved = _root(root)
        provider = _embedder(resolved, disabled=no_embeddings, download=False)
        run_mcp(resolved, embedder=provider)
    except (ConfigError, OSError, RuntimeError, ValueError) as error:
        _fail(error)


if __name__ == "__main__":  # pragma: no cover
    app()
