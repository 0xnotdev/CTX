"""Typer CLI over the same ContextEngine factory and services as MCP."""

from __future__ import annotations

import importlib.metadata
import json
import platform
import sqlite3
import sys
from pathlib import Path
from typing import Annotated, Any, TypedDict

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
from ctx.embeddings import (
    DEFAULT_MODEL,
    DEFAULT_REVISION,
    download_model,
    install_model,
    installed_model_path,
    model_root,
    verify_model,
)
from ctx.mcp_server import run_mcp
from ctx.models import Authority
from ctx.service import ContextEngine, create_context_engine
from ctx.store import SCHEMA_VERSION, SQLiteStore

app = typer.Typer(
    name="ctx",
    help="Local, offline-first, provenance-rich exact Markdown context infrastructure.",
    no_args_is_help=True,
)
model_app = typer.Typer(help="Explicit local embedding-model lifecycle.", no_args_is_help=True)
checkpoint_app = typer.Typer(help="Document-aware checkpoint operations.", no_args_is_help=True)
app.add_typer(model_app, name="model")
app.add_typer(checkpoint_app, name="checkpoint")
console = Console(stderr=True)


def version_callback(value: bool) -> None:
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
    """Original Markdown remains authoritative; all normal operations are local-only."""


def _root(path: Path) -> Path:
    return path.resolve(strict=True)


def _authority(value: str) -> Authority:
    try:
        return Authority[value.strip().upper()]
    except KeyError as error:
        choices = ", ".join(item.name.lower() for item in Authority)
        raise typer.BadParameter(f"authority must be one of: {choices}") from error


def _authorities(values: list[str] | None) -> set[Authority] | None:
    return {_authority(value) for value in values} if values else None


def _data(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list | tuple):
        return [_data(item) for item in value]
    return value


def _emit(value: Any, json_output: bool) -> None:
    data = _data(value)
    encoded = json.dumps(data, ensure_ascii=False, indent=None if json_output else 2)
    if json_output:
        typer.echo(encoded)
    else:
        # stdout remains machine result output; diagnostics/warnings use stderr.
        typer.echo(encoded)


def _fail(error: Exception, *, json_output: bool = False) -> None:
    if json_output and hasattr(error, "as_dict"):
        typer.echo(json.dumps(error.as_dict(), ensure_ascii=False))
    else:
        typer.echo(f"error: {error}", err=True)
    raise typer.Exit(code=2)


def _engine(
    root: Path,
    *,
    no_embeddings: bool = False,
    model_dir: Path | None = None,
) -> ContextEngine:
    return create_context_engine(
        _root(root), embeddings_enabled=not no_embeddings, model_dir=model_dir
    )


@app.command()
def init(
    root: Annotated[Path, typer.Argument(help="Workspace/repository root.")] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Initialize private config and SQLite V2 schema."""
    try:
        resolved = _root(root)
        config = initialize_workspace(resolved)
        with SQLiteStore(database_path(resolved)) as store:
            _emit(
                {
                    "workspace": str(resolved),
                    "config": str(resolved / ".ctx" / "config.toml"),
                    "index_generation": store.index_generation(),
                    "schema_version": SCHEMA_VERSION,
                    "documents": len(config.documents),
                },
                json_output,
            )
    except (ConfigError, OSError, RuntimeError) as error:
        _fail(error, json_output=json_output)


@app.command()
def add(
    path: Annotated[str, typer.Argument(help="Relative Markdown path.")],
    authority: Annotated[str, typer.Option()] = "reference",
    priority: Annotated[int, typer.Option(min=-100, max=100)] = 0,
    root: Annotated[Path, typer.Option()] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Add or update a source configuration."""
    try:
        config = add_document_config(_root(root), path, _authority(authority), priority)
        normalized = Path(path).as_posix()
        document = next(item for item in config.documents if item.path == normalized)
        _emit(document, json_output)
    except (ConfigError, OSError, StopIteration, typer.BadParameter, ValueError) as error:
        _fail(error, json_output=json_output)


@app.command()
def remove(
    path: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option()] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Remove configuration/index data; source is untouched."""
    try:
        resolved = _root(root)
        remove_document_config(resolved, path)
        with create_context_engine(resolved, embeddings_enabled=False) as engine:
            stats = engine.sync_workspace()
        _emit(
            {
                "path": path,
                "removed_from_index": stats.documents_removed > 0,
                "index_generation": stats.index_generation,
            },
            json_output,
        )
    except (ConfigError, OSError, RuntimeError) as error:
        _fail(error, json_output=json_output)


def _run_sync(root: Path, json_output: bool, no_embeddings: bool, model_dir: Path | None) -> None:
    try:
        with _engine(root, no_embeddings=no_embeddings, model_dir=model_dir) as engine:
            result = engine.sync_workspace()
            data = result.model_dump(mode="json")
            data["active_channels"] = engine.active_channels
            _emit(data, json_output)
    except (ConfigError, OSError, RuntimeError, ValueError) as error:
        _fail(error, json_output=json_output)


@app.command("index")
def index_command(
    root: Annotated[Path, typer.Option()] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    no_embeddings: Annotated[
        bool, typer.Option(help="Use structural+lexical channels only.")
    ] = False,
    model_dir: Annotated[Path | None, typer.Option(help="Override installed model cache.")] = None,
) -> None:
    """Atomically index all configured sources; never downloads a model."""
    _run_sync(root, json_output, no_embeddings, model_dir)


@app.command()
def sync(
    root: Annotated[Path, typer.Option()] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    no_embeddings: Annotated[bool, typer.Option()] = False,
    model_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Synchronize one atomic logical generation; never downloads a model."""
    _run_sync(root, json_output, no_embeddings, model_dir)


@app.command()
def status(
    root: Annotated[Path, typer.Option()] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    no_embeddings: Annotated[bool, typer.Option()] = False,
    model_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    try:
        with _engine(root, no_embeddings=no_embeddings, model_dir=model_dir) as engine:
            _emit(engine.status(), json_output)
    except (ConfigError, OSError, RuntimeError) as error:
        _fail(error, json_output=json_output)


@app.command("docs")
def docs_command(
    root: Annotated[Path, typer.Option()] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        with _engine(root, no_embeddings=True) as engine:
            _emit(engine.store.list_documents(), json_output)
    except (ConfigError, OSError, RuntimeError) as error:
        _fail(error, json_output=json_output)


@app.command()
def outline(
    path: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option()] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Return metadata-only document outline."""
    try:
        with _engine(root, no_embeddings=True) as engine:
            _emit(engine.document_outline(path), json_output)
    except (ConfigError, OSError, KeyError, RuntimeError) as error:
        _fail(error, json_output=json_output)


@app.command()
def section(
    section_id: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option()] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
    auto_sync: Annotated[bool, typer.Option()] = False,
) -> None:
    try:
        with _engine(root, no_embeddings=True) as engine:
            _emit(engine.get_section(section_id, auto_sync=auto_sync), json_output)
    except (ConfigError, OSError, KeyError, RuntimeError) as error:
        _fail(error, json_output=json_output)


@app.command()
def lines(
    path: Annotated[str, typer.Argument()],
    start_line: Annotated[int, typer.Argument(min=1)],
    end_line: Annotated[int, typer.Argument(min=1)],
    root: Annotated[Path, typer.Option()] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Return exact source line ranges via the shared service."""
    try:
        with _engine(root, no_embeddings=True) as engine:
            _emit(engine.get_lines(path, start_line, end_line), json_output)
    except (ConfigError, OSError, KeyError, RuntimeError, ValueError) as error:
        _fail(error, json_output=json_output)


class SearchFilters(TypedDict):
    documents: set[str] | None
    authority_floor: Authority | None
    authorities: set[Authority] | None
    exclude_documents: set[str] | None
    heading_prefix: tuple[str, ...] | None
    scope: str | None


def _search_filters(
    document: list[str] | None,
    authority_floor: str | None,
    authority: list[str] | None,
    exclude_document: list[str] | None,
    heading_prefix: list[str] | None,
    scope: str | None,
) -> SearchFilters:
    return {
        "documents": set(document) if document else None,
        "authority_floor": _authority(authority_floor) if authority_floor else None,
        "authorities": _authorities(authority),
        "exclude_documents": set(exclude_document) if exclude_document else None,
        "heading_prefix": tuple(heading_prefix) if heading_prefix else None,
        "scope": scope,
    }


@app.command()
def search(
    query: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option()] = Path("."),
    limit: Annotated[int, typer.Option(min=1, max=1_000)] = 10,
    exact: Annotated[bool, typer.Option()] = False,
    lexical: Annotated[bool, typer.Option()] = False,
    include_full_section: Annotated[bool, typer.Option()] = False,
    document: Annotated[list[str] | None, typer.Option("--document")] = None,
    authority_floor: Annotated[str | None, typer.Option()] = None,
    authority: Annotated[list[str] | None, typer.Option("--authority")] = None,
    exclude_document: Annotated[list[str] | None, typer.Option("--exclude-document")] = None,
    heading_prefix: Annotated[list[str] | None, typer.Option("--heading-prefix")] = None,
    scope: Annotated[str | None, typer.Option()] = None,
    no_embeddings: Annotated[bool, typer.Option()] = False,
    model_dir: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Search compact exact excerpts by default; full sections require explicit opt-in."""
    try:
        filters = _search_filters(
            document, authority_floor, authority, exclude_document, heading_prefix, scope
        )
        with _engine(
            root,
            no_embeddings=no_embeddings or exact or lexical,
            model_dir=model_dir,
        ) as engine:
            if exact:
                result = engine.search_exact(
                    query, limit=limit, include_full_section=include_full_section, **filters
                )
            elif lexical:
                result = engine.search_lexical(
                    query, limit=limit, include_full_section=include_full_section, **filters
                )
            else:
                result = engine.search(
                    query, limit=limit, include_full_section=include_full_section, **filters
                )
            _emit(result, json_output)
    except (ConfigError, OSError, KeyError, RuntimeError, ValueError) as error:
        _fail(error, json_output=json_output)


@app.command("find")
def find_symbol_command(
    symbol: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option()] = Path("."),
    limit: Annotated[int, typer.Option(min=1, max=1_000)] = 20,
    document: Annotated[list[str] | None, typer.Option("--document")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        with _engine(root, no_embeddings=True) as engine:
            _emit(
                engine.find_symbol(
                    symbol, limit=limit, documents=set(document) if document else None
                ),
                json_output,
            )
    except (ConfigError, OSError, RuntimeError, ValueError) as error:
        _fail(error, json_output=json_output)


@app.command()
def refs(
    section_id: Annotated[str, typer.Argument()],
    incoming: Annotated[bool, typer.Option()] = False,
    root: Annotated[Path, typer.Option()] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        with _engine(root, no_embeddings=True) as engine:
            _emit(engine.get_references(section_id, incoming=incoming), json_output)
    except (ConfigError, OSError, RuntimeError, KeyError) as error:
        _fail(error, json_output=json_output)


@app.command()
def deps(
    section_id: Annotated[str, typer.Argument()],
    root: Annotated[Path, typer.Option()] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        with _engine(root, no_embeddings=True) as engine:
            _emit(engine.get_dependencies(section_id), json_output)
    except (ConfigError, OSError, RuntimeError, KeyError) as error:
        _fail(error, json_output=json_output)


@app.command()
def pack(
    task: Annotated[str, typer.Argument()],
    token_budget: Annotated[int, typer.Option(min=1, max=2_000_000)] = 7_000,
    document: Annotated[list[str] | None, typer.Option("--document")] = None,
    authority_floor: Annotated[str | None, typer.Option()] = None,
    authority: Annotated[list[str] | None, typer.Option("--authority")] = None,
    exclude_document: Annotated[list[str] | None, typer.Option("--exclude-document")] = None,
    heading_prefix: Annotated[list[str] | None, typer.Option("--heading-prefix")] = None,
    scope: Annotated[str | None, typer.Option()] = None,
    root: Annotated[Path, typer.Option()] = Path("."),
    no_embeddings: Annotated[bool, typer.Option()] = False,
    model_dir: Annotated[Path | None, typer.Option()] = None,
    allow_required_budget_expansion: Annotated[
        bool, typer.Option("--allow-required-budget-expansion")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        filters = _search_filters(
            document, authority_floor, authority, exclude_document, heading_prefix, scope
        )
        with _engine(root, no_embeddings=no_embeddings, model_dir=model_dir) as engine:
            _emit(
                engine.get_context_pack(
                    task,
                    token_budget,
                    allow_required_budget_expansion=allow_required_budget_expansion,
                    **filters,
                ),
                json_output,
            )
    except (ConfigError, OSError, RuntimeError, ValueError, KeyError) as error:
        _fail(error, json_output=json_output)


@checkpoint_app.command("show")
def checkpoint_show(
    checkpoint_id: Annotated[str, typer.Argument()],
    document: Annotated[str | None, typer.Option("--document")] = None,
    root: Annotated[Path, typer.Option()] = Path("."),
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        with _engine(root, no_embeddings=True) as engine:
            _emit(engine.get_checkpoint(checkpoint_id, document=document), json_output)
    except (ConfigError, OSError, RuntimeError, ValueError, KeyError) as error:
        _fail(error, json_output=json_output)


@checkpoint_app.command("context")
def checkpoint_context(
    checkpoint_id: Annotated[str, typer.Argument()],
    document: Annotated[str | None, typer.Option("--document")] = None,
    budget: Annotated[int, typer.Option("--budget", min=1, max=2_000_000)] = 7_000,
    root: Annotated[Path, typer.Option()] = Path("."),
    no_embeddings: Annotated[bool, typer.Option()] = False,
    model_dir: Annotated[Path | None, typer.Option()] = None,
    allow_required_budget_expansion: Annotated[
        bool, typer.Option("--allow-required-budget-expansion")
    ] = False,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        with _engine(root, no_embeddings=no_embeddings, model_dir=model_dir) as engine:
            _emit(
                engine.get_checkpoint_context(
                    checkpoint_id,
                    document=document,
                    token_budget=budget,
                    allow_required_budget_expansion=allow_required_budget_expansion,
                ),
                json_output,
            )
    except (ConfigError, OSError, RuntimeError, ValueError, KeyError) as error:
        _fail(error, json_output=json_output)


@model_app.command("status")
def model_status(
    root: Annotated[Path, typer.Option(help="Workspace root for configured identity.")] = Path("."),
    model_dir: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        config = load_config(_root(root))
        directory = installed_model_path(
            config.embedding.model, config.embedding.revision, root=model_dir
        )
        try:
            manifest = verify_model(directory)
            result: dict[str, Any] = {
                "status": "INSTALLED_VERIFIED",
                "path": str(directory),
                "manifest": manifest.model_dump(mode="json"),
            }
        except RuntimeError as error:
            result = {"status": "MISSING_OR_INVALID", "path": str(directory), "reason": str(error)}
        _emit(result, json_output)
    except (ConfigError, OSError, RuntimeError) as error:
        _fail(error, json_output=json_output)


@model_app.command("download")
def model_download_command(
    model: Annotated[str, typer.Option()] = DEFAULT_MODEL,
    revision: Annotated[str, typer.Option()] = DEFAULT_REVISION,
    model_dir: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Explicitly permit network access to download and checksum a local model."""
    try:
        manifest = download_model(model_name=model, revision=revision, root=model_dir)
        _emit(manifest, json_output)
    except (OSError, RuntimeError, ValueError) as error:
        _fail(error, json_output=json_output)


@model_app.command("install")
def model_install_command(
    source: Annotated[Path, typer.Argument()],
    model: Annotated[str, typer.Option()] = DEFAULT_MODEL,
    revision: Annotated[str, typer.Option()] = DEFAULT_REVISION,
    dimensions: Annotated[int, typer.Option(min=1)] = 384,
    model_dir: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Install existing local artifacts without network access."""
    try:
        manifest = install_model(
            source,
            model_name=model,
            revision=revision,
            root=model_dir,
            dimensions=dimensions,
        )
        _emit(manifest, json_output)
    except (OSError, RuntimeError, ValueError) as error:
        _fail(error, json_output=json_output)


@model_app.command("verify")
def model_verify_command(
    model: Annotated[str, typer.Option()] = DEFAULT_MODEL,
    revision: Annotated[str, typer.Option()] = DEFAULT_REVISION,
    model_dir: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    try:
        _emit(verify_model(installed_model_path(model, revision, root=model_dir)), json_output)
    except (OSError, RuntimeError, ValueError) as error:
        _fail(error, json_output=json_output)


@app.command()
def doctor(
    root: Annotated[Path, typer.Option()] = Path("."),
    offline: Annotated[bool, typer.Option(help="Require complete offline readiness.")] = False,
    model_dir: Annotated[Path | None, typer.Option()] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Read-only diagnostics; performs no sync, migration download, or network request."""
    try:
        resolved = _root(root)
        config = load_config(resolved)
        with create_context_engine(
            resolved, embeddings_enabled=False, model_dir=model_dir, read_only=True
        ) as engine:
            status_value = engine.status()
            compile_options = [
                str(row[0])
                for row in engine.store.connection.execute("PRAGMA compile_options").fetchall()
            ]
            fts5 = any("ENABLE_FTS5" in item for item in compile_options)
            model_path = installed_model_path(
                config.embedding.model, config.embedding.revision, root=model_dir
            )
            try:
                manifest = verify_model(model_path)
                model_value: dict[str, Any] = {
                    "verified": True,
                    "identity": manifest.identity,
                    "path": str(model_path),
                }
            except RuntimeError as error:
                model_value = {"verified": False, "path": str(model_path), "reason": str(error)}
            result = {
                "ctx_version": __version__,
                "python": sys.version,
                "platform": platform.platform(),
                "sqlite": sqlite3.sqlite_version,
                "sqlite_threadsafe": sqlite3.threadsafety,
                "journal_mode": engine.store.connection.execute("PRAGMA journal_mode").fetchone()[
                    0
                ],
                "busy_timeout_ms": engine.store.connection.execute(
                    "PRAGMA busy_timeout"
                ).fetchone()[0],
                "fts5": fts5,
                "mcp_sdk": importlib.metadata.version("mcp"),
                "config_version": config.version,
                "schema_version": SCHEMA_VERSION,
                "model_cache": str(model_root(model_dir)),
                "model": model_value,
                "status": status_value.model_dump(mode="json"),
                "offline_ready": status_value.category.value == "CLEAN"
                and (config.embedding.backend == "disabled" or model_value["verified"]),
                "network_attempted": False,
            }
            _emit(result, json_output)
            if offline and not result["offline_ready"]:
                raise typer.Exit(code=2)
    except typer.Exit:
        raise
    except (ConfigError, OSError, RuntimeError, ValueError) as error:
        _fail(error, json_output=json_output)


@app.command()
def mcp(
    root: Annotated[Path, typer.Option()] = Path("."),
    no_embeddings: Annotated[bool, typer.Option()] = False,
    model_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Serve local stdio MCP only; no listener is created."""
    try:
        run_mcp(_root(root), embeddings_enabled=not no_embeddings, model_dir=model_dir)
    except (ConfigError, OSError, RuntimeError, ValueError) as error:
        _fail(error)


if __name__ == "__main__":  # pragma: no cover
    app()
