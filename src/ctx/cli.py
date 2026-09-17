"""Command-line entry point for ctx."""

from typing import Annotated

import typer

from ctx import __version__

app = typer.Typer(
    name="ctx",
    help="Index local Markdown and serve exact, provenance-rich source context.",
    no_args_is_help=True,
)


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


if __name__ == "__main__":  # pragma: no cover
    app()
