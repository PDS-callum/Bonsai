"""The ``bonsai`` command-line interface."""

from typing import Annotated

import typer

from bonsai import __version__

app = typer.Typer(
    name="bonsai",
    help="Cultivate small language models for small, CPU-bound hardware.",
    no_args_is_help=True,
    add_completion=False,
)


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"bonsai {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            help="Show the version and exit.",
            callback=_print_version,
            is_eager=True,
        ),
    ] = False,
) -> None:
    """Cultivate small language models for small, CPU-bound hardware."""
