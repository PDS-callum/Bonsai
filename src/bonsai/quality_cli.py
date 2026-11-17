"""``bonsai quality``: measure trees' quality on the potting bench."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Annotated

import typer

from bonsai.fetch import FetchError, destination, load_manifest
from bonsai.llama import LlamaError, default_bin_dir
from bonsai.quality import (
    Settings,
    format_report,
    load_results,
    measure,
    save_base_logits,
)

quality_app = typer.Typer(
    help="Measure trees' quality on the potting bench.", no_args_is_help=True
)

DEFAULT_DATA = Path("configs/data/quality-v1.toml")


@quality_app.command("run")
def run(
    trees: Annotated[
        list[Path],
        typer.Argument(help="GGUF files to measure.", exists=True, dir_okay=False),
    ],
    out: Annotated[
        Path,
        typer.Option(help="Directory for results; each tree gets a folder."),
    ],
    base: Annotated[
        Path | None,
        typer.Option(
            help="Tree to measure KL divergence from, usually the seed at full "
            "precision.",
            exists=True,
            dir_okay=False,
        ),
    ] = None,
    data: Annotated[
        Path,
        typer.Option(
            help="Manifest of the measuring data, fetched with `bonsai fetch`.",
            exists=True,
            dir_okay=False,
        ),
    ] = DEFAULT_DATA,
    gsm8k_size: Annotated[
        int, typer.Option(help="GSM8K problems per tree (fixed subset).", min=1)
    ] = 400,
    threads: Annotated[
        int | None, typer.Option(help="CPU threads (default: all).", min=1)
    ] = None,
) -> None:
    """Measure perplexity, KL divergence, ARC and GSM8K for each tree."""
    try:
        data_dir = destination(load_manifest(data))
    except FetchError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error
    if not data_dir.is_dir():
        typer.echo(f"error: no data in {data_dir}; run: bonsai fetch {data}", err=True)
        raise typer.Exit(1)

    settings = Settings(
        data_dir=data_dir, bin_dir=default_bin_dir(), gsm8k_size=gsm8k_size
    )
    if threads:
        settings = replace(settings, threads=threads)
    try:
        base_logits = None
        if base:
            typer.echo(f"saving {base.name}'s predictions for KL divergence")
            base_logits = save_base_logits(base, settings)
        for tree in trees:
            measure(tree, out, settings, base=base, base_logits=base_logits)
    except LlamaError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(format_report(load_results(out)), nl=False)


@quality_app.command("report")
def report(
    results: Annotated[
        Path,
        typer.Argument(
            help="Directory of results from `bonsai quality run`.",
            exists=True,
            file_okay=False,
        ),
    ],
) -> None:
    """Tabulate measured quality, smallest tree first."""
    found = load_results(results)
    if not found:
        typer.echo(f"error: no quality.json under {results}", err=True)
        raise typer.Exit(1)
    typer.echo(format_report(found), nl=False)
