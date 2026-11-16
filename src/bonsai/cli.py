"""The ``bonsai`` command-line interface."""

from pathlib import Path
from typing import Annotated

import typer

from bonsai import __version__
from bonsai.bench import BenchError, format_report, load_runs
from bonsai.pot import GIB, Pot, PotError, load_pot

app = typer.Typer(
    name="bonsai",
    help="Cultivate small language models for small, CPU-bound hardware.",
    no_args_is_help=True,
    add_completion=False,
)
pot_app = typer.Typer(help="Inspect pots: the hardware budgets trees must fit.")
app.add_typer(pot_app, name="pot")
bench_app = typer.Typer(help="Read benchmark results from the pot.")
app.add_typer(bench_app, name="bench")

PotFile = Annotated[
    Path,
    typer.Argument(
        help="Pot profile (TOML), e.g. configs/pots/pi5-4gb.toml.",
        exists=True,
        dir_okay=False,
    ),
]


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


def _load(path: Path) -> Pot:
    try:
        return load_pot(path)
    except PotError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error


@pot_app.command("show")
def pot_show(pot_file: PotFile) -> None:
    """Show a pot's budgets."""
    pot = _load(pot_file)
    typer.echo(f"{pot.name}: {pot.description}")
    typer.echo(f"  read bandwidth  {pot.read_bandwidth_gbs:.1f} GB/s")
    typer.echo(f"  memory budget   {pot.memory_budget_bytes / GIB:.2f} GiB")
    typer.echo(f"  storage budget  {pot.storage_budget_bytes / GIB:.1f} GiB")
    typer.echo(f"  threads         {pot.threads}")


@pot_app.command("ceiling")
def pot_ceiling(
    pot_file: PotFile,
    weights: Annotated[
        Path,
        typer.Argument(
            help="Model file (e.g. a GGUF) whose size approximates the bytes "
            "read per token.",
            exists=True,
            dir_okay=False,
        ),
    ],
) -> None:
    """Estimate the fastest generation a model could reach in a pot."""
    pot = _load(pot_file)
    size = weights.stat().st_size
    ceiling = pot.ceiling_tokens_per_second(size)
    typer.echo(f"{weights.name}: {size / 1e6:.0f} MB")
    typer.echo(
        f"  ceiling   {ceiling:.1f} tokens/s at {pot.read_bandwidth_gbs:.1f} GB/s"
    )
    typer.echo(
        f"  fits      {'yes' if pot.fits(size) else 'no'} "
        f"(budget {pot.memory_budget_bytes / GIB:.2f} GiB)"
    )


@bench_app.command("report")
def bench_report(
    pot_file: PotFile,
    results: Annotated[
        list[Path],
        typer.Argument(
            help="llama-bench JSONL output (-o jsonl).", exists=True, dir_okay=False
        ),
    ],
) -> None:
    """Tabulate tree speeds against the pot's bytes-per-token ceiling."""
    pot = _load(pot_file)
    try:
        runs = [run for path in results for run in load_runs(path)]
    except BenchError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(format_report(pot, runs), nl=False)
