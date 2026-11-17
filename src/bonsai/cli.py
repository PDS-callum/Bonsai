"""The ``bonsai`` command-line interface."""

from pathlib import Path
from typing import Annotated

import typer

from bonsai import __version__
from bonsai.bench import BenchError, format_report, load_runs
from bonsai.fetch import FetchError, destination, fetch, load_manifest
from bonsai.job_cli import job_app
from bonsai.pot import GIB, Pot, PotError, load_pot
from bonsai.weights import weight_bytes

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
app.add_typer(job_app, name="job")

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
            help="The tree's GGUF file.",
            exists=True,
            dir_okay=False,
        ),
    ],
) -> None:
    """Estimate the fastest generation a tree could reach in a pot."""
    pot = _load(pot_file)
    try:
        counted = weight_bytes(weights)
    except (OSError, ValueError) as error:
        typer.echo(f"error: cannot read {weights} as GGUF: {error}", err=True)
        raise typer.Exit(1) from error
    ceiling = pot.ceiling_tokens_per_second(counted.per_token)
    typer.echo(f"{weights.name}")
    typer.echo(f"  weights         {counted.total / 1e6:.0f} MB")
    typer.echo(f"  read per token  {counted.per_token / 1e6:.0f} MB")
    typer.echo(
        f"  ceiling         {ceiling:.1f} tokens/s at {pot.read_bandwidth_gbs:.1f} GB/s"
    )
    typer.echo(
        f"  weights fit     {'yes' if pot.fits(counted.total) else 'no'} "
        f"(budget {pot.memory_budget_bytes / GIB:.2f} GiB; "
        "the runtime needs more on top)"
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
    trees: Annotated[
        Path | None,
        typer.Option(
            help="Directory holding the measured GGUFs, to count the bytes each "
            "reads per token. Without it, whole model sizes are used.",
            exists=True,
            file_okay=False,
        ),
    ] = None,
) -> None:
    """Tabulate tree speeds against the pot's bytes-per-token ceiling."""
    pot = _load(pot_file)
    try:
        runs = [run for path in results for run in load_runs(path)]
    except BenchError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error
    per_token: dict[str, int] = {}
    if trees:
        for run in runs:
            path = trees / run.model_file
            if run.model_file not in per_token and path.is_file():
                per_token[run.model_file] = weight_bytes(path).per_token
    typer.echo(format_report(pot, runs, per_token), nl=False)


@app.command("fetch")
def fetch_command(
    manifest: Annotated[
        Path,
        typer.Argument(
            help="Manifest of pinned files, e.g. configs/data/quality-v1.toml.",
            exists=True,
            dir_okay=False,
        ),
    ],
    only: Annotated[
        list[str] | None,
        typer.Option("--only", help="Fetch just this file (repeatable)."),
    ] = None,
) -> None:
    """Download a manifest's files, checking each one's SHA-256."""
    try:
        loaded = load_manifest(manifest)
        fetch(loaded, only=only or ())
    except FetchError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error
    typer.echo(f"files in {destination(loaded)}")
