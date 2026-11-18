"""``bonsai journal``: build and query the journal."""

from contextlib import closing
from pathlib import Path
from typing import Annotated

import typer

from bonsai.fetch import bonsai_home
from bonsai.journal import (
    QUALITY_METRICS,
    JournalError,
    build,
    connect,
    find_tree,
    format_frontier,
    frontier,
    lineage,
    trees_summary,
)
from bonsai.plot import Panel, render
from bonsai.pot import PotError, load_pot

journal_app = typer.Typer(
    help="The journal: every tree, its lineage and its measurements."
)

Db = Annotated[
    Path | None,
    typer.Option(help="The journal database. Default: ~/bonsai/journal.sqlite."),
]


def _db(db: Path | None) -> Path:
    return db or bonsai_home() / "journal.sqlite"


def _fail(error: Exception) -> typer.Exit:
    typer.echo(f"error: {error}", err=True)
    return typer.Exit(1)


@journal_app.command("build")
def build_command(
    db: Db = None,
    root: Annotated[
        Path, typer.Option(help="The repository root.", file_okay=False, exists=True)
    ] = Path(),
) -> None:
    """Rebuild the journal from the repository's records."""
    try:
        report = build(root, _db(db))
    except JournalError as error:
        raise _fail(error) from error
    typer.echo(
        f"{report.seeds} seeds, {report.trees} trees, "
        f"{report.measurements} measurements in {_db(db)}"
    )
    for file in report.unmatched or []:
        typer.echo(f"  no tree matches {file}")


@journal_app.command("trees")
def trees_command(db: Db = None) -> None:
    """List every tree and how many measurements it has."""
    try:
        connection = connect(_db(db))
    except JournalError as error:
        raise _fail(error) from error
    typer.echo("| Tree | Seed | Made by | Size | Quality | Speed |")
    typer.echo("|---|---|---|---|---|---|")
    with closing(connection):
        for row in trees_summary(connection):
            size = f"{row['bytes'] / 1e6:.0f} MB" if row["bytes"] else "-"
            typer.echo(
                f"| {row['file']} | {row['seed']} | {row['operation']} | {size} "
                f"| {row['quality'] or 0} | {row['speed'] or 0} |"
            )


@journal_app.command("lineage")
def lineage_command(
    tree: Annotated[str, typer.Argument(help="File name or SHA-256 prefix.")],
    db: Db = None,
) -> None:
    """Trace a tree back to its original seed."""
    try:
        with closing(connect(_db(db))) as connection:
            steps = lineage(connection, find_tree(connection, tree)["sha256"])
    except JournalError as error:
        raise _fail(error) from error
    for depth, step in enumerate(steps):
        typer.echo(f"{'  ' * depth}{'<- ' if depth else ''}{step}")


@journal_app.command("frontier")
def frontier_command(
    metric: Annotated[
        str, typer.Option(help=f"Quality metric: {', '.join(QUALITY_METRICS)}.")
    ] = "gsm8k_pct",
    pot: Annotated[
        Path, typer.Option(help="Pot profile whose settings the speeds must match.")
    ] = Path("configs/pots/pi5-4gb.toml"),
    db: Db = None,
) -> None:
    """Quality against speed in the pot, marking the trees on the frontier."""
    try:
        profile = load_pot(pot)
        with closing(connect(_db(db))) as connection:
            points = frontier(
                connection, metric, threads=profile.threads, repack=profile.repack
            )
    except (JournalError, PotError, OSError) as error:
        raise _fail(error) from error
    typer.echo(format_frontier(points, metric), nl=False)


@journal_app.command("plot")
def plot_command(
    out: Annotated[Path, typer.Option(help="The SVG to write.")] = Path(
        "results/frontier/frontier.svg"
    ),
    pot: Annotated[
        Path, typer.Option(help="Pot profile whose settings the speeds must match.")
    ] = Path("configs/pots/pi5-4gb.toml"),
    db: Db = None,
) -> None:
    """Plot maths and science scores against speed in the pot."""
    try:
        profile = load_pot(pot)
        with closing(connect(_db(db))) as connection:
            maths, science = (
                frontier(
                    connection, metric, threads=profile.threads, repack=profile.repack
                )
                for metric in ("gsm8k_pct", "arc_pct")
            )
    except (JournalError, PotError, OSError) as error:
        raise _fail(error) from error
    svg = render(
        [
            Panel("Maths: GSM8K, 400 problems (95% interval)", maths, 0, 50, 10),
            Panel("Science: ARC, 1,172 questions (± 1 s.e.)", science, 20, 60, 10),
        ],
        title=f"Quality against speed in the pot ({profile.name}, "
        f"{profile.threads} threads)",
        x_label="Generation speed in the pot (tokens/s)",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(svg, encoding="utf-8", newline="\n")
    typer.echo(f"{len(maths)} trees plotted in {out}")
