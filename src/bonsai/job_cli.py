"""``bonsai job``: run measuring jobs in the pot and collect their results."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from fnmatch import fnmatch
from pathlib import Path
from typing import Annotated

import typer

from bonsai.jobs import JobError, PotConnection
from bonsai.pot import GIB, Pot, PotError, load_pot
from bonsai.weights import weight_bytes

job_app = typer.Typer(
    help="Run measuring jobs in the pot, detached, and collect their results.",
    no_args_is_help=True,
)

make_pot: Callable[[str], PotConnection] = PotConnection
"""Builds the connection to the pot; replaced in tests."""

DEFAULT_SCRIPTS = Path("scripts/pi")
DEFAULT_RESULTS = Path("results/jobs")
DEFAULT_PROFILE = Path("configs/pots/pi5-4gb.toml")

Host = Annotated[
    str,
    typer.Option(
        "--host",
        envvar="BONSAI_PI",
        help="The pot as user@address (or set BONSAI_PI).",
        show_envvar=True,
    ),
]
Scripts = Annotated[
    Path,
    typer.Option(
        "--scripts",
        help="Local directory of the pot's scripts, copied over before each job.",
        file_okay=False,
    ),
]
JobId = Annotated[str, typer.Argument(help="Job id, as printed by submit.")]


def _fail(error: JobError) -> typer.Exit:
    typer.echo(f"error: {error}", err=True)
    return typer.Exit(1)


def _start(
    pot: PotConnection,
    scripts: Path,
    name: str,
    command: list[str],
) -> None:
    """Sync the scripts, so the job runs the current versions, then submit it."""
    pot.sync_scripts(scripts)
    job_id = pot.submit(name, command)
    typer.echo(job_id)
    typer.echo(f"follow with: bonsai job log {job_id}", err=True)


@job_app.command("sync")
def sync(host: Host, scripts: Scripts = DEFAULT_SCRIPTS) -> None:
    """Copy the pot's scripts over."""
    try:
        make_pot(host).sync_scripts(scripts)
    except JobError as error:
        raise _fail(error) from error
    typer.echo(f"synced {scripts} to {host}")


@job_app.command("profile")
def profile(host: Host, scripts: Scripts = DEFAULT_SCRIPTS) -> None:
    """Profile the pot: bandwidth, memory, storage and thermals."""
    pot = make_pot(host)
    try:
        _start(
            pot, scripts, "profile-pot", [f"{pot.scripts}/profile-pot.sh", "results"]
        )
    except JobError as error:
        raise _fail(error) from error


def check_trees(profile: Pot, trees: Sequence[Path], reference: str) -> list[str]:
    """Problems that should stop these trees being sent to the pot.

    Each tree's weights must fit the memory budget (the runtime needs more on
    top, but a tree whose weights alone are over can never fit); together they
    must fit the storage budget; and exactly one must match ``reference``.
    """
    problems = []
    for tree in trees:
        size = weight_bytes(tree).total
        if not profile.fits(size):
            problems.append(
                f"{tree.name}: weights of {size / GIB:.2f} GiB exceed the "
                f"{profile.memory_budget_bytes / GIB:.2f} GiB memory budget"
            )
    stored = sum(tree.stat().st_size for tree in trees)
    if stored > profile.storage_budget_bytes:
        problems.append(
            f"the trees take {stored / GIB:.2f} GiB; the storage budget is "
            f"{profile.storage_budget_bytes / GIB:.2f} GiB"
        )
    matches = [tree.name for tree in trees if fnmatch(tree.name, f"{reference}.gguf")]
    if len(matches) != 1:
        problems.append(
            f"--reference {reference!r} must match exactly one tree; "
            f"it matches {len(matches)}"
        )
    return problems


@job_app.command("bench")
def bench(
    host: Host,
    trees: Annotated[
        list[Path],
        typer.Argument(help="GGUF files to measure.", exists=True, dir_okay=False),
    ],
    profile: Annotated[
        Path,
        typer.Option(
            "--pot",
            help="The pot's profile, for its budgets, threads and runtime settings.",
            exists=True,
            dir_okay=False,
        ),
    ] = DEFAULT_PROFILE,
    reference: Annotated[
        str,
        typer.Option(
            help="File-name pattern for the tree used in the thread and sustained runs."
        ),
    ] = "*q4_k_m*",
    name: Annotated[str, typer.Option(help="Job name prefix.")] = "bench",
    scripts: Scripts = DEFAULT_SCRIPTS,
) -> None:
    """Measure trees' speed, memory and heat. They are uploaded with the job.

    Trees run with the threads and repacking setting in the pot's profile, and
    the job starts only if the pot has the profile's memory budget free.
    """
    try:
        budgets = load_pot(profile)
    except PotError as error:
        typer.echo(f"error: {error}", err=True)
        raise typer.Exit(1) from error
    problems = check_trees(budgets, trees, reference)
    if problems:
        for problem in problems:
            typer.echo(f"error: {problem}", err=True)
        raise typer.Exit(1)

    pot = make_pot(host)
    command = [
        "env",
        f"THREADS={budgets.threads}",
        f"REPACK={int(budgets.repack)}",
        f"REFERENCE={reference}",
        f"{pot.scripts}/bench-trees.sh",
        "inputs",
        "results",
    ]
    try:
        pot.sync_scripts(scripts)
        job_id = pot.submit(
            name,
            command,
            inputs=trees,
            needs_free_bytes=budgets.memory_budget_bytes,
        )
    except JobError as error:
        raise _fail(error) from error
    typer.echo(job_id)
    typer.echo(f"follow with: bonsai job log {job_id}", err=True)


@job_app.command("status")
def status(
    host: Host,
    job_id: Annotated[
        str | None, typer.Argument(help="Job id; omit to list every job.")
    ] = None,
) -> None:
    """Show whether jobs are running, succeeded or failed."""
    pot = make_pot(host)
    try:
        job_ids = [job_id] if job_id else pot.list_jobs()
        for one in job_ids:
            state = pot.status(one)
            code = "" if state.exit_code is None else f" (exit {state.exit_code})"
            typer.echo(f"{one}  {state.state}{code}")
    except JobError as error:
        raise _fail(error) from error


@job_app.command("log")
def log(
    host: Host,
    job_id: JobId,
    lines: Annotated[int, typer.Option("--lines", "-n", min=1)] = 20,
) -> None:
    """Show the end of a job's log."""
    try:
        typer.echo(make_pot(host).log(job_id, lines), nl=False)
    except JobError as error:
        raise _fail(error) from error


@job_app.command("collect")
def collect(
    host: Host,
    job_id: JobId,
    to: Annotated[
        Path, typer.Option(help="Local directory for results.", file_okay=False)
    ] = DEFAULT_RESULTS,
    clean: Annotated[
        bool,
        typer.Option(help="Delete the job's uploaded inputs from the pot afterwards."),
    ] = False,
) -> None:
    """Copy a finished job's results and log from the pot."""
    try:
        target = make_pot(host).collect(job_id, to, clean=clean)
    except JobError as error:
        raise _fail(error) from error
    typer.echo(str(target))
