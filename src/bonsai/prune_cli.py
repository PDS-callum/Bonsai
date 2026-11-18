"""``bonsai prune`` and ``bonsai check``: cut layers, then check what was planted."""

import json
from pathlib import Path
from typing import Annotated, Any

import typer

from bonsai.check import (
    CheckError,
    compare,
    load_reference,
    reference_answers,
)
from bonsai.fetch import FetchError, bonsai_home, destination, load_manifest
from bonsai.llama import LlamaError, default_bin_dir, tool
from bonsai.prune import (
    METHODS,
    PruneError,
    choose_layers,
    load_scores,
    prune,
    score_layers,
)
from bonsai.server import LlamaServer

prune_app = typer.Typer(help="Prune seeds by depth: score their layers, then cut.")

Manifest = Annotated[
    Path,
    typer.Argument(
        help="The seed's manifest, e.g. "
        "configs/seeds/qwen2.5-0.5b-instruct-weights.toml.",
        exists=True,
        dir_okay=False,
    ),
]


def _fail(error: Exception) -> typer.Exit:
    typer.echo(f"error: {error}", err=True)
    return typer.Exit(1)


def _seed(manifest: Path) -> tuple[Path, str]:
    loaded = load_manifest(manifest)
    if loaded.kind != "seeds":
        msg = f"{manifest} is a {loaded.kind} manifest, not a seed"
        raise PruneError(msg)
    return destination(loaded), loaded.name


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")


@prune_app.command("score")
def score_command(
    manifest: Manifest,
    text: Annotated[
        Path | None,
        typer.Option(
            help="Calibration text. Default: WikiText-2's training split, from "
            "configs/data/calibration-v1.toml."
        ),
    ] = None,
    samples: Annotated[int, typer.Option(help="Windows of text to average over.")] = 64,
    length: Annotated[int, typer.Option(help="Tokens per window.")] = 512,
    out: Annotated[
        Path,
        typer.Option(help="Where to keep the scores, for later cuts and the record."),
    ] = Path("results/prunings"),
) -> None:
    """Score how much each run of a seed's layers changes its hidden state."""
    try:
        seed_dir, name = _seed(manifest)
        calibration = (
            text or bonsai_home() / "data" / "calibration-v1" / "wiki.train.raw"
        )
        if not calibration.is_file():
            msg = (
                f"no {calibration} (run: bonsai fetch configs/data/calibration-v1.toml)"
            )
            raise PruneError(msg)
        target = out / name / "scores.json"
        scores = score_layers(
            seed_dir, calibration, target, source=bonsai_home() / "llama.cpp",
            samples=samples, length=length,
        )  # fmt: skip
    except (FetchError, LlamaError, PruneError) as error:
        raise _fail(error) from error
    single = [row[0] for row in scores["distance"]]
    typer.echo(
        f"{scores['layers']} layers, {scores['tokens']} tokens; scores in {target}"
    )
    typer.echo("layer  change")
    for layer, value in enumerate(single):
        typer.echo(f"{layer:5d}  {value:.3f}")


@prune_app.command("cut")
def cut_command(
    manifest: Manifest,
    cut: Annotated[int, typer.Option(help="How many layers to remove.")],
    method: Annotated[
        str, typer.Option(help=f"How to choose them: {', '.join(METHODS)}.")
    ] = "block",
    scores: Annotated[
        Path | None,
        typer.Option(
            help="Layer scores. Default: results/prunings/<seed>/scores.json."
        ),
    ] = None,
    name: Annotated[
        str | None,
        typer.Option(help="The pruned seed's name. Default: <seed>-<method><cut>."),
    ] = None,
    record: Annotated[
        Path, typer.Option(help="Where to keep the pruning record, for the journal.")
    ] = Path("results/prunings"),
) -> None:
    """Cut layers from a seed, writing a new seed to ~/bonsai/seeds/<name>."""
    try:
        seed_dir, seed_name = _seed(manifest)
        scores_path = scores or record / seed_name / "scores.json"
        scored = load_scores(scores_path)
        removed = choose_layers(scored, cut, method)
        pruned_name = name or f"{seed_name.removesuffix('-weights')}-{method}{cut}"
        pruning = prune(
            seed_dir, bonsai_home() / "seeds" / pruned_name, name=pruned_name,
            seed_name=seed_name, removed=removed, method=method,
        )  # fmt: skip
    except (FetchError, PruneError) as error:
        raise _fail(error) from error
    kept = record / pruned_name / "prune.json"
    _write_json(kept, pruning.record(scored, scores_path.as_posix()))
    typer.echo(
        f"{pruned_name}: removed layers {', '.join(map(str, pruning.removed))} "
        f"({pruning.layers_before} -> {pruning.layers_after}); record in {kept}"
    )


def check_command(
    tree: Annotated[
        Path,
        typer.Argument(help="The planted tree (GGUF).", exists=True, dir_okay=False),
    ],
    seed: Annotated[
        Path,
        typer.Argument(
            help="The weights it was planted from.", exists=True, file_okay=False
        ),
    ],
    tokens: Annotated[int, typer.Option(help="Tokens to continue each prompt.")] = 16,
    out: Annotated[Path, typer.Option(help="Where to keep the check's result.")] = Path(
        "results/checks"
    ),
) -> None:
    """Check a planted tree continues fixed prompts as its weights do."""
    target = out / tree.stem
    try:
        reference = target / "reference.json"
        cached = load_reference(reference) if reference.is_file() else None
        if cached is None or cached.get("model") != str(seed):
            reference_answers(
                seed, reference, source=bonsai_home() / "llama.cpp", tokens=tokens
            )
        expected = load_reference(reference)
        server = tool(default_bin_dir(), "llama-server")
        with LlamaServer(
            server, tree, parallel=1, context=1024, log=target / "server.log"
        ) as running:
            result = compare(expected, running.complete)
    except (CheckError, LlamaError) as error:
        raise _fail(error) from error
    result["tree"] = tree.name
    _write_json(target / "check.json", result)
    verdict = "passed" if result["passed"] else "FAILED"
    typer.echo(
        f"{tree.name}: {verdict}. First tokens agree on {result['first_tokens_agree']}"
        f"/{result['prompts']} prompts; {result['tokens_agree']}/{result['tokens']} "
        f"tokens agree ({result['agreement']:.0%})."
    )
    if not result["passed"]:
        raise typer.Exit(1)
