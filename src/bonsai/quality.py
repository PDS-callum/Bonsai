"""Measure a tree's quality on the potting bench.

Four measurements, from cheapest and most sensitive to most task-like:

- **Perplexity** on the WikiText-2 test set: how well the tree predicts text.
- **KL divergence** from a base tree (usually the seed at full precision): how
  far each next-token prediction has moved. It catches damage that perplexity
  averages away, and needs far fewer tokens to resolve small differences.
- **ARC** (Easy and Challenge): science questions, scored by which answer the
  tree finds most probable, so no answer parsing is involved.
- **GSM8K**: maths problems the tree solves in writing, on a fixed subset.

Every run keeps the tools' full output beside its ``quality.json``, so a number
can always be traced to the run that produced it.
"""

from __future__ import annotations

import gzip
import json
import os
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from bonsai import gsm8k
from bonsai.fetch import bonsai_home, sha256_of
from bonsai.llama import (
    ToolRunner,
    parse_kl_divergence,
    parse_multiple_choice,
    parse_perplexity,
    run_tool,
    tool,
)
from bonsai.server import LlamaServer, Reply

WIKI = "wiki.test.raw"
ARC = "arc-combined-validation.bin"
GSM8K = "gsm8k-test.jsonl"


class Chat(Protocol):
    def chat(self, prompt: str, *, max_tokens: int = ..., seed: int = ...) -> Reply: ...


ServerFactory = Callable[[Path, Path], Any]
"""Given a tree and a log path, returns a context manager yielding a Chat."""


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    bin_dir: Path
    threads: int = field(default_factory=lambda: os.cpu_count() or 4)
    context: int = 512
    kld_chunks: int = 32
    arc_parallel: int = 8
    gsm8k_size: int = 400
    gsm8k_seed: int = 0
    gsm8k_parallel: int = 4
    gsm8k_max_tokens: int = 512


def _perplexity(settings: Settings) -> str:
    return str(tool(settings.bin_dir, "llama-perplexity"))


def base_logits_path(base: Path, settings: Settings) -> Path:
    """Where a base tree's saved predictions live: large, so outside the repo."""
    return bonsai_home() / "cache" / "kld" / f"{base.stem}-{settings.kld_chunks}.kld"


def save_base_logits(
    base: Path, settings: Settings, runner: ToolRunner = run_tool
) -> Path:
    """Record the base tree's predictions on the first chunks of WikiText-2.

    Reused if already saved. Each token keeps the whole vocabulary's
    probabilities, so this is about 2.5 GB for 32 chunks of a 150k vocabulary.
    """
    target = base_logits_path(base, settings)
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    runner(
        [
            _perplexity(settings),
            "--model", str(base),
            "--file", str(settings.data_dir / WIKI),
            "--ctx-size", str(settings.context),
            "--chunks", str(settings.kld_chunks),
            "--threads", str(settings.threads),
            "--kl-divergence-base", str(partial),
        ]
    )  # fmt: skip
    partial.replace(target)
    return target


def _run_logged(runner: ToolRunner, command: list[str], log: Path) -> str:
    output = runner(command)
    log.write_text(output, encoding="utf-8")
    return output


def measure_gsm8k(
    server: Chat, settings: Settings, replies_path: Path
) -> dict[str, Any]:
    problems = gsm8k.subset(
        gsm8k.load(settings.data_dir / GSM8K),
        settings.gsm8k_size,
        settings.gsm8k_seed,
    )

    def solve(problem: gsm8k.Problem) -> dict[str, Any]:
        reply = server.chat(
            gsm8k.PROMPT.format(question=problem.question),
            max_tokens=settings.gsm8k_max_tokens,
            seed=settings.gsm8k_seed,
        )
        predicted, rule = gsm8k.extract_answer(reply.text)
        return {
            "index": problem.index,
            "gold": problem.answer,
            "predicted": predicted,
            "rule": rule,
            "correct": gsm8k.is_correct(predicted, problem.answer),
            "tokens": reply.completion_tokens,
            "reply": reply.text,
        }

    with ThreadPoolExecutor(settings.gsm8k_parallel) as pool:
        results = list(pool.map(solve, problems))

    with gzip.open(replies_path, "wt", encoding="utf-8") as out:
        for result in results:
            out.write(json.dumps(result) + "\n")

    correct = sum(r["correct"] for r in results)
    low, high = gsm8k.wilson_interval(correct, len(results))
    return {
        "size": len(results),
        "seed": settings.gsm8k_seed,
        "correct": correct,
        "accuracy": correct / len(results) if results else 0.0,
        "ci95": [low, high],
        "answer_rules": dict(Counter(r["rule"] for r in results)),
        "mean_tokens": (
            sum(r["tokens"] for r in results) / len(results) if results else 0.0
        ),
    }


def measure(
    tree: Path,
    out_dir: Path,
    settings: Settings,
    *,
    base: Path | None = None,
    base_logits: Path | None = None,
    runner: ToolRunner = run_tool,
    server_factory: ServerFactory | None = None,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Measure one tree and write ``out_dir/<tree stem>/quality.json``."""
    target = out_dir / tree.stem
    target.mkdir(parents=True, exist_ok=True)
    perplexity = _perplexity(settings)
    threads = ["--threads", str(settings.threads)]

    log(f"{tree.name}: perplexity")
    ppl = parse_perplexity(
        _run_logged(
            runner,
            [
                perplexity,
                "--model",
                str(tree),
                "--file",
                str(settings.data_dir / WIKI),
                "--ctx-size",
                str(settings.context),
                *threads,
            ],
            target / "perplexity.log",
        )
    )

    divergence: dict[str, Any] | None = None
    if base_logits and base and tree.resolve() != base.resolve():
        log(f"{tree.name}: KL divergence from {base.name}")
        kld = parse_kl_divergence(
            _run_logged(
                runner,
                [
                    perplexity,
                    "--model",
                    str(tree),
                    "--kl-divergence-base",
                    str(base_logits),
                    "--kl-divergence",
                    *threads,
                ],
                target / "kl-divergence.log",
            )
        )
        divergence = {
            "base": base.name,
            "chunks": settings.kld_chunks,
            "mean": kld.mean_kld.value,
            "mean_stderr": kld.mean_kld.stderr,
            "p99": kld.kld_99,
            "same_top_pct": kld.same_top.value,
            "same_top_pct_stderr": kld.same_top.stderr,
            "ppl_ratio": kld.ppl_ratio.value,
            "ppl_ratio_stderr": kld.ppl_ratio.stderr,
        }

    log(f"{tree.name}: ARC")
    arc = parse_multiple_choice(
        _run_logged(
            runner,
            [
                perplexity,
                "--model",
                str(tree),
                "--binary-file",
                str(settings.data_dir / ARC),
                "--multiple-choice",
                "--parallel",
                str(settings.arc_parallel),
                "--ctx-size",
                str(512 * settings.arc_parallel),
                *threads,
            ],
            target / "arc.log",
        )
    )

    log(f"{tree.name}: GSM8K ({settings.gsm8k_size} problems)")
    factory = server_factory or default_server(settings)
    with factory(tree, target / "server.log") as server:
        maths = measure_gsm8k(server, settings, target / "gsm8k-replies.jsonl.gz")

    results: dict[str, Any] = {
        "tree": tree.name,
        "bytes": tree.stat().st_size,
        "sha256": sha256_of(tree),
        "measured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "llama_cpp": settings.bin_dir.name,
        "data": settings.data_dir.name,
        "perplexity": {
            "value": ppl.value,
            "stderr": ppl.stderr,
            "context": settings.context,
        },
        "kl_divergence": divergence,
        "arc": {"accuracy_pct": arc.value, "stderr_pct": arc.stderr},
        "gsm8k": maths,
    }
    (target / "quality.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    return results


def default_server(settings: Settings) -> ServerFactory:
    def make(tree: Path, log: Path) -> LlamaServer:
        return LlamaServer(
            tool(settings.bin_dir, "llama-server"),
            tree,
            parallel=settings.gsm8k_parallel,
            context=settings.gsm8k_parallel * 2048,
            threads=settings.threads,
            log=log,
        )

    return make


def load_results(directory: Path) -> list[dict[str, Any]]:
    """Every ``quality.json`` under a directory."""
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(directory.rglob("quality.json"))
    ]


def format_report(results: list[dict[str, Any]]) -> str:
    """A Markdown table of trees' quality, smallest first."""
    lines = [
        "| Tree | Size | Perplexity | KL divergence | Same top token | ARC | GSM8K |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in sorted(results, key=lambda r: r["bytes"]):
        kld = result.get("kl_divergence")
        maths = result["gsm8k"]
        low, high = maths["ci95"]
        lines.append(
            f"| {result['tree']} | {result['bytes'] / 1e6:.0f} MB "
            f"| {result['perplexity']['value']:.2f} "
            f"± {result['perplexity']['stderr']:.2f} "
            + (
                f"| {kld['mean']:.4f} | {kld['same_top_pct']:.1f}% "
                if kld
                else "| - | - "
            )
            + f"| {result['arc']['accuracy_pct']:.1f}% "
            f"± {result['arc']['stderr_pct']:.1f} "
            f"| {maths['accuracy']:.1%} ({low:.1%} to {high:.1%}) |"
        )
    return "\n".join(lines) + "\n"
