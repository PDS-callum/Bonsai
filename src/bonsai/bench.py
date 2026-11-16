"""Read llama-bench results and compare them with a pot's speed ceiling.

llama-bench (run with ``-o jsonl``) prints one JSON record per test: either
prompt processing (``n_prompt > 0``, "pp") or token generation (``n_gen > 0``,
"tg"). Generation is the test the bytes-per-token ceiling applies to: each new
token reads every weight once, so its speed can be set against
``read bandwidth / model bytes``. Prompt processing works on many tokens per
weight read and is limited by arithmetic instead.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bonsai._validate import require
from bonsai.pot import Pot


class BenchError(ValueError):
    """A llama-bench record is malformed or missing a field."""


@dataclass(frozen=True)
class BenchRun:
    """One llama-bench test: a model, a workload and its measured speed."""

    model_file: str
    model_type: str
    """llama-bench's description, e.g. "qwen2 1B Q4_K - Medium"."""
    model_bytes: int
    """Bytes of tensor data, as llama-bench reports it."""
    threads: int
    n_prompt: int
    n_gen: int
    tokens_per_second: float
    """Mean over the repetitions."""
    stddev: float
    samples: tuple[float, ...]
    build_commit: str

    @property
    def test(self) -> str:
        """llama-bench's test name, e.g. "pp128" or "tg128"."""
        if self.n_prompt and self.n_gen:
            return f"pp{self.n_prompt}+tg{self.n_gen}"
        return f"pp{self.n_prompt}" if self.n_prompt else f"tg{self.n_gen}"

    @property
    def is_generation(self) -> bool:
        return self.n_gen > 0 and self.n_prompt == 0


def _field(
    record: dict[str, Any], key: str, kind: type | tuple[type, ...], where: str
) -> Any:
    return require(record, key, kind, where, BenchError)


def _basename(path: str) -> str:
    """The file name from a path written on either Linux or Windows."""
    return path.replace("\\", "/").rsplit("/", 1)[-1]


def parse_run(record: dict[str, Any], where: str = "record") -> BenchRun:
    """Build a run from one parsed llama-bench JSON record."""
    samples = _field(record, "samples_ts", list, where)
    if not all(isinstance(s, int | float) for s in samples):
        msg = f"{where}: 'samples_ts' must hold numbers"
        raise BenchError(msg)
    return BenchRun(
        model_file=_basename(_field(record, "model_filename", str, where)),
        model_type=_field(record, "model_type", str, where),
        model_bytes=_field(record, "model_size", int, where),
        threads=_field(record, "n_threads", int, where),
        n_prompt=_field(record, "n_prompt", int, where),
        n_gen=_field(record, "n_gen", int, where),
        tokens_per_second=float(_field(record, "avg_ts", (int, float), where)),
        stddev=float(_field(record, "stddev_ts", (int, float), where)),
        samples=tuple(float(s) for s in samples),
        build_commit=_field(record, "build_commit", str, where),
    )


def load_runs(path: Path) -> list[BenchRun]:
    """Read every record in a llama-bench JSONL file."""
    runs = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        where = f"{path}:{number}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            msg = f"{where}: {error}"
            raise BenchError(msg) from error
        if not isinstance(record, dict):
            msg = f"{where}: expected a JSON object"
            raise BenchError(msg)
        runs.append(parse_run(record, where))
    return runs


@dataclass(frozen=True)
class CeilingComparison:
    """A generation run set against the pot's bytes-per-token ceiling."""

    run: BenchRun
    ceiling_tokens_per_second: float

    @property
    def fraction_of_ceiling(self) -> float:
        return self.run.tokens_per_second / self.ceiling_tokens_per_second

    @property
    def effective_bandwidth_gbs(self) -> float:
        """The read bandwidth the run implies if it reads every weight once."""
        return self.run.tokens_per_second * self.run.model_bytes / 1e9


def compare_with_ceiling(pot: Pot, runs: list[BenchRun]) -> list[CeilingComparison]:
    """Set every generation run against the pot's ceiling for its model size."""
    return [
        CeilingComparison(run, pot.ceiling_tokens_per_second(run.model_bytes))
        for run in runs
        if run.is_generation
    ]


def format_report(pot: Pot, runs: list[BenchRun]) -> str:
    """A Markdown table of each tree's speeds against the pot's ceiling.

    Trees appear in order of size. Prompt speed comes from the tree's
    prompt-processing run; generation speed, the ceiling and the fraction of it
    reached come from its generation run.
    """
    by_model: dict[str, dict[str, BenchRun]] = {}
    for run in runs:
        kind = "tg" if run.is_generation else "pp" if run.n_gen == 0 else None
        if kind:
            by_model.setdefault(run.model_file, {})[kind] = run

    lines = [
        "| Tree | Type | Size | Prompt | Generation | Ceiling | Of ceiling "
        "| Implied bandwidth |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ordered = sorted(
        by_model.items(), key=lambda item: next(iter(item[1].values())).model_bytes
    )
    for model_file, tests in ordered:
        any_run = next(iter(tests.values()))
        pp = tests.get("pp")
        tg = tests.get("tg")
        prompt = f"{pp.tokens_per_second:.1f} t/s" if pp else "-"
        if tg:
            comparison = CeilingComparison(
                tg, pot.ceiling_tokens_per_second(tg.model_bytes)
            )
            generation = f"{tg.tokens_per_second:.1f} ± {tg.stddev:.1f} t/s"
            ceiling = f"{comparison.ceiling_tokens_per_second:.1f} t/s"
            fraction = f"{comparison.fraction_of_ceiling:.0%}"
            implied = f"{comparison.effective_bandwidth_gbs:.1f} GB/s"
        else:
            generation = ceiling = fraction = implied = "-"
        lines.append(
            f"| {model_file} | {any_run.model_type} "
            f"| {any_run.model_bytes / 1e6:.0f} MB | {prompt} | {generation} "
            f"| {ceiling} | {fraction} | {implied} |"
        )
    return "\n".join(lines) + "\n"
