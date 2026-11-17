"""Run llama.cpp's tools on the potting bench and read what they report.

Quality is measured with the same pinned llama.cpp build the pot runs, so a
score and a speed always describe the same code. The tools print their results
as text; the parsers here turn that text into numbers, and each run's full
output is kept beside its results for provenance.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

ToolRunner = Callable[[Sequence[str]], str]
"""Runs a command and returns everything it printed (stdout and stderr)."""


class LlamaError(RuntimeError):
    """A llama.cpp tool is missing, failed, or printed no result."""


def run_tool(command: Sequence[str]) -> str:
    """Run a llama.cpp tool, returning its combined output."""
    try:
        result = subprocess.run(
            list(command),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as error:
        msg = f"{command[0]} not found"
        raise LlamaError(msg) from error
    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-5:])
        msg = f"{Path(command[0]).name} exited with {result.returncode}:\n{tail}"
        raise LlamaError(msg)
    return result.stdout


def read_env_file(path: Path) -> dict[str, str]:
    """Read ``KEY=value`` lines, as in ``configs/llama-cpp.env``."""
    values = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        key, sep, value = line.partition("=")
        if sep and not line.lstrip().startswith("#"):
            values[key.strip()] = value.strip()
    return values


def default_bin_dir(
    pin_file: Path = Path("configs/llama-cpp.env"), home: Path | None = None
) -> Path:
    """Where the pinned prebuilt tools live: ``$BONSAI_LLAMA_BIN``, or
    ``~/bonsai/llama.cpp-bin/<build>`` for the build ``pin_file`` names."""
    if env := os.environ.get("BONSAI_LLAMA_BIN"):
        return Path(env)
    home = home or Path(os.environ.get("BONSAI_HOME", Path.home() / "bonsai"))
    if pin_file.is_file():
        build = read_env_file(pin_file).get("LLAMA_CPP_BUILD")
        if build:
            return home / "llama.cpp-bin" / build
    return home / "llama.cpp" / "build" / "bin"


def tool(bin_dir: Path, name: str) -> Path:
    """The path of a llama.cpp tool, with ``.exe`` on Windows."""
    path = bin_dir / (f"{name}.exe" if sys.platform == "win32" else name)
    if not path.is_file():
        msg = f"{name} not found in {bin_dir} (set BONSAI_LLAMA_BIN?)"
        raise LlamaError(msg)
    return path


# Parsers. Each takes a tool's full output and returns its headline numbers.

_NUMBER = r"([-+]?\d+(?:\.\d+)?)"


@dataclass(frozen=True)
class Estimate:
    value: float
    stderr: float


def _final(pattern: str, output: str, what: str) -> Estimate:
    matches = re.findall(pattern, output)
    if not matches:
        msg = f"no {what} in the tool's output"
        raise LlamaError(msg)
    value, stderr = matches[-1]
    return Estimate(float(value), float(stderr))


def parse_perplexity(output: str) -> Estimate:
    """``Final estimate: PPL = 17.3755 +/- 1.16488``."""
    return _final(
        rf"Final estimate: PPL = {_NUMBER} \+/- {_NUMBER}", output, "perplexity"
    )


def parse_multiple_choice(output: str) -> Estimate:
    """``Final result: 48.7917 +/- 1.6966``, as a percentage."""
    return _final(
        rf"Final result: {_NUMBER} \+/- {_NUMBER}", output, "multiple-choice result"
    )


@dataclass(frozen=True)
class Divergence:
    """How far a tree's predictions are from its base's, token by token."""

    mean_kld: Estimate
    """Mean KL divergence from the base's next-token distribution, in nats."""
    same_top: Estimate
    """Percentage of tokens where both predict the same most likely token."""
    ppl_ratio: Estimate
    """The tree's perplexity divided by the base's, on the same tokens."""
    kld_99: float
    """The 99th percentile of KL divergence: how bad the worst tokens get."""


def parse_kl_divergence(output: str) -> Divergence:
    """Read the statistics ``llama-perplexity --kl-divergence`` prints."""

    def pair(label: str, suffix: str = "") -> Estimate:
        return _final(
            rf"{label}\s*:\s*{_NUMBER}\s*±\s*{_NUMBER}{suffix}", output, label
        )

    percentile = re.findall(rf"99\.0%\s+KLD:\s*{_NUMBER}", output)
    if not percentile:
        msg = "no KLD percentiles in the tool's output"
        raise LlamaError(msg)
    return Divergence(
        mean_kld=pair(r"Mean\s+KLD"),
        same_top=pair(r"Same top p", r"\s*%"),
        ppl_ratio=pair(r"Mean PPL\(Q\)/PPL\(base\)"),
        kld_99=float(percentile[-1]),
    )
