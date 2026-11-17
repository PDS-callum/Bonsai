"""GSM8K: grade-school maths problems, solved step by step.

Each problem has a numeric answer after ``####`` in its worked solution. A tree
is asked to reason and finish with "The answer is: <number>". Small models
don't always follow that format, so the answer is read in order of preference:
the last "answer is", then the last ``\\boxed{}``, then the last number in the
reply. Which rule found it is recorded, so formatting failures stay visible
apart from wrong arithmetic.

Measuring every problem takes about half an hour per tree on the bench, so
trees are compared on a fixed, seeded subset; accuracy comes with a Wilson
confidence interval so differences can be judged against noise.
"""

from __future__ import annotations

import json
import math
import random
import re
from dataclasses import dataclass
from pathlib import Path

PROMPT = (
    "{question}\n\nSolve the problem step by step. "
    'End with a final line of the form "The answer is: <number>".'
)

_NUMBER = r"-?\$?\d[\d,]*(?:\.\d+)?"


class Gsm8kError(ValueError):
    """The GSM8K data is malformed."""


@dataclass(frozen=True)
class Problem:
    index: int
    """Position in the full test set, so subsets can be traced."""
    question: str
    answer: float


def parse_number(text: str) -> float | None:
    """A number as written in a reply: ``$1,200``, ``-3``, ``4.50.``."""
    cleaned = text.replace(",", "").replace("$", "").rstrip(".")
    try:
        return float(cleaned)
    except ValueError:
        return None


def load(path: Path) -> list[Problem]:
    problems = []
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            question = record["question"]
            gold = parse_number(record["answer"].rsplit("####", 1)[1].strip())
        except (json.JSONDecodeError, KeyError, IndexError, AttributeError) as error:
            msg = f"{path}:{index + 1}: not a GSM8K record"
            raise Gsm8kError(msg) from error
        if gold is None:
            msg = f"{path}:{index + 1}: answer is not a number"
            raise Gsm8kError(msg)
        problems.append(Problem(index, question, gold))
    return problems


def subset(problems: list[Problem], size: int, seed: int = 0) -> list[Problem]:
    """A fixed sample of ``size`` problems, in their original order."""
    if size >= len(problems):
        return list(problems)
    chosen = random.Random(seed).sample(range(len(problems)), size)
    return [problems[i] for i in sorted(chosen)]


def extract_answer(reply: str) -> tuple[float | None, str]:
    """The reply's final answer, and the rule that found it."""
    rules = [
        ("answer-is", rf"answer is:?\s*\**\s*(?:\\\(\s*)?({_NUMBER})"),
        ("boxed", rf"\\boxed\{{\s*({_NUMBER})\s*\}}"),
        ("last-number", rf"({_NUMBER})"),
    ]
    for name, pattern in rules:
        matches = re.findall(pattern, reply, flags=re.IGNORECASE)
        for match in reversed(matches):
            value = parse_number(match)
            if value is not None:
                return value, name
    return None, "none"


def is_correct(predicted: float | None, gold: float) -> bool:
    return predicted is not None and math.isclose(predicted, gold, abs_tol=1e-6)


def wilson_interval(correct: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """95% confidence interval for a proportion; sound even near 0 or 1."""
    if total == 0:
        return 0.0, 1.0
    p = correct / total
    centre = (p + z * z / (2 * total)) / (1 + z * z / total)
    spread = (
        z
        * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total))
        / (1 + z * z / total)
    )
    return max(0.0, centre - spread), min(1.0, centre + spread)
