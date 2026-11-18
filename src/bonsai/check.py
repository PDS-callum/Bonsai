"""Check a tree against the weights it was planted from, before measuring it.

A conversion can go wrong quietly: a misread config, a tensor renumbered
wrongly by pruning, a tokenizer mismatch. The tree still runs; its scores are
just meaningless. So before a planted tree is measured, it continues a fixed
set of prompts greedily, and the tokens are compared with what the original
weights produce in PyTorch at float32 (``layer_tool.py reference``).

Both sides start from the same token ids, so tokenisation cannot hide a
difference. Two correct implementations can still differ where the next token
is a near-tie, which grows more common as pruning damages a model. So the
check asks that every prompt's first token agrees, or is one PyTorch scores
within ``TIE`` nats of its own choice, and that most tokens agree up to the
first difference. A broken conversion fails both by a wide margin.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from bonsai.llama import ToolRunner, run_tool
from bonsai.prune import tool_command

PROMPTS = Path("prompts/reference-v1.json")
TIE = 0.1
"""How close, in nats, two first tokens' log-probabilities count as a tie."""
Continue = Callable[[Sequence[int], int], list[int]]


class CheckError(ValueError):
    """The reference answers are unusable."""


def reference_answers(
    seed: Path,
    out: Path,
    *,
    source: Path,
    prompts: Path = PROMPTS,
    tokens: int = 16,
    runner: ToolRunner = run_tool,
) -> dict[str, Any]:
    """Greedy continuations from the weights in ``seed``; saved to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    runner(tool_command(
        source, "reference", str(seed), str(prompts), str(out), "--tokens", str(tokens)
    ))  # fmt: skip
    return load_reference(out)


def load_reference(path: Path) -> dict[str, Any]:
    try:
        reference: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        for answer in reference["answers"]:
            answer["prompt_ids"], answer["continuation_ids"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        msg = f"{path}: not a reference answer file ({error!r})"
        raise CheckError(msg) from error
    return reference


def _prefix(a: Sequence[int], b: Sequence[int]) -> int:
    """How many tokens agree before the first difference."""
    for i, (x, y) in enumerate(zip(a, b, strict=False)):
        if x != y:
            return i
    return min(len(a), len(b))


def _near_tie(answer: dict[str, Any], token: int | None) -> bool:
    """Whether PyTorch scored ``token`` within ``TIE`` of its first choice."""
    top = answer.get("first_top") or []
    if not top or token is None:
        return False
    best = top[0][1]
    return any(t == token and best - logprob <= TIE for t, logprob in top)


def compare(
    reference: dict[str, Any], continue_: Continue, *, min_agreement: float = 0.75
) -> dict[str, Any]:
    """Continue each reference prompt with the tree and score the agreement."""
    results = []
    for answer in reference["answers"]:
        expected = answer["continuation_ids"]
        got = continue_(answer["prompt_ids"], len(expected))
        agree = _prefix(expected, got)
        results.append({
            "prompt": answer.get("prompt", ""),
            "expected": expected,
            "got": got,
            "agree": agree,
            "near_tie": agree == 0 and _near_tie(answer, got[0] if got else None),
        })  # fmt: skip
    total = sum(len(r["expected"]) for r in results)
    agreed = sum(r["agree"] for r in results)
    first = sum(r["agree"] > 0 or r["near_tie"] for r in results)
    agreement = agreed / total if total else 0.0
    return {
        "prompts": len(results),
        "first_tokens_agree": first,
        "near_ties": sum(r["near_tie"] for r in results),
        "tokens_agree": agreed,
        "tokens": total,
        "agreement": round(agreement, 4),
        "min_agreement": min_agreement,
        "passed": bool(results)
        and first == len(results)
        and agreement >= min_agreement,
        "answers": results,
    }
