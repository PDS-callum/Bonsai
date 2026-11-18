"""Score a model's layers for pruning, or record reference answers to check
a converted tree against. Runs with PyTorch and transformers, in the throwaway
environment llama.cpp's converter uses (see ``bonsai.prune``), so Bonsai itself
does not depend on them.

score      For each layer, and for each run of n consecutive layers, the angular
           distance between the hidden state going in and the one coming out,
           averaged over tokens of calibration text. A small distance means the
           layers barely change the state, so cutting them should cost little
           (Gromov et al., 2024, "The Unreasonable Ineffectiveness of the Deeper
           Layers"; Men et al., 2024, "ShortGPT").

reference  Greedy continuations of fixed prompts, at float32, as token ids. A
           GGUF converted from the same weights should continue them the same.

Usage:
    python layer_tool.py score <model-dir> <text> <out.json> [--samples N] [--length L]
    python layer_tool.py reference <model-dir> <prompts.json> <out.json> [--tokens N]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def load(model_dir: Path):  # type: ignore[no-untyped-def]
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32)
    model.eval()
    return tokenizer, model


def versions() -> dict[str, str]:
    return {"torch": torch.__version__, "transformers": transformers.__version__}


def angular(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angular distance in [0, 1] between matching rows of a and b."""
    cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).clamp(-1.0, 1.0)
    return torch.arccos(cos) / math.pi


def score(args: argparse.Namespace) -> dict:
    tokenizer, model = load(args.model)
    text = args.text.read_text(encoding="utf-8")
    ids = tokenizer(text, return_tensors="pt").input_ids[0]
    windows = len(ids) // args.length
    if windows < args.samples:
        sys.exit(f"only {windows} windows of {args.length} tokens in {args.text}")
    # Evenly spaced windows, so the sample covers the whole text.
    starts = [i * (windows // args.samples) * args.length for i in range(args.samples)]
    layers = model.config.num_hidden_layers
    # hidden_states[i] is the input to layer i; the last one has the final norm
    # applied, so a layer's output is only seen raw up to layer layers - 2.
    usable = layers - 1
    sums = torch.zeros(usable + 1, usable + 1, dtype=torch.float64)
    count = 0
    with torch.no_grad():
        for start in starts:
            window = ids[start : start + args.length].unsqueeze(0)
            states = model(window, output_hidden_states=True).hidden_states
            # Skip the first position, whose state is an outlier in most models.
            states = [s[0, 1:].double() for s in states[: usable + 1]]
            for i in range(usable):
                for j in range(i + 1, usable + 1):
                    sums[i, j] += angular(states[i], states[j]).sum()
            count += states[0].shape[0]
    mean = sums / count
    return {
        "model": str(args.model),
        "layers": layers,
        "text": args.text.name,
        "samples": args.samples,
        "length": args.length,
        "tokens": count,
        "versions": versions(),
        "note": "distance[i][n] is the angular distance from the input of layer i "
        "to the input of layer i + n (so cutting layers i .. i + n - 1); the last "
        "layer is excluded because its raw output is not exposed",
        "distance": [
            [round(float(mean[i, i + n]), 6) for n in range(1, usable - i + 1)]
            for i in range(usable)
        ],
    }


def reference(args: argparse.Namespace) -> dict:
    tokenizer, model = load(args.model)
    prompts = json.loads(args.prompts.read_text(encoding="utf-8"))["prompts"]
    answers = []
    with torch.no_grad():
        for prompt in prompts:
            ids = tokenizer(prompt, return_tensors="pt").input_ids
            out = model.generate(
                ids, max_new_tokens=args.tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )  # fmt: skip
            answers.append({
                "prompt": prompt,
                "prompt_ids": ids[0].tolist(),
                "continuation_ids": out[0, ids.shape[1]:].tolist(),
                "continuation": tokenizer.decode(out[0, ids.shape[1]:]),
            })  # fmt: skip
    return {"model": str(args.model), "tokens": args.tokens,
            "versions": versions(), "answers": answers}  # fmt: skip


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    s = sub.add_parser("score")
    s.add_argument("model", type=Path)
    s.add_argument("text", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--samples", type=int, default=64)
    s.add_argument("--length", type=int, default=512)
    r = sub.add_parser("reference")
    r.add_argument("model", type=Path)
    r.add_argument("prompts", type=Path)
    r.add_argument("out", type=Path)
    r.add_argument("--tokens", type=int, default=16)
    args = parser.parse_args()
    torch.manual_seed(0)
    result = score(args) if args.mode == "score" else reference(args)
    args.out.write_text(
        json.dumps(result, indent=1) + "\n", encoding="utf-8", newline="\n"
    )


if __name__ == "__main__":
    main()
