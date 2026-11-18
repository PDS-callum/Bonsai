"""Prune by depth: score a seed's layers, then cut whole layers out of it.

Scoring runs ``scripts/pc/layer_tool.py`` in the same throwaway environment as
llama.cpp's converter, which already has PyTorch and transformers. For each
run of consecutive layers it reports the angular distance between the hidden
state going in and the one coming out: how much those layers change it. Two
ways of choosing layers are offered:

``block``   the run of ``cut`` consecutive layers that changes the state least,
            judged as a whole (Gromov et al., 2024)
``single``  the ``cut`` layers that each change it least on their own (Men et
            al., 2024, "ShortGPT")

Cutting needs no machine learning library at all. A ``safetensors`` file is a
JSON header naming each tensor's byte range, followed by the bytes, so layers
are removed by copying every other tensor into a new file and renumbering the
layers that remain. Every remaining layer keeps its shape, so the pruned seed
still converts to a standard GGUF with ``bonsai plant``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import struct
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from bonsai.llama import ToolRunner, run_tool
from bonsai.plant import CONVERT_REQUIREMENTS

LAYER_TOOL = Path("scripts/pc/layer_tool.py")
METHODS = ("block", "single")
WEIGHTS = "model.safetensors"
_LAYER = re.compile(r"^model\.layers\.(\d+)\.(.+)$")
CHUNK = 16 * 1024 * 1024


class PruneError(ValueError):
    """Scores, a seed or a requested cut are unusable."""


def tool_command(source: Path, *args: str) -> list[str]:
    """``layer_tool.py`` in the converter's environment (PyTorch, transformers)."""
    return [
        "uv", "run", "--no-project", "--python", "3.12",
        "--with-requirements", str(source / CONVERT_REQUIREMENTS),
        "python", str(LAYER_TOOL), *args,
    ]  # fmt: skip


def score_layers(
    seed: Path,
    text: Path,
    out: Path,
    *,
    source: Path,
    samples: int = 64,
    length: int = 512,
    runner: ToolRunner = run_tool,
) -> dict[str, Any]:
    """Score every run of layers in ``seed`` on ``text``; saved to ``out``."""
    out.parent.mkdir(parents=True, exist_ok=True)
    runner(tool_command(
        source, "score", str(seed), str(text), str(out),
        "--samples", str(samples), "--length", str(length),
    ))  # fmt: skip
    return load_scores(out)


def load_scores(path: Path) -> dict[str, Any]:
    try:
        scores: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        distance = scores["distance"]
        layers = scores["layers"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        msg = f"{path}: not a layer score file ({error!r})"
        raise PruneError(msg) from error
    if len(distance) != layers - 1:
        msg = f"{path}: expected scores for {layers - 1} layers, got {len(distance)}"
        raise PruneError(msg)
    return scores


def choose_layers(scores: dict[str, Any], cut: int, method: str) -> list[int]:
    """The layers to remove, lowest first. The last layer is never chosen: its
    raw output is not exposed, so it has no score."""
    distance: list[list[float]] = scores["distance"]
    candidates = len(distance)
    if method not in METHODS:
        msg = f"method must be one of {', '.join(METHODS)}, got {method!r}"
        raise PruneError(msg)
    if not 0 < cut <= candidates - 1:
        msg = f"can cut 1 to {candidates - 1} of {scores['layers']} layers, not {cut}"
        raise PruneError(msg)
    if method == "block":
        start = min(range(candidates - cut + 1), key=lambda i: distance[i][cut - 1])
        return list(range(start, start + cut))
    ranked = sorted(range(candidates), key=lambda i: distance[i][0])
    return sorted(ranked[:cut])


def read_header(path: Path) -> tuple[dict[str, Any], int]:
    """A safetensors file's header, and where its tensor data starts."""
    with path.open("rb") as file:
        raw = file.read(8)
        if len(raw) != 8:
            msg = f"{path}: too short to be safetensors"
            raise PruneError(msg)
        (size,) = struct.unpack("<Q", raw)
        try:
            header: dict[str, Any] = json.loads(file.read(size))
        except ValueError as error:
            msg = f"{path}: unreadable safetensors header ({error})"
            raise PruneError(msg) from error
    return header, 8 + size


def _renamed(name: str, keep: dict[int, int]) -> str | None:
    """The tensor's new name, or None if its layer is removed."""
    match = _LAYER.match(name)
    if not match:
        return name
    old = int(match.group(1))
    return f"model.layers.{keep[old]}.{match.group(2)}" if old in keep else None


def cut_weights(source: Path, out: Path, removed: Sequence[int], layers: int) -> str:
    """Write ``source`` without the removed layers to ``out``; its SHA-256."""
    header, start = read_header(source)
    keep = {
        old: new for new, old in enumerate(i for i in range(layers) if i not in removed)
    }
    tensors = sorted(
        ((name, spec) for name, spec in header.items() if name != "__metadata__"),
        key=lambda item: item[1]["data_offsets"][0],
    )
    new_header: dict[str, Any] = {}
    if "__metadata__" in header:
        new_header["__metadata__"] = header["__metadata__"]
    plan = []
    offset = 0
    for name, spec in tensors:
        new_name = _renamed(name, keep)
        if new_name is None:
            continue
        begin, end = spec["data_offsets"]
        new_header[new_name] = {**spec, "data_offsets": [offset, offset + end - begin]}
        plan.append((begin, end))
        offset += end - begin
    encoded = json.dumps(new_header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)  # the format pads the header to 8 bytes
    digest = hashlib.sha256()
    part = out.with_name(out.name + ".part")
    with source.open("rb") as reader, part.open("wb") as writer:

        def emit(data: bytes) -> None:
            writer.write(data)
            digest.update(data)

        emit(struct.pack("<Q", len(encoded)) + encoded)
        for begin, end in plan:
            reader.seek(start + begin)
            remaining = end - begin
            while remaining:
                data = reader.read(min(CHUNK, remaining))
                if not data:
                    msg = f"{source}: ends before its tensors do"
                    raise PruneError(msg)
                emit(data)
                remaining -= len(data)
    part.replace(out)
    return digest.hexdigest()


def cut_config(config: dict[str, Any], removed: Sequence[int]) -> dict[str, Any]:
    """The model config with the removed layers gone."""
    layers = config["num_hidden_layers"]
    remaining = layers - len(removed)
    updated = {**config, "num_hidden_layers": remaining}
    if isinstance(config.get("max_window_layers"), int):
        updated["max_window_layers"] = min(config["max_window_layers"], remaining)
    if isinstance(config.get("layer_types"), list):
        updated["layer_types"] = [
            kind for i, kind in enumerate(config["layer_types"]) if i not in removed
        ]
    return updated


@dataclass(frozen=True)
class Pruning:
    name: str
    seed: str
    removed: tuple[int, ...]
    method: str
    weights_sha256: str
    layers_before: int
    layers_after: int

    def record(
        self, scores: dict[str, Any] | None, scores_file: str | None
    ) -> dict[str, Any]:
        """The pruning record kept in the repository, read by the journal."""
        return {
            "name": self.name,
            "seed": self.seed,
            "method": self.method,
            "removed_layers": list(self.removed),
            "layers_before": self.layers_before,
            "layers_after": self.layers_after,
            "weights_sha256": self.weights_sha256,
            "scores": scores_file,
            "calibration": None if scores is None else {
                key: scores.get(key) for key in ("text", "samples", "length", "tokens")
            },
        }  # fmt: skip


def prune(
    seed: Path,
    out: Path,
    *,
    name: str,
    seed_name: str,
    removed: Sequence[int],
    method: str,
) -> Pruning:
    """Write a copy of the seed in ``seed`` without the ``removed`` layers."""
    weights = seed / WEIGHTS
    if not weights.is_file() or not (seed / "config.json").is_file():
        msg = (
            f"{seed} has no {WEIGHTS} and config.json (sharded seeds are not supported)"
        )
        raise PruneError(msg)
    config = json.loads((seed / "config.json").read_text(encoding="utf-8"))
    layers = config["num_hidden_layers"]
    if (
        not removed
        or any(not 0 <= i < layers for i in removed)
        or len(set(removed)) != len(removed)
    ):
        msg = (
            f"layers to remove must be distinct and within 0..{layers - 1}: "
            f"{list(removed)}"
        )
        raise PruneError(msg)
    out.mkdir(parents=True, exist_ok=True)
    for file in seed.iterdir():
        if file.is_file() and file.name not in (WEIGHTS, "config.json"):
            shutil.copyfile(file, out / file.name)
    (out / "config.json").write_text(
        json.dumps(cut_config(config, removed), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    sha = cut_weights(weights, out / WEIGHTS, sorted(removed), layers)
    return Pruning(
        name=name,
        seed=seed_name,
        removed=tuple(sorted(removed)),
        method=method,
        weights_sha256=sha,
        layers_before=layers,
        layers_after=layers - len(removed),
    )
