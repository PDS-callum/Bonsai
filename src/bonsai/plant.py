"""Plant trees: turn a seed's original weights into GGUF trees.

A seed fetched as Hugging Face weights (``bonsai fetch``) is converted to a
full-precision GGUF with llama.cpp's own converter, then quantised with
``llama-quantize``, both from the pinned llama.cpp. Quantising without an
importance matrix is deterministic, so the same seed and pin always give the
same bytes; the planting record (``planting.json``) lists each tree's checksum
so that can be checked.

Trees are written to ``.part`` files and moved into place when complete, and a
tree that already exists is kept, so an interrupted planting can be rerun.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from bonsai.fetch import sha256_of
from bonsai.llama import ToolRunner, read_env_file, run_tool, tool

DEFAULT_TYPES = ("q2_k", "q4_0", "q4_k_m", "q6_k", "q8_0")
BASE_TYPES = ("bf16", "f16", "f32")
CONVERT_REQUIREMENTS = "requirements/requirements-convert_hf_to_gguf.txt"


class PlantError(RuntimeError):
    """A seed or the llama.cpp source is unusable, or a planting step failed."""


@dataclass(frozen=True)
class Pin:
    """The pinned llama.cpp, from ``configs/llama-cpp.env``."""

    commit: str
    build: str

    @classmethod
    def load(cls, path: Path) -> Pin:
        if not path.is_file():
            msg = f"no llama.cpp pin at {path} (run from the repository root)"
            raise PlantError(msg)
        values = read_env_file(path)
        try:
            return cls(values["LLAMA_CPP_COMMIT"], values["LLAMA_CPP_BUILD"])
        except KeyError as error:
            msg = f"{path}: missing {error.args[0]}"
            raise PlantError(msg) from error


def convert_command(source: Path, seed: Path, out: Path, outtype: str) -> list[str]:
    """llama.cpp's converter, run in a throwaway environment with its own
    pinned requirements (PyTorch among them), so Bonsai does not depend on them.
    """
    return [
        "uv", "run", "--no-project", "--python", "3.12",
        "--with-requirements", str(source / CONVERT_REQUIREMENTS),
        "python", str(source / "convert_hf_to_gguf.py"), str(seed),
        "--outtype", outtype, "--outfile", str(out),
    ]  # fmt: skip


def quantize_command(quantize: Path, base: Path, out: Path, qtype: str) -> list[str]:
    return [str(quantize), str(base), str(out), qtype.upper()]


def check_source(source: Path, pin: Pin, runner: ToolRunner) -> None:
    """The converter must come from the pinned commit, like the binaries."""
    if not (source / "convert_hf_to_gguf.py").is_file():
        msg = f"no llama.cpp source in {source} (run scripts/pc/fetch-llama-cpp.sh)"
        raise PlantError(msg)
    commit = runner(["git", "-C", str(source), "rev-parse", "HEAD"]).strip()
    if commit != pin.commit:
        msg = f"llama.cpp source in {source} is at {commit}, not the pin {pin.commit}"
        raise PlantError(msg)


def _make(
    out: Path, command: Callable[[Path], list[str]], runner: ToolRunner, log: Path
) -> None:
    """Run ``command(part)`` and move the finished part to ``out``."""
    part = out.with_name(out.name + ".part")
    part.unlink(missing_ok=True)
    log.write_text(runner(command(part)), encoding="utf-8", newline="\n")
    if not part.is_file():
        msg = f"{out.name} was not written (see {log})"
        raise PlantError(msg)
    part.replace(out)


def plant(
    seed: Path,
    out_dir: Path,
    name: str,
    *,
    source: Path,
    bin_dir: Path,
    pin: Pin,
    types: Sequence[str] = DEFAULT_TYPES,
    base_type: str = "bf16",
    runner: ToolRunner = run_tool,
    log: Callable[[str], None] = print,
) -> dict[str, Any]:
    """Plant ``<name>-<type>.gguf`` trees from the seed in ``seed``.

    Returns the planting record, also written to ``out_dir/planting.json``.
    """
    if base_type not in BASE_TYPES:
        msg = f"base type must be one of {', '.join(BASE_TYPES)}, got {base_type!r}"
        raise PlantError(msg)
    if not (seed / "config.json").is_file():
        msg = f"no seed in {seed} (run: bonsai fetch configs/seeds/{name}.toml)"
        raise PlantError(msg)
    check_source(source, pin, runner)
    quantize = tool(bin_dir, "llama-quantize")
    logs = out_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)

    def tree(kind: str) -> Path:
        return out_dir / f"{name}-{kind}.gguf"

    base = tree(base_type)
    steps: list[tuple[str, Callable[[Path], list[str]]]] = [
        (base_type, partial(convert_command, source, seed, outtype=base_type))
    ]
    steps += [
        (kind, partial(quantize_command, quantize, base, qtype=kind)) for kind in types
    ]
    for kind, command in steps:
        if tree(kind).is_file():
            log(f"kept   {tree(kind).name}")
            continue
        log(f"plant  {tree(kind).name}")
        _make(tree(kind), command, runner, logs / f"{kind}.log")

    record = {
        "seed": name,
        "planted_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "llama_cpp": {"commit": pin.commit, "build": pin.build},
        "base": base.name,
        "trees": [
            {
                "file": tree(kind).name,
                "type": kind,
                "bytes": tree(kind).stat().st_size,
                "sha256": sha256_of(tree(kind)),
            }
            for kind, _ in steps
        ],
    }
    (out_dir / "planting.json").write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return record
