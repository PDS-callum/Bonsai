"""Pots: the hardware a tree must fit, and the budgets that come with it.

A pot profile is a TOML file under ``configs/pots/``. It records measured facts
about the machine (memory bandwidth, free memory, storage) and the budgets
Bonsai works within. The central estimate is the bytes-per-token ceiling:

    tokens per second  <=  read bandwidth / bytes of active weights

Generating a token reads every active weight once, so no tree can run faster
than this on the pot, whatever its parameter count.
"""

from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GIB = 1024**3


class PotError(ValueError):
    """A pot profile is missing a field or has an invalid value."""


@dataclass(frozen=True)
class Pot:
    """The budgets a tree must fit within on one machine."""

    name: str
    description: str
    read_bandwidth_gbs: float
    """Sustained read bandwidth at full thread count, in GB/s (10^9 bytes)."""
    memory_budget_bytes: int
    """Memory a tree may use: weights, context and runtime together."""
    storage_budget_bytes: int
    """Space for trees kept on the pot at once."""
    threads: int
    """Threads to run trees with."""

    def ceiling_tokens_per_second(self, weight_bytes: int) -> float:
        """The fastest generation possible if every weight is read once per token."""
        if weight_bytes <= 0:
            msg = f"weight_bytes must be positive, got {weight_bytes}"
            raise ValueError(msg)
        return self.read_bandwidth_gbs * 1e9 / weight_bytes

    def fits(self, weight_bytes: int) -> bool:
        """Whether weights of this size fit the memory budget."""
        return weight_bytes <= self.memory_budget_bytes


def _require(
    table: dict[str, Any], key: str, kind: type | tuple[type, ...], where: str
) -> Any:
    if key not in table:
        msg = f"{where}: missing '{key}'"
        raise PotError(msg)
    value = table[key]
    # bool is a subclass of int, but a flag never belongs where a number does.
    if isinstance(value, bool) or not isinstance(value, kind):
        kinds = kind if isinstance(kind, tuple) else (kind,)
        expected = " or ".join(k.__name__ for k in kinds)
        msg = f"{where}: '{key}' must be {expected}, got {value!r}"
        raise PotError(msg)
    return value


def _positive(value: float, key: str, where: str) -> float:
    if value <= 0:
        msg = f"{where}: '{key}' must be positive, got {value}"
        raise PotError(msg)
    return value


def parse_pot(data: dict[str, Any], where: str = "pot") -> Pot:
    """Build a pot from a parsed profile, checking every field it needs."""
    name = _require(data, "name", str, where)
    description = _require(data, "description", str, where)
    bandwidth = _require(data, "bandwidth", dict, where)
    budget = _require(data, "budget", dict, where)

    read_gbs = _require(bandwidth, "read_gbs", (int, float), f"{where} [bandwidth]")
    memory_gib = _require(budget, "memory_gib", (int, float), f"{where} [budget]")
    storage_gib = _require(budget, "storage_gib", (int, float), f"{where} [budget]")
    threads = _require(budget, "threads", int, f"{where} [budget]")

    return Pot(
        name=name,
        description=description,
        read_bandwidth_gbs=float(
            _positive(read_gbs, "read_gbs", f"{where} [bandwidth]")
        ),
        memory_budget_bytes=int(
            _positive(memory_gib, "memory_gib", f"{where} [budget]") * GIB
        ),
        storage_budget_bytes=int(
            _positive(storage_gib, "storage_gib", f"{where} [budget]") * GIB
        ),
        threads=int(_positive(threads, "threads", f"{where} [budget]")),
    )


def load_pot(path: Path) -> Pot:
    """Load and validate a pot profile from a TOML file."""
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        msg = f"{path}: {error}"
        raise PotError(msg) from error
    return parse_pot(data, where=str(path))
