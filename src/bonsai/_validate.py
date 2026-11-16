"""Field checks shared by the parsers of profiles and results."""

from __future__ import annotations

from typing import Any


def require(
    table: dict[str, Any],
    key: str,
    kind: type | tuple[type, ...],
    where: str,
    error: type[ValueError],
) -> Any:
    """Return ``table[key]``, raising ``error`` if it is missing or mistyped."""
    if key not in table:
        msg = f"{where}: missing '{key}'"
        raise error(msg)
    value = table[key]
    # bool is a subclass of int, but a flag never belongs where a number does.
    if isinstance(value, bool) or not isinstance(value, kind):
        kinds = kind if isinstance(kind, tuple) else (kind,)
        expected = " or ".join(k.__name__ for k in kinds)
        msg = f"{where}: '{key}' must be {expected}, got {value!r}"
        raise error(msg)
    return value
