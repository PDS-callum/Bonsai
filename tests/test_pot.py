from pathlib import Path
from typing import Any

import pytest
from conftest import MakeGguf
from typer.testing import CliRunner

from bonsai.cli import app
from bonsai.pot import GIB, Pot, PotError, load_pot, parse_pot

REPO = Path(__file__).resolve().parents[1]
PI5 = REPO / "configs" / "pots" / "pi5-4gb.toml"

runner = CliRunner()


def profile(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "test",
        "description": "a test pot",
        "bandwidth": {"read_gbs": 10.0},
        "budget": {"memory_gib": 1.5, "storage_gib": 4, "threads": 4},
    }
    data.update(overrides)
    return data


def test_parse_reads_budgets() -> None:
    pot = parse_pot(profile())

    assert pot == Pot(
        name="test",
        description="a test pot",
        read_bandwidth_gbs=10.0,
        memory_budget_bytes=int(1.5 * GIB),
        storage_budget_bytes=4 * GIB,
        threads=4,
    )


def test_ceiling_is_bandwidth_over_bytes() -> None:
    pot = parse_pot(profile())

    assert pot.ceiling_tokens_per_second(500_000_000) == pytest.approx(20.0)


def test_ceiling_rejects_empty_weights() -> None:
    pot = parse_pot(profile())

    with pytest.raises(ValueError, match="positive"):
        pot.ceiling_tokens_per_second(0)


def test_fits_compares_against_memory_budget() -> None:
    pot = parse_pot(profile())

    assert pot.fits(int(1.5 * GIB))
    assert not pot.fits(int(1.5 * GIB) + 1)


def test_missing_section_is_reported() -> None:
    data = profile()
    del data["bandwidth"]

    with pytest.raises(PotError, match="missing 'bandwidth'"):
        parse_pot(data)


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("bandwidth", "read_gbs", "fast", "must be int or float"),
        ("bandwidth", "read_gbs", 0, "must be positive"),
        ("budget", "threads", True, "must be int"),
        ("budget", "threads", 2.5, "must be int"),
        ("budget", "memory_gib", -1, "must be positive"),
    ],
)
def test_invalid_values_are_reported(
    section: str, key: str, value: object, message: str
) -> None:
    data = profile()
    data[section] = {**data[section], key: value}

    with pytest.raises(PotError, match=message):
        parse_pot(data)


def test_invalid_toml_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "broken.toml"
    path.write_text("name = ", encoding="utf-8")

    with pytest.raises(PotError, match=r"broken\.toml"):
        load_pot(path)


def test_shipped_pi_profile_loads() -> None:
    pot = load_pot(PI5)

    assert pot.name == "pi5-4gb"
    assert pot.threads == 4
    assert 5 < pot.read_bandwidth_gbs < 20


def test_cli_ceiling_counts_bytes_read_per_token(
    tmp_path: Path, make_gguf: MakeGguf
) -> None:
    pot_file = tmp_path / "pot.toml"
    pot_file.write_text(
        'name = "test"\n'
        'description = "a test pot"\n'
        "[bandwidth]\nread_gbs = 10.0\n"
        "[budget]\nmemory_gib = 1.5\nstorage_gib = 4\nthreads = 4\n",
        encoding="utf-8",
    )
    # 4 bytes per value: 2 MB of weights, of which the 1 MB output layer is
    # read per token; the embedding table is only looked up.
    weights = make_gguf(**{"token_embd.weight": 250_000, "output.weight": 250_000})

    result = runner.invoke(app, ["pot", "ceiling", str(pot_file), str(weights)])

    assert result.exit_code == 0, result.output
    assert "weights         2 MB" in result.output
    assert "read per token  1 MB" in result.output
    assert "ceiling         10000.0 tokens/s" in result.output
    assert "weights fit     yes" in result.output


def test_cli_ceiling_rejects_non_gguf(tmp_path: Path) -> None:
    weights = tmp_path / "not.gguf"
    weights.write_bytes(b"not a gguf file")

    result = runner.invoke(app, ["pot", "ceiling", str(PI5), str(weights)])

    assert result.exit_code == 1
    assert "cannot read" in result.output


def test_cli_show_reports_invalid_profile(tmp_path: Path) -> None:
    pot_file = tmp_path / "pot.toml"
    pot_file.write_text('name = "test"\n', encoding="utf-8")

    result = runner.invoke(app, ["pot", "show", str(pot_file)])

    assert result.exit_code == 1
    assert "missing 'description'" in result.output


def test_cli_show_prints_budgets() -> None:
    result = runner.invoke(app, ["pot", "show", str(PI5)])

    assert result.exit_code == 0, result.output
    assert "pi5-4gb" in result.output
    assert "GB/s" in result.output
