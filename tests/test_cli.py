import re

from typer.testing import CliRunner

from bonsai import __version__
from bonsai.cli import app

runner = CliRunner()

ANSI_ESCAPE = re.compile(r"\x1b\[[0-9;]*m")


def plain(text: str) -> str:
    """Strip terminal styling, which Rich adds when it detects colour (as in CI)."""
    return ANSI_ESCAPE.sub("", text)


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert plain(result.stdout).strip() == f"bonsai {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "Usage: bonsai" in plain(result.output)
