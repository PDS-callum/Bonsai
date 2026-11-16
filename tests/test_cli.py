from conftest import plain
from typer.testing import CliRunner

from bonsai import __version__
from bonsai.cli import app

runner = CliRunner()


def test_version_prints_package_version() -> None:
    result = runner.invoke(app, ["--version"])

    assert result.exit_code == 0
    assert plain(result.stdout).strip() == f"bonsai {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])

    assert "Usage: bonsai" in plain(result.output)
