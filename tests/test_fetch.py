import gzip
import hashlib
from pathlib import Path
from typing import Any

import pytest
from conftest import plain
from typer.testing import CliRunner

from bonsai.cli import app
from bonsai.fetch import (
    FetchError,
    destination,
    fetch,
    load_manifest,
    parse_manifest,
)

runner = CliRunner()


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def manifest(**overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "name": "test",
        "kind": "data",
        "license": "mit",
        "base_url": "https://example.org/fixed",
        "files": [{"name": "a.txt", "sha256": "0" * 64}],
    }
    data.update(overrides)
    return data


@pytest.fixture
def source(tmp_path: Path) -> Path:
    """A directory standing in for the remote server, served over file://."""
    directory = tmp_path / "remote"
    directory.mkdir()
    return directory


def served(source: Path, name: str, data: bytes) -> dict[str, Any]:
    (source / name).write_bytes(data)
    return {"name": name, "url": (source / name).as_uri(), "sha256": sha(data)}


# Manifests


def test_files_default_to_base_url() -> None:
    (spec,) = parse_manifest(manifest()).files

    assert spec.url == "https://example.org/fixed/a.txt"


def test_file_url_overrides_base_url() -> None:
    files = [{"name": "a.txt", "url": "https://x.org/a", "sha256": "f" * 64}]

    (spec,) = parse_manifest(manifest(files=files)).files

    assert spec.url == "https://x.org/a"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"kind": "models"}, "'kind' must be one of seeds, data"),
        ({"files": []}, "no files listed"),
        ({"files": ["a.txt"]}, "expected a table"),
        ({"files": [{"name": "../a", "sha256": "0" * 64}]}, "plain file name"),
        ({"files": [{"name": "a", "sha256": "xyz"}]}, "64 hex digits"),
        ({"files": [{"name": "a", "sha256": "0" * 64, "gunzip": 1}]}, "true or false"),
        ({"base_url": None}, "needs 'url'"),
    ],
)
def test_invalid_manifests_are_reported(
    overrides: dict[str, Any], message: str
) -> None:
    with pytest.raises(FetchError, match=message):
        parse_manifest(manifest(**overrides))


def test_invalid_toml_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "m.toml"
    path.write_text("name = ", encoding="utf-8")

    with pytest.raises(FetchError, match=r"m\.toml"):
        load_manifest(path)


def test_shipped_manifests_load() -> None:
    repo = Path(__file__).resolve().parents[1]
    for path in [
        *repo.glob("configs/seeds/*.toml"),
        *repo.glob("configs/data/*.toml"),
    ]:
        assert load_manifest(path).files


def test_destination_is_kind_then_name(tmp_path: Path) -> None:
    loaded = parse_manifest(manifest(kind="seeds", name="qwen"))

    assert destination(loaded, tmp_path) == tmp_path / "seeds" / "qwen"


# Fetching


def test_fetch_downloads_and_verifies(tmp_path: Path, source: Path) -> None:
    loaded = parse_manifest(manifest(files=[served(source, "a.txt", b"hello")]))
    logged: list[str] = []

    (path,) = fetch(loaded, tmp_path / "home", log=logged.append)

    assert path.read_bytes() == b"hello"
    assert logged == ["fetch  a.txt", "ok     a.txt"]
    assert not list(path.parent.glob("*.part"))


def test_fetch_skips_verified_files(tmp_path: Path, source: Path) -> None:
    loaded = parse_manifest(manifest(files=[served(source, "a.txt", b"hello")]))
    fetch(loaded, tmp_path, log=lambda _: None)
    (source / "a.txt").unlink()  # a second download would now fail
    logged: list[str] = []

    fetch(loaded, tmp_path, log=logged.append)

    assert logged == ["ok     a.txt (already present)"]


def test_fetch_rejects_a_bad_checksum(tmp_path: Path, source: Path) -> None:
    entry = served(source, "a.txt", b"hello")
    entry["sha256"] = sha(b"something else")
    loaded = parse_manifest(manifest(files=[entry]))

    with pytest.raises(FetchError, match="checksum mismatch"):
        fetch(loaded, tmp_path, log=lambda _: None)
    assert not list(destination(loaded, tmp_path).iterdir())


def test_fetch_reports_download_failures(tmp_path: Path, source: Path) -> None:
    entry = {"name": "gone.txt", "url": (source / "gone.txt").as_uri()}
    entry["sha256"] = "0" * 64
    loaded = parse_manifest(manifest(files=[entry]))

    with pytest.raises(FetchError, match="download failed"):
        fetch(loaded, tmp_path, log=lambda _: None)


def test_fetch_can_unzip(tmp_path: Path, source: Path) -> None:
    entry = served(source, "t.raw.gz", gzip.compress(b"text"))
    entry["gunzip"] = True
    loaded = parse_manifest(manifest(files=[entry]))

    paths = fetch(loaded, tmp_path, log=lambda _: None)

    assert [p.name for p in paths] == ["t.raw.gz", "t.raw"]
    assert paths[1].read_bytes() == b"text"


def test_fetch_only_some_files(tmp_path: Path, source: Path) -> None:
    files = [served(source, "a.txt", b"a"), served(source, "b.txt", b"b")]
    loaded = parse_manifest(manifest(files=files))

    paths = fetch(loaded, tmp_path, only=["b.txt"], log=lambda _: None)

    assert [p.name for p in paths] == ["b.txt"]


def test_fetch_rejects_unknown_files(tmp_path: Path) -> None:
    with pytest.raises(FetchError, match=r"not in the manifest: c\.txt"):
        fetch(parse_manifest(manifest()), tmp_path, only=["c.txt"])


# The CLI


def test_cli_fetch(
    tmp_path: Path, source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BONSAI_HOME", str(tmp_path / "home"))
    entry = served(source, "a.txt", b"hello")
    path = tmp_path / "m.toml"
    path.write_text(
        'name = "test"\nkind = "data"\nlicense = "mit"\n'
        f'[[files]]\nname = "a.txt"\nurl = "{entry["url"]}"\n'
        f'sha256 = "{entry["sha256"]}"\n',
        encoding="utf-8",
    )

    result = runner.invoke(app, ["fetch", str(path)])

    assert result.exit_code == 0, result.output
    assert (tmp_path / "home" / "data" / "test" / "a.txt").is_file()


def test_cli_fetch_reports_errors(tmp_path: Path) -> None:
    path = tmp_path / "m.toml"
    path.write_text('name = "test"\n', encoding="utf-8")

    result = runner.invoke(app, ["fetch", str(path)])

    assert result.exit_code == 1
    assert "missing 'kind'" in plain(result.output)
