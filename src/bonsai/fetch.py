"""Fetch pinned, checksummed files: seeds to plant, and data to measure with.

A manifest (TOML, under ``configs/seeds`` or ``configs/data``) lists files by
URL and SHA-256. URLs point at fixed revisions, so a manifest always means the
same bytes; the checksum proves it. Files land outside the repository, in
``$BONSAI_HOME/<kind>/<name>/`` (``~/bonsai`` by default)::

    name = "qwen2.5-0.5b-instruct"
    kind = "seeds"                       # or "data"
    license = "apache-2.0"
    base_url = "https://huggingface.co/<repo>/resolve/<revision>"

    [[files]]
    name = "model-q4_0.gguf"             # fetched from base_url/name
    sha256 = "..."

    [[files]]
    name = "wiki.test.raw.gz"
    url = "https://example.org/fixed/wiki.test.raw.gz"   # overrides base_url
    sha256 = "..."
    gunzip = true                        # also write wiki.test.raw

Downloads go to a ``.part`` file and are moved into place only once their
checksum matches, so an interrupted run never leaves a file that looks done.
"""

from __future__ import annotations

import gzip
import hashlib
import os
import shutil
import tomllib
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import IO, Any

from bonsai._validate import require

CHUNK = 1024 * 1024
KINDS = ("seeds", "data")

Opener = Callable[[urllib.request.Request], IO[bytes]]


class FetchError(ValueError):
    """A manifest is invalid, or a file failed to download or verify."""


@dataclass(frozen=True)
class FileSpec:
    name: str
    url: str
    sha256: str
    gunzip: bool = False


@dataclass(frozen=True)
class Manifest:
    name: str
    kind: str
    license: str
    files: tuple[FileSpec, ...]


def _field(table: dict[str, Any], key: str, kind: type, where: str) -> Any:
    return require(table, key, kind, where, FetchError)


def parse_manifest(data: dict[str, Any], where: str = "manifest") -> Manifest:
    kind = _field(data, "kind", str, where)
    if kind not in KINDS:
        msg = f"{where}: 'kind' must be one of {', '.join(KINDS)}, got {kind!r}"
        raise FetchError(msg)
    base_url = data.get("base_url")
    files = []
    for number, entry in enumerate(_field(data, "files", list, where), 1):
        place = f"{where} [[files]] #{number}"
        if not isinstance(entry, dict):
            msg = f"{place}: expected a table"
            raise FetchError(msg)
        name = _field(entry, "name", str, place)
        if Path(name).name != name or name in {"", ".", ".."}:
            msg = f"{place}: 'name' must be a plain file name, got {name!r}"
            raise FetchError(msg)
        url = entry.get("url") or (f"{base_url}/{name}" if base_url else None)
        if not isinstance(url, str):
            msg = f"{place}: needs 'url', or 'base_url' in the manifest"
            raise FetchError(msg)
        sha256 = _field(entry, "sha256", str, place).lower()
        if len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            msg = f"{place}: 'sha256' must be 64 hex digits"
            raise FetchError(msg)
        gunzip = entry.get("gunzip", False)
        if not isinstance(gunzip, bool):
            msg = f"{place}: 'gunzip' must be true or false"
            raise FetchError(msg)
        files.append(FileSpec(name, url, sha256, gunzip))
    if not files:
        msg = f"{where}: no files listed"
        raise FetchError(msg)
    return Manifest(
        name=_field(data, "name", str, where),
        kind=kind,
        license=_field(data, "license", str, where),
        files=tuple(files),
    )


def load_manifest(path: Path) -> Manifest:
    try:
        with path.open("rb") as file:
            data = tomllib.load(file)
    except tomllib.TOMLDecodeError as error:
        msg = f"{path}: {error}"
        raise FetchError(msg) from error
    return parse_manifest(data, str(path))


def bonsai_home() -> Path:
    """Where Bonsai keeps large files outside the repository."""
    return Path(os.environ.get("BONSAI_HOME", Path.home() / "bonsai"))


def destination(manifest: Manifest, home: Path | None = None) -> Path:
    return (home or bonsai_home()) / manifest.kind / manifest.name


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def _download(spec: FileSpec, target: Path, opener: Opener) -> None:
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(spec.url, headers={"User-Agent": "bonsai"})
    try:
        with opener(request) as response, partial.open("wb") as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
                digest.update(chunk)
    except OSError as error:
        partial.unlink(missing_ok=True)
        msg = f"{spec.name}: download failed: {error}"
        raise FetchError(msg) from error
    if digest.hexdigest() != spec.sha256:
        partial.unlink()
        msg = f"{spec.name}: checksum mismatch (got {digest.hexdigest()})"
        raise FetchError(msg)
    partial.replace(target)


def _gunzip(source: Path) -> Path:
    target = source.with_suffix("")
    partial = target.with_name(target.name + ".part")
    with gzip.open(source, "rb") as compressed, partial.open("wb") as out:
        shutil.copyfileobj(compressed, out, CHUNK)
    partial.replace(target)
    return target


def fetch(
    manifest: Manifest,
    home: Path | None = None,
    *,
    only: Sequence[str] = (),
    opener: Opener = urllib.request.urlopen,
    log: Callable[[str], None] = print,
) -> list[Path]:
    """Download the manifest's files (or just ``only``), skipping verified ones.

    Returns the paths of the files, and of their unpacked copies where
    ``gunzip`` is set.
    """
    known = {spec.name for spec in manifest.files}
    unknown = sorted(set(only) - known)
    if unknown:
        msg = f"not in the manifest: {', '.join(unknown)}"
        raise FetchError(msg)

    directory = destination(manifest, home)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for spec in manifest.files:
        if only and spec.name not in only:
            continue
        target = directory / spec.name
        if target.exists() and sha256_of(target) == spec.sha256:
            log(f"ok     {spec.name} (already present)")
        else:
            log(f"fetch  {spec.name}")
            _download(spec, target, opener)
            log(f"ok     {spec.name}")
        paths.append(target)
        if spec.gunzip:
            unpacked = target.with_suffix("")
            if not unpacked.exists():
                _gunzip(target)
                log(f"unzip  {unpacked.name}")
            paths.append(unpacked)
    return paths
