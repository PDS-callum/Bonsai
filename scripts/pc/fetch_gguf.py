#!/usr/bin/env python3
"""Download the GGUF files listed in a seed manifest, checking each checksum.

Files come from Hugging Face at the exact revision the manifest pins, and land
in ``$BONSAI_HOME/seeds/<name>/`` (``~/bonsai`` by default), outside the repo.
A file already present with the right checksum is skipped. Downloads go to a
``.part`` file first, so an interrupted run never leaves a file that looks
complete.

Usage: fetch_gguf.py <manifest.toml> [file-name ...]
       (with no file names, every file in the manifest is fetched)
"""

from __future__ import annotations

import hashlib
import os
import sys
import tomllib
import urllib.request
from pathlib import Path
from typing import Any

CHUNK = 1024 * 1024


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, target: Path, expected_sha256: str) -> None:
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    request = urllib.request.Request(url, headers={"User-Agent": "bonsai"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as out:
        total = int(response.headers.get("Content-Length", 0))
        done = 0
        while chunk := response.read(CHUNK):
            out.write(chunk)
            digest.update(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {done / total:6.1%} of {total / 1e6:.0f} MB", end="")
    print()
    if digest.hexdigest() != expected_sha256:
        partial.unlink()
        msg = f"checksum mismatch for {target.name}"
        raise SystemExit(msg)
    partial.replace(target)


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    manifest_path = Path(sys.argv[1])
    wanted = set(sys.argv[2:])
    with manifest_path.open("rb") as file:
        manifest: dict[str, Any] = tomllib.load(file)

    home = Path(os.environ.get("BONSAI_HOME", Path.home() / "bonsai"))
    destination = home / "seeds" / manifest["name"]
    destination.mkdir(parents=True, exist_ok=True)

    files = [f for f in manifest["files"] if not wanted or f["name"] in wanted]
    unknown = wanted - {f["name"] for f in manifest["files"]}
    if unknown:
        msg = f"not in {manifest_path}: {', '.join(sorted(unknown))}"
        raise SystemExit(msg)

    for entry in files:
        target = destination / entry["name"]
        if target.exists() and sha256_of(target) == entry["sha256"]:
            print(f"ok    {entry['name']} (already present)")
            continue
        url = (
            f"https://huggingface.co/{manifest['repo']}/resolve/"
            f"{manifest['revision']}/{entry['name']}"
        )
        print(f"fetch {entry['name']}")
        download(url, target, entry["sha256"])
        print(f"ok    {entry['name']}")

    print(f"files in {destination}")


if __name__ == "__main__":
    main()
