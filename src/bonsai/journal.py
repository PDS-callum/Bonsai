"""The journal: every tree, where it came from, and how it measured.

The repository's own records are the source of truth: seed manifests
(``configs/seeds``), planting and pruning records, quality results and the
pot's benchmark output (all under ``results/``). The journal gathers them into
one SQLite database that can be queried, and rebuilt from scratch at any time
with ``bonsai journal build``, so it is never out of step with what is
committed.

Seeds form a tree of their own: a pruned seed's parent is the seed it was cut
from. Trees (GGUF files) are identified by SHA-256. A fetched tree belongs to
its seed directly; a planted tree's parent is the full-precision tree it was
quantised from, whose seed is the one it was converted from. Following parents
from any tree therefore leads back to an original seed.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bonsai.bench import BenchError, BenchRun, load_runs
from bonsai.fetch import FetchError, Manifest, load_manifest

SCHEMA = """
CREATE TABLE seeds (
    name      TEXT PRIMARY KEY,
    kind      TEXT NOT NULL,      -- gguf, weights or pruned
    parent    TEXT REFERENCES seeds(name),
    operation TEXT,               -- how it was made from its parent
    details   TEXT,               -- JSON
    license   TEXT,
    source    TEXT NOT NULL       -- the manifest or record it came from
);
CREATE TABLE trees (
    sha256    TEXT PRIMARY KEY,
    file      TEXT NOT NULL,
    seed      TEXT NOT NULL REFERENCES seeds(name),
    quant     TEXT NOT NULL,
    bytes     INTEGER,
    parent    TEXT REFERENCES trees(sha256),
    operation TEXT NOT NULL,      -- fetch, convert or quantize
    source    TEXT NOT NULL
);
CREATE TABLE measurements (
    id          INTEGER PRIMARY KEY,
    tree        TEXT NOT NULL REFERENCES trees(sha256),
    kind        TEXT NOT NULL,    -- quality or speed
    metric      TEXT NOT NULL,
    value       REAL NOT NULL,
    stderr      REAL,
    low         REAL,
    high        REAL,
    threads     INTEGER,
    repack      INTEGER,
    tokens      INTEGER,
    measured_at TEXT,
    source      TEXT NOT NULL
);
CREATE INDEX measurements_by_tree ON measurements(tree, metric);
"""

QUALITY_METRICS = (
    "perplexity",
    "kl_divergence",
    "same_top_pct",
    "arc_pct",
    "gsm8k_pct",
)
BENCH_FILES = ("bench.jsonl", "threads.jsonl", "sustained.jsonl")


class JournalError(ValueError):
    """A record could not be read, or a tree is not in the journal."""


@dataclass
class BuildReport:
    seeds: int = 0
    trees: int = 0
    measurements: int = 0
    unmatched: list[str] = field(default_factory=list)
    """Measured files that no tree in the journal matches."""


def quant_of(file: str) -> str:
    """``qwen2.5-0.5b-instruct-q4_k_m.gguf`` -> ``q4_k_m``."""
    return file.removesuffix(".gguf").rsplit("-", 1)[-1].lower()


def _rel(path: Path, root: Path) -> str:
    return path.relative_to(root).as_posix()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        msg = f"{path}: {error}"
        raise JournalError(msg) from error
    if not isinstance(data, dict):
        msg = f"{path}: expected a JSON object"
        raise JournalError(msg)
    return data


def connect(path: Path) -> sqlite3.Connection:
    if not path.is_file():
        msg = f"no journal at {path} (run: bonsai journal build)"
        raise JournalError(msg)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    return connection


# Building


def build(root: Path, db: Path) -> BuildReport:
    """Rebuild the journal at ``db`` from the records under ``root``."""
    db.parent.mkdir(parents=True, exist_ok=True)
    part = db.with_name(db.name + ".part")
    part.unlink(missing_ok=True)
    connection = sqlite3.connect(part)
    try:
        connection.executescript(SCHEMA)
        report = _fill(connection, root)
        connection.commit()
    finally:
        connection.close()
    part.replace(db)
    return report


def _fill(connection: sqlite3.Connection, root: Path) -> BuildReport:
    report = BuildReport()
    for path in sorted((root / "configs" / "seeds").glob("*.toml")):
        report.seeds += 1
        report.trees += _add_manifest(connection, path, root)
    for path in sorted((root / "results").rglob("prune.json")):
        report.seeds += 1
        _add_pruning(connection, path, root)
    for path in sorted((root / "results").rglob("planting.json")):
        report.trees += _add_planting(connection, path, root)
    known = {row[0] for row in connection.execute("SELECT sha256 FROM trees")}
    for path in sorted((root / "results").rglob("quality.json")):
        report.measurements += _add_quality(connection, path, root, known)
    for directory in sorted({p.parent for name in BENCH_FILES
                             for p in (root / "results").rglob(name)}):  # fmt: skip
        added, unmatched = _add_bench(connection, directory, root)
        report.measurements += added
        report.unmatched += unmatched
    return report


def _add_manifest(connection: sqlite3.Connection, path: Path, root: Path) -> int:
    try:
        manifest: Manifest = load_manifest(path)
    except FetchError as error:
        raise JournalError(str(error)) from error
    trees = [spec for spec in manifest.files if spec.name.endswith(".gguf")]
    connection.execute(
        "INSERT INTO seeds (name, kind, license, source) VALUES (?, ?, ?, ?)",
        (
            manifest.name,
            "gguf" if trees else "weights",
            manifest.license,
            _rel(path, root),
        ),
    )
    connection.executemany(
        "INSERT INTO trees (sha256, file, seed, quant, operation, source)"
        " VALUES (?, ?, ?, ?, 'fetch', ?)",
        [(spec.sha256, spec.name, manifest.name, quant_of(spec.name), spec.url)
         for spec in trees],
    )  # fmt: skip
    return len(trees)


def _add_pruning(connection: sqlite3.Connection, path: Path, root: Path) -> None:
    record = _read_json(path)
    try:
        name, parent = record["name"], record["seed"]
    except KeyError as error:
        msg = f"{path}: missing {error.args[0]!r}"
        raise JournalError(msg) from error
    details = {
        key: value for key, value in record.items() if key not in ("name", "seed")
    }
    connection.execute(
        "INSERT INTO seeds (name, kind, parent, operation, details, source)"
        " VALUES (?, 'pruned', ?, 'prune', ?, ?)",
        (name, parent, json.dumps(details), _rel(path, root)),
    )


def _add_planting(connection: sqlite3.Connection, path: Path, root: Path) -> int:
    record = _read_json(path)
    try:
        seed, base, trees = record["seed"], record["base"], record["trees"]
        base_sha = next(tree["sha256"] for tree in trees if tree["file"] == base)
        rows = [
            (
                tree["sha256"], tree["file"], seed, tree["type"], tree["bytes"],
                None if tree["file"] == base else base_sha,
                "convert" if tree["file"] == base else "quantize",
                _rel(path, root),
            )
            for tree in trees
        ]  # fmt: skip
    except (KeyError, TypeError, StopIteration) as error:
        msg = f"{path}: not a planting record ({error!r})"
        raise JournalError(msg) from error
    # A tree planted twice (the same bytes) is recorded once.
    connection.executemany(
        "INSERT OR IGNORE INTO trees"
        " (sha256, file, seed, quant, bytes, parent, operation, source)"
        " VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def _add_quality(
    connection: sqlite3.Connection, path: Path, root: Path, known: set[str]
) -> int:
    record = _read_json(path)
    sha = record.get("sha256")
    if sha not in known:
        msg = f"{path}: tree {record.get('tree')} ({sha}) is not in the journal"
        raise JournalError(msg)
    if record.get("bytes"):
        connection.execute(
            "UPDATE trees SET bytes = ? WHERE sha256 = ? AND bytes IS NULL",
            (record["bytes"], sha),
        )
    rows: list[tuple[str, float, float | None, float | None, float | None]] = []
    ppl = record["perplexity"]
    rows.append(("perplexity", ppl["value"], ppl["stderr"], None, None))
    if kld := record.get("kl_divergence"):
        rows.append(("kl_divergence", kld["mean"], kld["mean_stderr"], None, None))
        same = kld["same_top_pct"]
        rows.append(("same_top_pct", same, kld["same_top_pct_stderr"], None, None))
    arc = record["arc"]
    rows.append(("arc_pct", arc["accuracy_pct"], arc["stderr_pct"], None, None))
    maths = record["gsm8k"]
    low, high = maths["ci95"]
    rows.append(("gsm8k_pct", maths["accuracy"] * 100, None, low * 100, high * 100))
    connection.executemany(
        "INSERT INTO measurements"
        " (tree, kind, metric, value, stderr, low, high, measured_at, source)"
        " VALUES (?, 'quality', ?, ?, ?, ?, ?, ?, ?)",
        [(sha, *row, record.get("measured_at"), _rel(path, root)) for row in rows],
    )
    return len(rows)


def _checksums(directory: Path) -> dict[str, str]:
    """``trees.sha256`` (``sha256sum`` output), written by newer pot jobs."""
    path = directory / "trees.sha256"
    if not path.is_file():
        return {}
    sums = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        sha, _, name = line.partition(" ")
        if sha:
            sums[name.strip().lstrip("*")] = sha
    return sums


def _add_bench(
    connection: sqlite3.Connection, directory: Path, root: Path
) -> tuple[int, list[str]]:
    sums = _checksums(directory)
    runs: list[tuple[BenchRun, str]] = []
    for name in BENCH_FILES:
        if (directory / name).is_file():
            try:
                runs += [(run, name) for run in load_runs(directory / name)]
            except BenchError as error:
                raise JournalError(str(error)) from error
    unmatched: set[str] = set()

    def resolve(file: str) -> str | None:
        if file in sums:
            return sums[file]
        matches = connection.execute(
            "SELECT sha256 FROM trees WHERE file = ?", (file,)
        ).fetchall()
        if len(matches) == 1:
            return str(matches[0][0])
        unmatched.add(f"{_rel(directory, root)}/{file}")
        return None

    rows: list[tuple[Any, ...]] = []
    settings: dict[str, tuple[int, bool | None]] = {}
    for run, name in runs:
        sha = resolve(run.model_file)
        if sha is None:
            continue
        settings.setdefault(run.model_file, (run.threads, run.repack))
        metric = "generation_tps" if run.is_generation else "prompt_tps"
        stderr = run.stddev / math.sqrt(len(run.samples)) if run.samples else None
        rows.append((
            sha, metric, run.tokens_per_second, stderr, run.threads,
            None if run.repack is None else int(run.repack),
            run.n_gen or run.n_prompt, run.test_time,
            _rel(directory / name, root),
        ))  # fmt: skip
    rss = directory / "peak-rss.tsv"
    if rss.is_file():
        for line in rss.read_text(encoding="utf-8").splitlines():
            file, _, value = line.partition("\t")
            sha = resolve(file) if value else None
            if sha is None:
                continue
            threads, repack = settings.get(file, (None, None))
            rows.append((
                sha, "peak_memory_bytes", float(value), None, threads,
                None if repack is None else int(repack), None, None, _rel(rss, root),
            ))  # fmt: skip
    connection.executemany(
        "INSERT INTO measurements"
        " (tree, kind, metric, value, stderr, threads, repack, tokens,"
        "  measured_at, source)"
        " VALUES (?, 'speed', ?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )
    return len(rows), sorted(unmatched)


# Queries


def find_tree(connection: sqlite3.Connection, query: str) -> sqlite3.Row:
    """A tree by file name or SHA-256 prefix (at least 8 hex digits)."""
    rows = connection.execute(
        "SELECT * FROM trees"
        " WHERE file = ? OR (length(?) >= 8 AND sha256 LIKE ? || '%')",
        (query, query, query.lower()),
    ).fetchall()
    if len(rows) != 1:
        found = "no tree" if not rows else f"{len(rows)} trees"
        msg = f"{found} matching {query!r}; use a file name or longer checksum"
        raise JournalError(msg)
    found_row: sqlite3.Row = rows[0]
    return found_row


def lineage(connection: sqlite3.Connection, sha256: str) -> list[str]:
    """Steps from a tree back to its original seed, one line each."""
    steps = []
    tree: sqlite3.Row | None = connection.execute(
        "SELECT * FROM trees WHERE sha256 = ?", (sha256,)
    ).fetchone()
    seed_name = tree["seed"] if tree else None
    while tree is not None:
        how = {"quantize": f"quantised to {tree['quant'].upper()}",
               "convert": f"converted to {tree['quant'].upper()}",
               "fetch": "fetched as GGUF"}[tree["operation"]]  # fmt: skip
        steps.append(f"{tree['file']}  ({tree['sha256'][:12]})  {how}")
        tree = (
            connection.execute(
                "SELECT * FROM trees WHERE sha256 = ?", (tree["parent"],)
            ).fetchone()
            if tree["parent"]
            else None
        )
    while seed_name:
        seed = connection.execute(
            "SELECT * FROM seeds WHERE name = ?", (seed_name,)
        ).fetchone()
        if seed is None:
            steps.append(f"seed {seed_name}  (not in the journal)")
            break
        if seed["operation"] == "prune":
            details = json.loads(seed["details"] or "{}")
            removed = details.get("removed_layers", [])
            steps.append(
                f"seed {seed['name']}  pruned from {seed['parent']}: "
                f"layers {', '.join(map(str, removed))} removed"
            )
        else:
            steps.append(f"seed {seed['name']}  ({seed['kind']}, {seed['source']})")
        seed_name = seed["parent"]
    return steps


@dataclass(frozen=True)
class FrontierPoint:
    file: str
    seed: str
    quant: str
    generation_tps: float
    peak_memory_bytes: float | None
    quality: float
    low: float | None
    high: float | None
    on_frontier: bool = False
    origin: str = ""
    """The original seed this tree descends from."""
    pruned: bool = False


def frontier(
    connection: sqlite3.Connection,
    metric: str = "gsm8k_pct",
    *,
    threads: int,
    repack: bool,
    tokens: int = 128,
) -> list[FrontierPoint]:
    """Every tree with both a quality score and a speed at the given settings,
    fastest first, marking those no other tree beats on both."""
    if metric not in QUALITY_METRICS:
        msg = f"metric must be one of {', '.join(QUALITY_METRICS)}"
        raise JournalError(msg)
    rows = connection.execute(
        """
        SELECT t.file, t.seed, t.quant,
               (SELECT AVG(value) FROM measurements WHERE tree = t.sha256
                  AND metric = 'generation_tps' AND threads = :threads
                  AND repack = :repack AND tokens = :tokens) AS tps,
               (SELECT MAX(value) FROM measurements WHERE tree = t.sha256
                  AND metric = 'peak_memory_bytes' AND threads = :threads
                  AND repack = :repack) AS memory,
               q.value, q.low, q.high
        FROM trees t
        JOIN measurements q ON q.tree = t.sha256 AND q.metric = :metric
        WHERE q.id = (SELECT MAX(id) FROM measurements
                      WHERE tree = t.sha256 AND metric = :metric)
        """,
        {"threads": threads, "repack": int(repack), "tokens": tokens, "metric": metric},
    ).fetchall()
    seeds = {
        row["name"]: (row["kind"], row["parent"])
        for row in connection.execute("SELECT name, kind, parent FROM seeds")
    }

    def origin(seed: str) -> tuple[str, bool]:
        pruned = False
        while seed in seeds and seeds[seed][1]:
            pruned = pruned or seeds[seed][0] == "pruned"
            seed = seeds[seed][1]
        return seed, pruned

    points = [
        FrontierPoint(r["file"], r["seed"], r["quant"], r["tps"], r["memory"],
                      r["value"], r["low"], r["high"], False, *origin(r["seed"]))
        for r in rows if r["tps"] is not None
    ]  # fmt: skip
    # Lower is better for these; flip them so "higher is better" holds.
    sign = -1 if metric in ("perplexity", "kl_divergence") else 1

    def beaten(p: FrontierPoint) -> bool:
        return any(
            q.generation_tps >= p.generation_tps
            and sign * q.quality >= sign * p.quality
            and (q.generation_tps, sign * q.quality)
            != (p.generation_tps, sign * p.quality)
            for q in points
        )

    marked = [
        FrontierPoint(**{**p.__dict__, "on_frontier": not beaten(p)}) for p in points
    ]
    return sorted(marked, key=lambda p: -p.generation_tps)


def trees_summary(connection: sqlite3.Connection) -> Iterator[sqlite3.Row]:
    yield from connection.execute(
        """
        SELECT t.file, t.seed, t.quant, t.operation, t.bytes,
               SUM(m.kind = 'quality') AS quality, SUM(m.kind = 'speed') AS speed
        FROM trees t LEFT JOIN measurements m ON m.tree = t.sha256
        GROUP BY t.sha256 ORDER BY t.seed, t.bytes
        """
    )


def format_frontier(points: Iterable[FrontierPoint], metric: str) -> str:
    lines = [
        f"| Tree | Generation | Peak memory | {metric} | Frontier |",
        "|---|---|---|---|---|",
    ]
    digits = 1 if metric.endswith("_pct") else 4
    for p in points:
        memory = f"{p.peak_memory_bytes / 1e6:.0f} MB" if p.peak_memory_bytes else "-"
        has_interval = p.low is not None and p.high is not None
        interval = f" ({p.low:.1f} to {p.high:.1f})" if has_interval else ""
        lines.append(
            f"| {p.file} | {p.generation_tps:.1f} t/s | {memory} | "
            f"{p.quality:.{digits}f}{interval} | {'yes' if p.on_frontier else ''} |"
        )
    return "\n".join(lines) + "\n"
