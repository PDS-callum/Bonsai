import json
import sqlite3
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from conftest import plain
from typer.testing import CliRunner

from bonsai.cli import app
from bonsai.journal import (
    JournalError,
    build,
    connect,
    find_tree,
    format_frontier,
    frontier,
    lineage,
    quant_of,
)

runner = CliRunner()
_opened: list[sqlite3.Connection] = []


def open_db(path: Path) -> sqlite3.Connection:
    connection = connect(path)
    _opened.append(connection)
    return connection


@pytest.fixture(autouse=True)
def _close_connections() -> Iterator[None]:
    yield
    while _opened:
        _opened.pop().close()


SHA = {name: name[0] * 64 for name in ("a", "b", "c", "d", "e", "f")}
FETCHED_Q4 = "a" * 64  # seed-gguf-q4_0.gguf
FETCHED_F16 = "b" * 64  # seed-gguf-f16.gguf
BASE = "c" * 64  # small-bf16.gguf
PLANTED_Q4 = "d" * 64  # small-q4_0.gguf
PRUNED_BASE = "e" * 64  # small-cut2-bf16.gguf
PRUNED_Q4 = "f" * 64  # small-cut2-q4_0.gguf


def write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def bench_record(file: str, tps: float, **overrides: Any) -> dict[str, Any]:
    data: dict[str, Any] = {
        "build_commit": "b1b770d90",
        "model_filename": f"/home/pi/jobs/x/inputs/{file}",
        "model_type": "test",
        "model_size": 1000,
        "n_threads": 3,
        "n_prompt": 0,
        "n_gen": 128,
        "avg_ts": tps,
        "stddev_ts": 0.4,
        "samples_ts": [tps] * 4,
        "repack": False,
        "test_time": "2026-11-18T10:00:00Z",
    }
    data.update(overrides)
    return data


def quality(tree: str, sha: str, gsm8k: float, kld: float | None) -> str:
    return json.dumps({
        "tree": tree, "sha256": sha, "bytes": 1000,
        "measured_at": "2026-11-18T10:00:00Z",
        "perplexity": {"value": 15.0, "stderr": 0.1, "context": 512},
        "kl_divergence": None if kld is None else {
            "mean": kld, "mean_stderr": 0.001, "same_top_pct": 90.0,
            "same_top_pct_stderr": 0.4,
        },
        "arc": {"accuracy_pct": 45.0, "stderr_pct": 1.7},
        "gsm8k": {"accuracy": gsm8k, "ci95": [gsm8k - 0.05, gsm8k + 0.05]},
    })  # fmt: skip


def planting(seed: str, base: tuple[str, str], q4: tuple[str, str]) -> str:
    return json.dumps({
        "seed": seed, "base": base[0],
        "trees": [
            {"file": base[0], "type": "bf16", "bytes": 2000, "sha256": base[1]},
            {"file": q4[0], "type": "q4_0", "bytes": 1000, "sha256": q4[1]},
        ],
    })  # fmt: skip


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    """A repository holding one of every kind of record."""
    root = tmp_path / "repo"
    seeds = root / "configs" / "seeds"
    write(seeds / "gguf.toml", (
        'name = "seed-gguf"\nkind = "seeds"\nlicense = "apache-2.0"\n'
        'base_url = "https://example.org/r/abc"\n'
        f'[[files]]\nname = "seed-gguf-q4_0.gguf"\nsha256 = "{FETCHED_Q4}"\n'
        f'[[files]]\nname = "seed-gguf-f16.gguf"\nsha256 = "{FETCHED_F16}"\n'
    ))  # fmt: skip
    write(seeds / "small.toml", (
        'name = "small"\nkind = "seeds"\nlicense = "mit"\n'
        'base_url = "https://example.org/r/def"\n'
        f'[[files]]\nname = "model.safetensors"\nsha256 = "{"0" * 64}"\n'
    ))  # fmt: skip
    results = root / "results"
    write(
        results / "prunings" / "small-cut2" / "prune.json",
        json.dumps({"name": "small-cut2", "seed": "small", "removed_layers": [3, 5]}),
    )
    write(
        results / "plantings" / "small" / "2026-11-18" / "planting.json",
        planting("small", ("small-bf16.gguf", BASE), ("small-q4_0.gguf", PLANTED_Q4)),
    )
    write(results / "plantings" / "small-cut2" / "2026-11-18" / "planting.json",
          planting("small-cut2", ("small-cut2-bf16.gguf", PRUNED_BASE),
                   ("small-cut2-q4_0.gguf", PRUNED_Q4)))  # fmt: skip
    q = results / "quality"
    write(
        q / "seed-gguf-q4_0" / "quality.json",
        quality("seed-gguf-q4_0.gguf", FETCHED_Q4, 0.40, 0.03),
    )
    write(
        q / "small-q4_0" / "quality.json",
        quality("small-q4_0.gguf", PLANTED_Q4, 0.10, 0.10),
    )
    write(
        q / "small-cut2-q4_0" / "quality.json",
        quality("small-cut2-q4_0.gguf", PRUNED_Q4, 0.05, 0.30),
    )
    # An older job: no checksums, so trees are matched by file name.
    old = results / "jobs" / "old" / "results"
    write(old / "bench.jsonl", "\n".join(json.dumps(r) for r in [
        bench_record("seed-gguf-q4_0.gguf", 30.0),
        bench_record("seed-gguf-q4_0.gguf", 100.0, n_gen=0, n_prompt=128),
        bench_record("seed-gguf-q4_0.gguf", 28.0, n_threads=4, repack=True),
        bench_record("mystery.gguf", 99.0),
    ]))  # fmt: skip
    write(old / "peak-rss.tsv", "seed-gguf-q4_0.gguf\t500000000\nmystery.gguf\t1\n")
    # A newer job, with checksums.
    new = results / "jobs" / "new" / "results"
    write(new / "bench.jsonl", "\n".join(json.dumps(r) for r in [
        bench_record("small-q4_0.gguf", 50.0),
        bench_record("small-cut2-q4_0.gguf", 60.0),
    ]))  # fmt: skip
    write(
        new / "trees.sha256",
        f"{PLANTED_Q4}  small-q4_0.gguf\n{PRUNED_Q4} *small-cut2-q4_0.gguf\n\n",
    )
    write(new / "peak-rss.tsv", "small-q4_0.gguf\t300000000\n")
    return root


@pytest.fixture
def db(repo: Path, tmp_path: Path) -> Path:
    path = tmp_path / "journal.sqlite"
    build(repo, path)
    return path


def test_quant_is_the_last_part_of_the_name() -> None:
    assert quant_of("qwen2.5-0.5b-instruct-Q4_K_M.gguf") == "q4_k_m"


def test_build_counts_everything(repo: Path, tmp_path: Path) -> None:
    report = build(repo, tmp_path / "j.sqlite")

    assert (report.seeds, report.trees) == (3, 6)
    assert report.unmatched == ["results/jobs/old/results/mystery.gguf"]
    assert not list(tmp_path.glob("*.part"))


def test_rebuilding_replaces_the_journal(repo: Path, db: Path) -> None:
    build(repo, db)

    trees = open_db(db).execute("SELECT COUNT(*) FROM trees").fetchone()[0]
    assert trees == 6


def test_lineage_of_a_pruned_tree_reaches_the_seed(db: Path) -> None:
    steps = lineage(open_db(db), PRUNED_Q4)

    assert steps[0].startswith(
        "small-cut2-q4_0.gguf  (ffffffffffff)  quantised to Q4_0"
    )
    assert "converted to BF16" in steps[1]
    assert steps[2] == "seed small-cut2  pruned from small: layers 3, 5 removed"
    assert steps[3] == "seed small  (weights, configs/seeds/small.toml)"


def test_lineage_of_a_fetched_tree(db: Path) -> None:
    steps = lineage(open_db(db), FETCHED_Q4)

    assert steps == [
        f"seed-gguf-q4_0.gguf  ({FETCHED_Q4[:12]})  fetched as GGUF",
        "seed seed-gguf  (gguf, configs/seeds/gguf.toml)",
    ]


def test_lineage_notes_a_missing_seed(repo: Path, tmp_path: Path) -> None:
    write(
        repo / "results" / "prunings" / "x" / "prune.json",
        json.dumps({"name": "x", "seed": "gone"}),
    )
    write(
        repo / "results" / "plantings" / "x" / "planting.json",
        planting("x", ("x-bf16.gguf", "9" * 64), ("x-q4_0.gguf", "8" * 64)),
    )
    build(repo, tmp_path / "j.sqlite")

    steps = lineage(open_db(tmp_path / "j.sqlite"), "8" * 64)

    assert steps[-1] == "seed gone  (not in the journal)"


def test_quality_and_speed_are_recorded(db: Path) -> None:
    rows = (
        open_db(db)
        .execute(
            "SELECT metric, value, low, high, threads, repack, tokens FROM measurements"
            " WHERE tree = ? ORDER BY metric, value",
            (FETCHED_Q4,),
        )
        .fetchall()
    )
    found = {(r["metric"], r["value"]) for r in rows}

    assert ("gsm8k_pct", 40.0) in found
    assert ("kl_divergence", 0.03) in found
    assert ("prompt_tps", 100.0) in found
    assert ("peak_memory_bytes", 5e8) in found
    gsm8k = next(r for r in rows if r["metric"] == "gsm8k_pct")
    assert (round(gsm8k["low"]), round(gsm8k["high"])) == (35, 45)
    fast = [r for r in rows if r["metric"] == "generation_tps"]
    assert {(r["threads"], r["repack"]) for r in fast} == {(3, 0), (4, 1)}


def test_frontier_marks_trees_nothing_beats(db: Path) -> None:
    points = frontier(open_db(db), threads=3, repack=False)

    assert [(p.file, p.on_frontier) for p in points] == [
        ("small-cut2-q4_0.gguf", True),  # fastest
        ("small-q4_0.gguf", True),
        ("seed-gguf-q4_0.gguf", True),  # best maths
    ]
    assert points[2].generation_tps == 30.0  # the repacked run is not counted
    assert points[2].peak_memory_bytes == 5e8


def test_frontier_for_a_lower_is_better_metric(db: Path) -> None:
    points = frontier(open_db(db), "kl_divergence", threads=3, repack=False)

    assert [p.on_frontier for p in points] == [True, True, True]
    table = format_frontier(points, "kl_divergence")
    assert "| small-q4_0.gguf | 50.0 t/s | 300 MB | 0.1000 | yes |" in table
    assert "| small-cut2-q4_0.gguf | 60.0 t/s | - | 0.3000 | yes |" in table


def test_a_dominated_tree_is_off_the_frontier(repo: Path, tmp_path: Path) -> None:
    new = repo / "results" / "jobs" / "new" / "results"
    write(new / "bench.jsonl", json.dumps(bench_record("small-q4_0.gguf", 20.0)))
    build(repo, tmp_path / "j.sqlite")

    points = frontier(open_db(tmp_path / "j.sqlite"), threads=3, repack=False)

    assert [(p.file, p.on_frontier) for p in points] == [
        ("seed-gguf-q4_0.gguf", True),
        ("small-q4_0.gguf", False),
    ]
    assert "40.0 (35.0 to 45.0) | yes |" in format_frontier(points, "gsm8k_pct")


def test_frontier_rejects_unknown_metrics(db: Path) -> None:
    with pytest.raises(JournalError, match="metric must be one of"):
        frontier(open_db(db), "speed", threads=3, repack=False)


def test_find_tree_by_name_or_checksum(db: Path) -> None:
    connection = open_db(db)

    assert find_tree(connection, "small-q4_0.gguf")["sha256"] == PLANTED_Q4
    assert find_tree(connection, "DDDDDDDD")["file"] == "small-q4_0.gguf"
    with pytest.raises(JournalError, match="no tree matching 'ddd'"):
        find_tree(connection, "ddd")


def test_connect_needs_a_journal(tmp_path: Path) -> None:
    with pytest.raises(JournalError, match="bonsai journal build"):
        connect(tmp_path / "none.sqlite")


@pytest.mark.parametrize(
    ("path", "text", "message"),
    [
        ("results/q/quality.json", quality("x.gguf", "9" * 64, 0.1, None),
         "not in the journal"),
        ("results/p/prune.json", json.dumps({"name": "x"}), "missing 'seed'"),
        ("results/p/planting.json", json.dumps({"seed": "x"}),
         "not a planting record"),
        ("results/p/planting.json", "[1]", "expected a JSON object"),
        ("results/p/planting.json", "{", "Expecting"),
        ("results/j/bench.jsonl", "[1]", "expected a JSON object"),
        ("configs/seeds/bad.toml", 'name = "x"', "missing 'kind'"),
    ],
)  # fmt: skip
def test_bad_records_are_reported(
    repo: Path, tmp_path: Path, path: str, text: str, message: str
) -> None:
    write(repo / path, text)

    with pytest.raises(JournalError, match=message):
        build(repo, tmp_path / "j.sqlite")


# The CLI


def test_cli_build_and_query(repo: Path, tmp_path: Path) -> None:
    db = str(tmp_path / "j.sqlite")
    pot = Path(__file__).resolve().parents[1] / "configs" / "pots" / "pi5-4gb.toml"

    built = runner.invoke(app, ["journal", "build", "--db", db, "--root", str(repo)])
    trees = runner.invoke(app, ["journal", "trees", "--db", db])
    traced = runner.invoke(
        app, ["journal", "lineage", "small-cut2-q4_0.gguf", "--db", db]
    )
    best = runner.invoke(app, ["journal", "frontier", "--db", db, "--pot", str(pot)])

    assert built.exit_code == 0, built.output
    assert "3 seeds, 6 trees, " in built.output
    assert "no tree matches results/jobs/old/results/mystery.gguf" in built.output
    assert (
        "| small-cut2-q4_0.gguf | small-cut2 | quantize | 0 MB | 5 | 1 |"
        in trees.output
    )
    assert "| seed-gguf-f16.gguf | seed-gguf | fetch | - | 0 | 0 |" in trees.output
    assert traced.output.splitlines()[2].startswith("    <- seed small-cut2")
    assert "| seed-gguf-q4_0.gguf | 30.0 t/s | 500 MB | 40.0" in best.output


@pytest.mark.parametrize(
    "args",
    [
        ["journal", "trees"],
        ["journal", "lineage", "x.gguf"],
        ["journal", "frontier"],
        ["journal", "build", "--root", "."],
    ],
)
def test_cli_reports_errors(
    args: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("BONSAI_HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    write(tmp_path / "configs" / "seeds" / "bad.toml", "nonsense")

    result = runner.invoke(app, args)

    assert result.exit_code == 1
    assert "error:" in plain(result.output)
