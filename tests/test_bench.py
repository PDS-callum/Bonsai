import json
from pathlib import Path
from typing import Any

import pytest
from conftest import MakeGguf
from typer.testing import CliRunner

from bonsai.bench import (
    BenchError,
    compare_with_ceiling,
    format_report,
    load_runs,
    parse_run,
)
from bonsai.cli import app
from bonsai.pot import Pot

runner = CliRunner()

POT = Pot(
    name="test",
    description="a test pot",
    read_bandwidth_gbs=10.0,
    memory_budget_bytes=2 * 1024**3,
    storage_budget_bytes=4 * 1024**3,
    threads=4,
)


def record(**overrides: Any) -> dict[str, Any]:
    """A llama-bench JSONL record, trimmed to the fields Bonsai reads."""
    data: dict[str, Any] = {
        "build_commit": "b1b770d90",
        "model_filename": "/home/pi/trees/small-q4_k_m.gguf",
        "model_type": "qwen2 1B Q4_K - Medium",
        "model_size": 500_000_000,
        "n_threads": 4,
        "n_prompt": 0,
        "n_gen": 128,
        "avg_ts": 10.0,
        "stddev_ts": 0.5,
        "samples_ts": [9.5, 10.0, 10.5],
    }
    data.update(overrides)
    return data


def write_jsonl(path: Path, *records: dict[str, Any]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    return path


def test_parse_generation_run() -> None:
    run = parse_run(record())

    assert run.model_file == "small-q4_k_m.gguf"
    assert run.model_bytes == 500_000_000
    assert run.test == "tg128"
    assert run.is_generation
    assert run.samples == (9.5, 10.0, 10.5)


def test_parse_prompt_run() -> None:
    run = parse_run(record(n_prompt=128, n_gen=0))

    assert run.test == "pp128"
    assert not run.is_generation


def test_combined_test_is_named_and_not_generation() -> None:
    run = parse_run(record(n_prompt=512, n_gen=128))

    assert run.test == "pp512+tg128"
    assert not run.is_generation


def test_windows_paths_give_the_file_name() -> None:
    run = parse_run(record(model_filename="C:\\trees\\small.gguf"))

    assert run.model_file == "small.gguf"


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"avg_ts": "fast"}, "'avg_ts' must be int or float"),
        ({"model_size": 1.5}, "'model_size' must be int"),
        ({"samples_ts": [1.0, "x"]}, "must hold numbers"),
    ],
)
def test_invalid_records_are_reported(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(BenchError, match=message):
        parse_run(record(**overrides))


def test_missing_field_is_reported() -> None:
    data = record()
    del data["n_gen"]

    with pytest.raises(BenchError, match="missing 'n_gen'"):
        parse_run(data)


def test_load_runs_skips_blank_lines(tmp_path: Path) -> None:
    path = write_jsonl(tmp_path / "bench.jsonl", record(), record(n_threads=2))
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")

    assert [run.threads for run in load_runs(path)] == [4, 2]


def test_load_runs_reports_the_bad_line(tmp_path: Path) -> None:
    path = tmp_path / "bench.jsonl"
    path.write_text(json.dumps(record()) + "\n{oops\n", encoding="utf-8")

    with pytest.raises(BenchError, match=r"bench\.jsonl:2"):
        load_runs(path)


def test_load_runs_rejects_non_objects(tmp_path: Path) -> None:
    path = tmp_path / "bench.jsonl"
    path.write_text("[1, 2]\n", encoding="utf-8")

    with pytest.raises(BenchError, match="expected a JSON object"):
        load_runs(path)


def test_compare_keeps_generation_runs_only() -> None:
    runs = [parse_run(record()), parse_run(record(n_prompt=128, n_gen=0))]

    (comparison,) = compare_with_ceiling(POT, runs)

    # 10 GB/s over 500 MB is a ceiling of 20 tokens/s; 10 tokens/s is half.
    assert comparison.ceiling_tokens_per_second == pytest.approx(20.0)
    assert comparison.fraction_of_ceiling == pytest.approx(0.5)
    assert comparison.effective_bandwidth_gbs == pytest.approx(5.0)


def test_compare_uses_bytes_read_per_token_when_known() -> None:
    runs = [parse_run(record())]

    (comparison,) = compare_with_ceiling(POT, runs, {"small-q4_k_m.gguf": 250_000_000})

    # Reading 250 MB per token at 10 GB/s allows 40 tokens/s.
    assert comparison.bytes_per_token == 250_000_000
    assert comparison.ceiling_tokens_per_second == pytest.approx(40.0)
    assert comparison.effective_bandwidth_gbs == pytest.approx(2.5)


def test_report_orders_trees_by_size() -> None:
    runs = [
        parse_run(record(model_filename="big.gguf", model_size=1_000_000_000)),
        parse_run(record(model_filename="small.gguf", n_prompt=64, n_gen=0)),
        parse_run(record(model_filename="small.gguf")),
    ]

    lines = format_report(POT, runs, {"small.gguf": 400_000_000}).splitlines()

    assert lines[0].startswith("| Tree |")
    assert lines[2].startswith("| small.gguf |")
    assert (
        "| 400 MB | 4 | 10.0 t/s (pp64) | 10.0 ± 0.5 t/s (tg128) | 25.0 t/s "
        "| 40% | 4.0 GB/s |"
    ) in lines[2]
    assert lines[3].startswith("| big.gguf |")
    assert "| * | 4 | - | 10.0 ± 0.5 t/s (tg128) | 10.0 t/s | 100% |" in lines[3]


def test_report_has_a_row_per_thread_count_and_length() -> None:
    runs = [
        parse_run(record(n_threads=4, n_gen=1024)),
        parse_run(record(n_threads=4)),
        parse_run(record(n_threads=1)),
    ]

    rows = format_report(POT, runs).splitlines()[2:]

    assert [row.split(" | ")[4] for row in rows] == ["1", "4", "4"]
    assert "(tg128)" in rows[1]
    assert "(tg1024)" in rows[2]


def test_report_marks_missing_generation() -> None:
    runs = [parse_run(record(n_prompt=128, n_gen=0))]

    (row,) = format_report(POT, runs).splitlines()[2:]

    assert row.endswith("| - | - | - | - |")


def test_cli_report(tmp_path: Path) -> None:
    pot_file = tmp_path / "pot.toml"
    pot_file.write_text(
        'name = "test"\ndescription = "a test pot"\n'
        "[bandwidth]\nread_gbs = 10.0\n"
        "[budget]\nmemory_gib = 2\nstorage_gib = 4\nthreads = 4\n",
        encoding="utf-8",
    )
    results = write_jsonl(tmp_path / "bench.jsonl", record())

    result = runner.invoke(app, ["bench", "report", str(pot_file), str(results)])

    assert result.exit_code == 0, result.output
    assert "| small-q4_k_m.gguf |" in result.output


def test_cli_report_rejects_bad_results(tmp_path: Path) -> None:
    pot_file = Path(__file__).resolve().parents[1] / "configs/pots/pi5-4gb.toml"
    results = tmp_path / "bench.jsonl"
    results.write_text("not json\n", encoding="utf-8")

    result = runner.invoke(app, ["bench", "report", str(pot_file), str(results)])

    assert result.exit_code == 1
    assert "bench.jsonl:1" in result.output


def test_cli_report_counts_bytes_from_trees(
    tmp_path: Path, make_gguf: MakeGguf
) -> None:
    pot_file = Path(__file__).resolve().parents[1] / "configs/pots/pi5-4gb.toml"
    make_gguf(
        "small-q4_k_m.gguf", **{"token_embd.weight": 250_000, "output.weight": 250_000}
    )
    results = write_jsonl(tmp_path / "bench.jsonl", record(), record())

    result = runner.invoke(
        app,
        ["bench", "report", str(pot_file), str(results), "--trees", str(tmp_path)],
    )

    assert result.exit_code == 0, result.output
    assert "| 1 MB | 4 |" in result.output
