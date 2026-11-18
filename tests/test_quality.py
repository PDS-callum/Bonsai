import gzip
import json
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest
from conftest import plain
from typer.testing import CliRunner

from bonsai import quality_cli
from bonsai.cli import app
from bonsai.quality import (
    Settings,
    base_logits_path,
    default_server,
    format_report,
    load_results,
    measure,
    save_base_logits,
)
from bonsai.server import Reply

FIXTURES = Path(__file__).parent / "data" / "llama"

cli = CliRunner()


class FakeTools:
    """Stands in for llama-perplexity, replaying real output by mode."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> str:
        self.commands.append(list(command))
        if "--kl-divergence-base" in command and "--kl-divergence" not in command:
            base = Path(command[command.index("--kl-divergence-base") + 1])
            base.write_bytes(b"logits")
            return (FIXTURES / "perplexity.txt").read_text(encoding="utf-8")
        if "--kl-divergence" in command:
            return (FIXTURES / "kl-divergence.txt").read_text(encoding="utf-8")
        if "--multiple-choice" in command:
            return (FIXTURES / "multiple-choice.txt").read_text(encoding="utf-8")
        return (FIXTURES / "perplexity.txt").read_text(encoding="utf-8")


class FakeServer:
    """Answers 42 to everything."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def chat(self, prompt: str, *, max_tokens: int = 512, seed: int = 0) -> Reply:
        self.prompts.append(prompt)
        return Reply("Working... The answer is: 42", completion_tokens=10)


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.setenv("BONSAI_HOME", str(tmp_path / "home"))
    data = tmp_path / "data"
    data.mkdir()
    problems = [("What is 6 x 7?", "42"), ("What is 1 + 1?", "2")] * 3
    (data / "gsm8k-test.jsonl").write_text(
        "".join(
            json.dumps({"question": q, "answer": f"#### {a}"}) + "\n"
            for q, a in problems
        ),
        encoding="utf-8",
    )
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("llama-perplexity", "llama-perplexity.exe"):
        (bin_dir / name).write_text("", encoding="utf-8")
    return Settings(data_dir=data, bin_dir=bin_dir, threads=2, gsm8k_size=6)


@pytest.fixture
def trees(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "seed-f16.gguf"
    base.write_bytes(b"x" * 2000)
    small = tmp_path / "seed-q4_0.gguf"
    small.write_bytes(b"x" * 1000)
    return base, small


def fake_factory(server: FakeServer) -> Any:
    @contextmanager
    def make(tree: Path, log: Path) -> Iterator[FakeServer]:
        log.write_text("server log", encoding="utf-8")
        yield server

    return make


def test_base_logits_are_saved_once(
    settings: Settings, trees: tuple[Path, Path]
) -> None:
    tools = FakeTools()
    base, _ = trees

    first = save_base_logits(base, settings, runner=tools)
    second = save_base_logits(base, settings, runner=tools)

    assert first == second == base_logits_path(base, settings)
    assert first.read_bytes() == b"logits"
    assert len(tools.commands) == 1
    assert "--chunks" in tools.commands[0]


def test_measure_writes_results_and_logs(
    settings: Settings, trees: tuple[Path, Path], tmp_path: Path
) -> None:
    tools, server = FakeTools(), FakeServer()
    base, small = trees
    logits = save_base_logits(base, settings, runner=tools)

    results = measure(
        small,
        tmp_path / "out",
        settings,
        base=base,
        base_logits=logits,
        runner=tools,
        server_factory=fake_factory(server),
        log=lambda _: None,
    )

    folder = tmp_path / "out" / "seed-q4_0"
    assert json.loads((folder / "quality.json").read_text(encoding="utf-8")) == results
    for log in ("perplexity.log", "kl-divergence.log", "arc.log", "server.log"):
        assert (folder / log).is_file()
    assert results["perplexity"]["value"] == 17.4968
    assert results["kl_divergence"]["mean"] == 0.116853
    assert results["kl_divergence"]["base"] == "seed-f16.gguf"
    assert results["arc"]["accuracy_pct"] == 48.7917
    maths = results["gsm8k"]
    assert (maths["size"], maths["correct"]) == (6, 3)  # half the answers are 42
    assert maths["answer_rules"] == {"answer-is": 6}
    assert len(server.prompts) == 6
    with gzip.open(folder / "gsm8k-replies.jsonl.gz", "rt", encoding="utf-8") as f:
        assert len(f.readlines()) == 6


def test_the_base_is_not_compared_with_itself(
    settings: Settings, trees: tuple[Path, Path], tmp_path: Path
) -> None:
    tools = FakeTools()
    base, _ = trees

    results = measure(
        base,
        tmp_path / "out",
        settings,
        base=base,
        base_logits=save_base_logits(base, settings, runner=tools),
        runner=tools,
        server_factory=fake_factory(FakeServer()),
        log=lambda _: None,
    )

    assert results["kl_divergence"] is None
    assert not any("--kl-divergence" in c for c in tools.commands[1:])


def test_report_tables_trees_by_size(
    settings: Settings, trees: tuple[Path, Path], tmp_path: Path
) -> None:
    tools = FakeTools()
    base, small = trees
    logits = save_base_logits(base, settings, runner=tools)
    for tree in (base, small):
        measure(
            tree,
            tmp_path / "out",
            settings,
            base=base,
            base_logits=logits,
            runner=tools,
            server_factory=fake_factory(FakeServer()),
            log=lambda _: None,
        )

    rows = format_report(load_results(tmp_path / "out")).splitlines()[2:]

    assert rows[0].startswith(
        "| seed-q4_0.gguf | 0 MB | 17.50 ± 0.14 | 0.1169 | 82.6% "
    )
    assert rows[0].endswith("| 48.8% ± 1.7 | 50.0% (18.8% to 81.2%) |")
    assert rows[1].startswith("| seed-f16.gguf | 0 MB | 17.50 ± 0.14 | - | - |")


# The CLI


@pytest.fixture
def data_manifest(tmp_path: Path, settings: Settings) -> Path:
    path = tmp_path / "data.toml"
    path.write_text(
        'name = "q"\nkind = "data"\nlicense = "mit"\n'
        f'[[files]]\nname = "a"\nurl = "file:///a"\nsha256 = "{"0" * 64}"\n',
        encoding="utf-8",
    )
    return path


def test_cli_run_measures_each_tree(
    tmp_path: Path,
    trees: tuple[Path, Path],
    data_manifest: Path,
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "home" / "data" / "q").mkdir(parents=True)
    monkeypatch.setattr(quality_cli, "default_bin_dir", lambda: settings.bin_dir)
    calls: list[tuple[str, Any]] = []

    def save(base: Path, used: Settings) -> Path:
        calls.append(("save", used.threads))
        return tmp_path / "base.kld"

    def run_one(tree: Path, out: Path, used: Settings, **kw: Any) -> dict[str, Any]:
        calls.append(("measure", tree.name))
        return {}

    monkeypatch.setattr(quality_cli, "save_base_logits", save)
    monkeypatch.setattr(quality_cli, "measure", run_one)
    monkeypatch.setattr(quality_cli, "load_results", lambda out: [])
    base, small = trees

    result = cli.invoke(
        app,
        [
            "quality", "run", str(small), str(base), "--base", str(base),
            "--out", str(tmp_path / "out"), "--data", str(data_manifest),
            "--threads", "3",
        ],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert calls == [
        ("save", 3),
        ("measure", "seed-q4_0.gguf"),
        ("measure", "seed-f16.gguf"),
    ]


def test_cli_run_needs_fetched_data(
    trees: tuple[Path, Path], data_manifest: Path, tmp_path: Path
) -> None:
    result = cli.invoke(
        app,
        [
            "quality", "run", str(trees[1]),
            "--out", str(tmp_path / "out"), "--data", str(data_manifest),
        ],
    )  # fmt: skip

    assert result.exit_code == 1
    assert "run: bonsai fetch" in plain(result.output)


def test_cli_run_reports_a_bad_manifest(
    trees: tuple[Path, Path], tmp_path: Path
) -> None:
    bad = tmp_path / "bad.toml"
    bad.write_text('name = "q"\n', encoding="utf-8")

    result = cli.invoke(
        app,
        ["quality", "run", str(trees[1]), "--out", str(tmp_path), "--data", str(bad)],
    )

    assert result.exit_code == 1
    assert "missing 'kind'" in plain(result.output)


def test_cli_run_reports_tool_errors(
    tmp_path: Path,
    trees: tuple[Path, Path],
    data_manifest: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "home" / "data" / "q").mkdir(parents=True)
    monkeypatch.setattr(quality_cli, "default_bin_dir", lambda: tmp_path / "nowhere")

    result = cli.invoke(
        app,
        [
            "quality", "run", str(trees[1]), "--base", str(trees[0]),
            "--out", str(tmp_path / "out"), "--data", str(data_manifest),
        ],
    )  # fmt: skip

    assert result.exit_code == 1
    assert "llama-perplexity not found" in plain(result.output)


def test_cli_report(
    settings: Settings, trees: tuple[Path, Path], tmp_path: Path
) -> None:
    measure(
        trees[1],
        tmp_path / "out",
        settings,
        runner=FakeTools(),
        server_factory=fake_factory(FakeServer()),
        log=lambda _: None,
    )

    result = cli.invoke(app, ["quality", "report", str(tmp_path / "out")])

    assert result.exit_code == 0, result.output
    assert "| seed-q4_0.gguf |" in plain(result.output)


def test_cli_report_needs_results(tmp_path: Path) -> None:
    result = cli.invoke(app, ["quality", "report", str(tmp_path)])

    assert result.exit_code == 1
    assert "no quality.json" in plain(result.output)


def testdefault_server_uses_the_settings(settings: Settings, tmp_path: Path) -> None:
    for name in ("llama-server", "llama-server.exe"):
        (settings.bin_dir / name).write_text("", encoding="utf-8")
    tree = tmp_path / "tree.gguf"

    server = default_server(settings)(tree, tmp_path / "server.log")

    assert server.command[server.command.index("--parallel") + 1] == "4"
    assert server.command[server.command.index("--ctx-size") + 1] == "8192"
    assert server.command[server.command.index("--threads") + 1] == "2"
    assert server.log == tmp_path / "server.log"
