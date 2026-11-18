import json
from collections.abc import Sequence
from functools import partial
from pathlib import Path

import pytest
from conftest import plain
from typer.testing import CliRunner

from bonsai import cli
from bonsai.cli import app
from bonsai.fetch import sha256_of
from bonsai.plant import (
    Pin,
    PlantError,
    check_source,
    convert_command,
    plant,
    quantize_command,
)

COMMIT = "b1b770d90d94cad9e9ea5b993bdbc383a522fb28"
PIN = Pin(COMMIT, "b12359")

runner = CliRunner()


class FakeTools:
    """Stands in for git, the converter and llama-quantize."""

    def __init__(self, commit: str = COMMIT, write: bool = True) -> None:
        self.commit = commit
        self.write = write
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> str:
        self.commands.append(list(command))
        if command[0] == "git":
            return self.commit + "\n"
        out = Path(command[command.index("--outfile") + 1] if "--outfile" in command
                   else command[2])  # fmt: skip
        if self.write:
            out.write_bytes(f"tree from {' '.join(command[-2:])}".encode())
        return "done\n"


@pytest.fixture
def setup(tmp_path: Path) -> tuple[Path, Path, Path]:
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "config.json").write_text("{}", encoding="utf-8")
    source = tmp_path / "llama.cpp"
    source.mkdir()
    (source / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("llama-quantize", "llama-quantize.exe"):
        (bin_dir / name).write_text("", encoding="utf-8")
    return seed, source, bin_dir


def test_commands(tmp_path: Path) -> None:
    convert = convert_command(
        tmp_path / "src", tmp_path / "seed", tmp_path / "o", "bf16"
    )
    quantize = quantize_command(
        tmp_path / "q", tmp_path / "b", tmp_path / "o", "q4_k_m"
    )

    assert convert[:3] == ["uv", "run", "--no-project"]
    assert convert[-4:] == ["--outtype", "bf16", "--outfile", str(tmp_path / "o")]
    assert quantize[-1] == "Q4_K_M"


def test_plant_converts_then_quantises(
    tmp_path: Path, setup: tuple[Path, Path, Path]
) -> None:
    seed, source, bin_dir = setup
    tools = FakeTools()
    out = tmp_path / "trees"

    record = plant(
        seed, out, "smol", source=source, bin_dir=bin_dir, pin=PIN,
        types=["q4_0", "q8_0"], runner=tools, log=lambda _: None,
    )  # fmt: skip

    names = [tree["file"] for tree in record["trees"]]
    assert names == ["smol-bf16.gguf", "smol-q4_0.gguf", "smol-q8_0.gguf"]
    assert all((out / name).is_file() for name in names)
    assert not list(out.glob("*.part"))
    assert record["trees"][1]["sha256"] == sha256_of(out / "smol-q4_0.gguf")
    assert record["llama_cpp"] == {"commit": COMMIT, "build": "b12359"}
    assert json.loads((out / "planting.json").read_text(encoding="utf-8")) == record
    assert (out / "logs" / "q8_0.log").read_text(encoding="utf-8") == "done\n"
    quantise = tools.commands[2]
    assert quantise[1] == str(out / "smol-bf16.gguf")
    assert quantise[-1] == "Q4_0"


def test_plant_keeps_existing_trees(
    tmp_path: Path, setup: tuple[Path, Path, Path]
) -> None:
    seed, source, bin_dir = setup
    out = tmp_path / "trees"
    out.mkdir()
    (out / "smol-bf16.gguf").write_bytes(b"base")
    tools = FakeTools()
    logged: list[str] = []

    plant(
        seed, out, "smol", source=source, bin_dir=bin_dir, pin=PIN,
        types=["q4_0"], runner=tools, log=logged.append,
    )  # fmt: skip

    assert logged == ["kept   smol-bf16.gguf", "plant  smol-q4_0.gguf"]
    assert len(tools.commands) == 2  # git, then one quantisation


@pytest.mark.parametrize(
    ("tools", "kwargs", "message"),
    [
        (FakeTools(), {"base_type": "q4_0"}, "base type must be one of"),
        (FakeTools(commit="abc123"), {}, "not the pin"),
        (FakeTools(write=False), {}, "was not written"),
    ],
)
def test_plant_errors(
    tmp_path: Path,
    setup: tuple[Path, Path, Path],
    tools: FakeTools,
    kwargs: dict[str, str],
    message: str,
) -> None:
    seed, source, bin_dir = setup

    with pytest.raises(PlantError, match=message):
        plant(
            seed, tmp_path / "trees", "smol", source=source, bin_dir=bin_dir,
            pin=PIN, runner=tools, log=lambda _: None, **kwargs,
        )  # fmt: skip


def test_plant_needs_a_fetched_seed(
    tmp_path: Path, setup: tuple[Path, Path, Path]
) -> None:
    _, source, bin_dir = setup

    with pytest.raises(PlantError, match="bonsai fetch"):
        plant(
            tmp_path / "missing", tmp_path / "trees", "smol", source=source,
            bin_dir=bin_dir, pin=PIN, runner=FakeTools(),
        )  # fmt: skip


def test_source_must_exist(tmp_path: Path) -> None:
    with pytest.raises(PlantError, match="fetch-llama-cpp"):
        check_source(tmp_path, PIN, FakeTools())


def test_pin_loads(tmp_path: Path) -> None:
    good = tmp_path / "good.env"
    good.write_text(f"LLAMA_CPP_COMMIT={COMMIT}\nLLAMA_CPP_BUILD=b12359\n")
    bad = tmp_path / "bad.env"
    bad.write_text("LLAMA_CPP_BUILD=b12359\n")

    assert Pin.load(good) == PIN
    with pytest.raises(PlantError, match="LLAMA_CPP_COMMIT"):
        Pin.load(bad)
    with pytest.raises(PlantError, match="repository root"):
        Pin.load(tmp_path / "missing.env")


def test_shipped_pin_loads() -> None:
    repo = Path(__file__).resolve().parents[1]

    assert Pin.load(repo / "configs" / "llama-cpp.env").commit == COMMIT


# The CLI


def seed_manifest(tmp_path: Path, kind: str = "seeds") -> Path:
    path = tmp_path / "seed.toml"
    path.write_text(
        f'name = "smol"\nkind = "{kind}"\nlicense = "apache-2.0"\n'
        f'[[files]]\nname = "config.json"\nurl = "file:///x"\nsha256 = "{"0" * 64}"\n',
        encoding="utf-8",
    )
    return path


def test_cli_plant(
    tmp_path: Path, setup: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, source, bin_dir = setup
    home = tmp_path / "home"
    (home / "seeds").mkdir(parents=True)
    seed.rename(home / "seeds" / "smol")
    source.rename(home / "llama.cpp")
    monkeypatch.setenv("BONSAI_HOME", str(home))
    monkeypatch.setattr(cli, "default_bin_dir", lambda: bin_dir)
    monkeypatch.setattr(cli, "plant", partial(plant, runner=FakeTools()))

    records = tmp_path / "records"

    result = runner.invoke(
        app,
        ["plant", str(seed_manifest(tmp_path)), "--types", "Q4_0",
         "--record", str(records)],
    )  # fmt: skip

    assert result.exit_code == 0, result.output
    assert "2 trees in" in plain(result.output)
    assert (home / "trees" / "smol" / "smol-q4_0.gguf").is_file()
    (kept,) = records.glob("smol/*/planting.json")
    assert json.loads(kept.read_text(encoding="utf-8"))["seed"] == "smol"


def test_cli_plant_from_a_directory(
    tmp_path: Path, setup: tuple[Path, Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    seed, source, bin_dir = setup
    home = tmp_path / "home"
    home.mkdir()
    source.rename(home / "llama.cpp")
    monkeypatch.setenv("BONSAI_HOME", str(home))
    monkeypatch.setattr(cli, "default_bin_dir", lambda: bin_dir)
    monkeypatch.setattr(cli, "plant", partial(plant, runner=FakeTools()))

    result = runner.invoke(
        app,
        ["plant", str(seed), "--types", "q4_0", "--record", str(tmp_path / "r")],
    )

    assert result.exit_code == 0, result.output
    assert (home / "trees" / "seed" / "seed-q4_0.gguf").is_file()


def test_cli_plant_rejects_data_manifests(tmp_path: Path) -> None:
    result = runner.invoke(app, ["plant", str(seed_manifest(tmp_path, kind="data"))])

    assert result.exit_code == 1
    assert "not a seed" in plain(result.output)
