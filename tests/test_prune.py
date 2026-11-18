import json
import struct
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest
from conftest import plain
from typer.testing import CliRunner

from bonsai import prune_cli
from bonsai.check import CheckError, compare, load_reference, reference_answers
from bonsai.cli import app
from bonsai.prune import (
    PruneError,
    choose_layers,
    cut_config,
    load_scores,
    prune,
    read_header,
    score_layers,
    tool_command,
)

runner = CliRunner()


def write_safetensors(path: Path, tensors: dict[str, bytes]) -> None:
    """A safetensors file of uint8 tensors, stored in the order given but
    listed in the header in reverse, as real files need not match."""
    header: dict[str, Any] = {"__metadata__": {"format": "pt"}}
    offset = 0
    for name, data in tensors.items():
        header[name] = {
            "dtype": "U8",
            "shape": [len(data)],
            "data_offsets": [offset, offset + len(data)],
        }
        offset += len(data)
    listed = dict(reversed(list(header.items())))
    encoded = json.dumps(listed).encode()
    path.write_bytes(
        struct.pack("<Q", len(encoded)) + encoded + b"".join(tensors.values())
    )


def read_tensors(path: Path) -> dict[str, bytes]:
    header, start = read_header(path)
    data = path.read_bytes()
    return {
        name: data[start + spec["data_offsets"][0] : start + spec["data_offsets"][1]]
        for name, spec in header.items()
        if name != "__metadata__"
    }


@pytest.fixture
def seed(tmp_path: Path) -> Path:
    """A four-layer seed whose tensors hold their layer number."""
    directory = tmp_path / "seed"
    directory.mkdir()
    tensors = {"model.embed_tokens.weight": b"E" * 4}
    for layer in range(4):
        tensors[f"model.layers.{layer}.mlp.weight"] = bytes([layer]) * 3
        tensors[f"model.layers.{layer}.input_layernorm.weight"] = bytes([layer])
    tensors["model.norm.weight"] = b"N"
    write_safetensors(directory / "model.safetensors", tensors)
    config = {"num_hidden_layers": 4, "max_window_layers": 4,
              "layer_types": ["a", "b", "c", "d"]}  # fmt: skip
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    return directory


SCORES = {
    "layers": 5,
    "tokens": 1000,
    # distance[i][n - 1]: from the input of layer i to the input of layer i + n
    "distance": [
        [0.5, 0.6, 0.7, 0.8],
        [0.2, 0.25, 0.4],
        [0.1, 0.3],
        [0.15],
    ],
}


def test_block_cuts_the_run_that_changes_least() -> None:
    assert choose_layers(SCORES, 1, "block") == [2]
    assert choose_layers(SCORES, 2, "block") == [1, 2]
    assert choose_layers(SCORES, 3, "block") == [1, 2, 3]


def test_single_cuts_the_layers_that_each_change_least() -> None:
    assert choose_layers(SCORES, 2, "single") == [2, 3]
    assert choose_layers(SCORES, 3, "single") == [1, 2, 3]


@pytest.mark.parametrize(
    ("cut", "method", "message"),
    [(0, "block", "can cut 1 to 3"), (4, "block", "not 4"), (1, "magic", "method")],
)
def test_impossible_cuts_are_refused(cut: int, method: str, message: str) -> None:
    with pytest.raises(PruneError, match=message):
        choose_layers(SCORES, cut, method)


def test_prune_removes_and_renumbers_layers(seed: Path, tmp_path: Path) -> None:
    out = tmp_path / "pruned"

    pruning = prune(
        seed, out, name="seed-block2", seed_name="seed", removed=[2, 1], method="block"
    )

    tensors = read_tensors(out / "model.safetensors")
    assert tensors["model.layers.0.mlp.weight"] == bytes([0]) * 3
    assert tensors["model.layers.1.mlp.weight"] == bytes([3]) * 3
    assert tensors["model.layers.1.input_layernorm.weight"] == bytes([3])
    assert "model.layers.2.mlp.weight" not in tensors
    assert tensors["model.embed_tokens.weight"] == b"E" * 4
    assert tensors["model.norm.weight"] == b"N"
    header, start = read_header(out / "model.safetensors")
    assert header["__metadata__"] == {"format": "pt"}
    assert start % 8 == 0
    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config == {"num_hidden_layers": 2, "max_window_layers": 2,
                      "layer_types": ["a", "d"]}  # fmt: skip
    assert (out / "tokenizer.json").is_file()
    assert pruning.removed == (1, 2)
    assert (pruning.layers_before, pruning.layers_after) == (4, 2)
    record = pruning.record({"text": "t", "samples": 2}, "scores.json")
    assert record["removed_layers"] == [1, 2]
    assert record["calibration"]["samples"] == 2
    assert pruning.record(None, None)["calibration"] is None


def test_config_without_optional_fields() -> None:
    assert cut_config({"num_hidden_layers": 3}, [0]) == {"num_hidden_layers": 2}


@pytest.mark.parametrize("removed", [[], [4], [1, 1], [-1]])
def test_prune_refuses_bad_layers(
    seed: Path, tmp_path: Path, removed: list[int]
) -> None:
    with pytest.raises(PruneError, match="must be distinct"):
        prune(seed, tmp_path / "o", name="x", seed_name="s", removed=removed,
              method="block")  # fmt: skip


def test_prune_needs_single_file_weights(tmp_path: Path) -> None:
    with pytest.raises(PruneError, match="sharded seeds are not supported"):
        prune(tmp_path, tmp_path / "o", name="x", seed_name="s", removed=[0],
              method="block")  # fmt: skip


def test_truncated_weights_are_reported(seed: Path, tmp_path: Path) -> None:
    weights = seed / "model.safetensors"
    weights.write_bytes(weights.read_bytes()[:-5])

    with pytest.raises(PruneError, match="ends before its tensors do"):
        prune(seed, tmp_path / "o", name="x", seed_name="s", removed=[0],
              method="block")  # fmt: skip


@pytest.mark.parametrize(
    ("data", "message"),
    [(b"abc", "too short"), (struct.pack("<Q", 3) + b"{x}", "unreadable")],
)
def test_bad_headers_are_reported(tmp_path: Path, data: bytes, message: str) -> None:
    path = tmp_path / "bad.safetensors"
    path.write_bytes(data)

    with pytest.raises(PruneError, match=message):
        read_header(path)


# Scores and reference answers come from layer_tool.py, run through uv.


class FakeTool:
    """Stands in for layer_tool.py: writes the output file it is asked for."""

    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> str:
        self.commands.append(list(command))
        tool = command.index(str(prune_cli_tool()))
        out = Path(command[tool + 4])
        out.write_text(json.dumps(self.output), encoding="utf-8")
        return ""


def prune_cli_tool() -> Path:
    return Path("scripts/pc/layer_tool.py")


def test_tool_runs_in_the_converters_environment(tmp_path: Path) -> None:
    command = tool_command(tmp_path, "score", "a")

    assert command[:3] == ["uv", "run", "--no-project"]
    assert command[-3:] == [str(prune_cli_tool()), "score", "a"]


def test_score_layers_runs_the_tool(tmp_path: Path) -> None:
    tool = FakeTool(SCORES)

    scores = score_layers(
        tmp_path / "seed", tmp_path / "t.txt", tmp_path / "s" / "scores.json",
        source=tmp_path, samples=4, runner=tool,
    )  # fmt: skip

    assert scores["layers"] == 5
    assert tool.commands[0][-4:] == ["--samples", "4", "--length", "512"]


@pytest.mark.parametrize(
    ("content", "message"),
    [("[]", "not a layer score file"), (json.dumps({"layers": 3, "distance": []}),
                                          "expected scores for 2 layers")],
)  # fmt: skip
def test_bad_scores_are_reported(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "scores.json"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(PruneError, match=message):
        load_scores(path)


REFERENCE = {
    "model": "seed",
    "answers": [
        {"prompt": "a", "prompt_ids": [1, 2], "continuation_ids": [3, 4, 5, 6]},
        {"prompt": "b", "prompt_ids": [7], "continuation_ids": [8, 9, 10, 11]},
    ],
}


def counting(prompt_ids: Sequence[int], tokens: int) -> list[int]:
    """Continues by counting up, like the fake llama-server."""
    return list(range(prompt_ids[-1] + 1, prompt_ids[-1] + 1 + tokens))


def test_matching_continuations_pass() -> None:
    result = compare(REFERENCE, counting)

    assert result["passed"]
    assert (result["tokens_agree"], result["tokens"]) == (8, 8)


def test_a_late_difference_is_tolerated_but_an_early_one_is_not() -> None:
    def late(ids: Sequence[int], n: int) -> list[int]:
        return [*counting(ids, n)[:-1], 0]

    def early(ids: Sequence[int], n: int) -> list[int]:
        got = counting(ids, n)
        return [0, *got[1:]] if ids[-1] == 7 else got

    assert compare(REFERENCE, late)["passed"]  # 6 of 8 tokens agree
    failed = compare(REFERENCE, early)
    assert not failed["passed"]
    assert failed["first_tokens_agree"] == 1


def test_a_near_tie_on_the_first_token_is_tolerated() -> None:
    tied = {"answers": [
        {"prompt_ids": [1], "continuation_ids": [5, 6],
         "first_top": [[5, -1.0], [9, -1.05], [7, -2.0]]},
        {"prompt_ids": [2], "continuation_ids": [3, 4]},
    ]}  # fmt: skip

    def answer(token: int) -> Any:
        return lambda ids, n: [token, 0] if ids == [1] else counting(ids, n)

    assert compare(tied, answer(9))["near_ties"] == 1
    assert compare(tied, answer(9))["first_tokens_agree"] == 2
    assert compare(tied, answer(7))["first_tokens_agree"] == 1  # not a tie
    assert compare(tied, answer(8))["first_tokens_agree"] == 1  # not in the top
    no_top = {"answers": [{"prompt_ids": [1], "continuation_ids": [5]}]}
    assert compare(no_top, lambda ids, n: [])["first_tokens_agree"] == 0


def test_reference_answers_run_the_tool(tmp_path: Path) -> None:
    tool = FakeTool(REFERENCE)

    reference = reference_answers(
        tmp_path / "seed", tmp_path / "ref.json", source=tmp_path, runner=tool
    )

    assert reference["answers"][0]["continuation_ids"] == [3, 4, 5, 6]
    assert tool.commands[0][-2:] == ["--tokens", "16"]


def test_bad_reference_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "ref.json"
    path.write_text(json.dumps({"answers": [{"prompt": "a"}]}), encoding="utf-8")

    with pytest.raises(CheckError, match="not a reference answer file"):
        load_reference(path)


# The CLI


def seed_manifest(tmp_path: Path, kind: str = "seeds") -> Path:
    path = tmp_path / "seed.toml"
    path.write_text(
        f'name = "seed-weights"\nkind = "{kind}"\nlicense = "mit"\n'
        f'[[files]]\nname = "config.json"\nurl = "file:///x"\nsha256 = "{"0" * 64}"\n',
        encoding="utf-8",
    )
    return path


@pytest.fixture
def home(tmp_path: Path, seed: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    home = tmp_path / "home"
    (home / "seeds").mkdir(parents=True)
    seed.rename(home / "seeds" / "seed-weights")
    monkeypatch.setenv("BONSAI_HOME", str(home))
    return home


def test_cli_score_and_cut(
    tmp_path: Path, home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = tmp_path / "text.txt"
    text.write_text("words", encoding="utf-8")
    records = tmp_path / "records"

    def score(seed: Path, text: Path, out: Path, **kw: Any) -> dict[str, Any]:
        return score_layers(seed, text, out, runner=FakeTool(SCORES), **kw)

    monkeypatch.setattr(prune_cli, "score_layers", score)
    manifest = str(seed_manifest(tmp_path))

    scored = runner.invoke(
        app, ["prune", "score", manifest, "--text", str(text), "--out", str(records)]
    )
    cut = runner.invoke(
        app, ["prune", "cut", manifest, "--cut", "2", "--record", str(records)]
    )

    assert scored.exit_code == 0, scored.output
    assert "    2  0.100" in scored.output
    assert cut.exit_code == 0, cut.output
    assert "seed-block2: removed layers 1, 2 (4 -> 2)" in plain(cut.output)
    assert (home / "seeds" / "seed-block2" / "model.safetensors").is_file()
    record = json.loads((records / "seed-block2" / "prune.json").read_text("utf-8"))
    assert record["seed"] == "seed-weights"
    assert record["scores"].endswith("seed-weights/scores.json")


def test_cli_score_needs_calibration_text(tmp_path: Path, home: Path) -> None:
    result = runner.invoke(app, ["prune", "score", str(seed_manifest(tmp_path))])

    assert result.exit_code == 1
    assert "calibration-v1.toml" in plain(result.output)


def test_cli_cut_rejects_data_manifests(tmp_path: Path, home: Path) -> None:
    result = runner.invoke(
        app, ["prune", "cut", str(seed_manifest(tmp_path, "data")), "--cut", "1"]
    )

    assert result.exit_code == 1
    assert "not a seed" in plain(result.output)


class FakeServer:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.complete = counting

    def __enter__(self) -> "FakeServer":
        return self

    def __exit__(self, *args: object) -> None:
        pass


@pytest.mark.parametrize(("answers", "code", "verdict"), [
    (REFERENCE["answers"], 0, "passed"),
    ([{"prompt": "x", "prompt_ids": [1], "continuation_ids": [0, 0]}], 1, "FAILED"),
])  # fmt: skip
def test_cli_check(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    answers: list[dict[str, Any]],
    code: int,
    verdict: str,
) -> None:
    tree = tmp_path / "tree-bf16.gguf"
    tree.write_bytes(b"gguf")
    weights = tmp_path / "weights"
    weights.mkdir()
    reference = {"model": str(weights), "answers": answers}
    made: list[Path] = []

    def make_reference(seed: Path, out: Path, **kw: Any) -> dict[str, Any]:
        made.append(out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(reference), encoding="utf-8")
        return reference

    monkeypatch.setattr(prune_cli, "reference_answers", make_reference)
    monkeypatch.setattr(prune_cli, "tool", lambda bin_dir, name: Path(name))
    monkeypatch.setattr(prune_cli, "LlamaServer", FakeServer)
    args = ["check", str(tree), str(weights), "--out", str(tmp_path / "checks")]

    first = runner.invoke(app, args)
    second = runner.invoke(app, args)  # the reference is reused

    assert first.exit_code == second.exit_code == code, first.output
    assert f"tree-bf16.gguf: {verdict}." in first.output
    assert len(made) == 1
    saved = tmp_path / "checks" / "tree-bf16" / "check.json"
    assert json.loads(saved.read_text(encoding="utf-8"))["tree"] == "tree-bf16.gguf"


def test_cli_check_reports_errors(tmp_path: Path) -> None:
    tree = tmp_path / "tree.gguf"
    tree.write_bytes(b"gguf")
    target = tmp_path / "checks" / "tree"
    target.mkdir(parents=True)
    (target / "reference.json").write_text("[]", encoding="utf-8")

    result = runner.invoke(
        app, ["check", str(tree), str(tmp_path), "--out", str(tmp_path / "checks")]
    )

    assert result.exit_code == 1
    assert "not a reference answer file" in plain(result.output)
