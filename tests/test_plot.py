import xml.etree.ElementTree as ET
from dataclasses import replace
from pathlib import Path

import pytest
from conftest import plain
from typer.testing import CliRunner

from bonsai.cli import app
from bonsai.journal import FrontierPoint
from bonsai.plot import Panel, family, pretty, render, series_of, short_label

runner = CliRunner()
SVG = "{http://www.w3.org/2000/svg}"


def point(**overrides: object) -> FrontierPoint:
    base = FrontierPoint(
        file="qwen2.5-0.5b-instruct-q4_0.gguf",
        seed="qwen2.5-0.5b-instruct",
        quant="q4_0",
        generation_tps=34.0,
        peak_memory_bytes=5e8,
        quality=29.0,
        low=24.8,
        high=33.6,
        on_frontier=True,
        origin="qwen2.5-0.5b-instruct",
    )
    return replace(base, **overrides)  # type: ignore[arg-type]


PRUNED = point(
    file="qwen2.5-0.5b-instruct-block6-q4_k_m.gguf",
    seed="qwen2.5-0.5b-instruct-block6",
    quant="q4_k_m",
    origin="qwen2.5-0.5b-instruct-weights",
    pruned=True,
    generation_tps=36.0,
)


@pytest.mark.parametrize(
    ("name", "shown"),
    [
        ("qwen2.5-0.5b-instruct", "Qwen2.5-0.5B"),
        ("smollm2-360m-instruct", "SmolLM2-360M"),
        ("llama-3.2-1b-instruct", "Llama-3.2-1B"),
        ("tiny", "tiny"),
    ],
)
def test_family_names(name: str, shown: str) -> None:
    assert pretty(name) == shown


def test_labels_and_families() -> None:
    assert family(PRUNED) == "qwen2.5-0.5b-instruct"
    assert family(point(origin="")) == "qwen2.5-0.5b-instruct"
    assert short_label(PRUNED) == "block6 Q4_K_M"
    assert short_label(point()) == "Q4_0"


def test_series_fold_past_three() -> None:
    points = [point(origin=f"family{i}-1b") for i in range(4)]

    series = series_of(points)

    assert [s.slot for s in series.values()] == [0, 1, 2, None]
    assert list(series.values())[3].label == "Other"


def test_render_is_valid_svg_with_every_point() -> None:
    smol = point(
        file="smollm2-360m-instruct-q2_k.gguf", seed="smollm2-360m-instruct",
        origin="smollm2-360m-instruct", quant="q2_k", generation_tps=48.8,
        quality=1.5, low=None, high=None, on_frontier=False,
    )  # fmt: skip
    others = [point(origin=f"zz{i}-1b", generation_tps=10.0 + i) for i in range(3)]
    panels = [
        Panel("Maths", [point(), PRUNED, smol, *others], 0, 50, 10),
        Panel("Science", [point(quality=48.0, low=46.0, high=50.0)], 20, 60, 10),
    ]

    svg = render(panels, title="A & B", x_label="Speed")

    root = ET.fromstring(svg)
    assert root.find(f"{SVG}title").text == "A & B"  # type: ignore[union-attr]
    tips = [t.text for t in root.iter(f"{SVG}title")]
    assert "qwen2.5-0.5b-instruct-q4_0.gguf: 34.0 tokens/s, 29.0%" in tips
    texts = [t.text for t in root.iter(f"{SVG}text")]
    assert "Qwen2.5-0.5B, pruned" in texts
    assert texts.count("Other") == 1
    assert "block6 Q4_K_M" in texts
    assert "Q2_K" not in texts  # only frontier trees are labelled
    assert "60" in texts  # headroom past the fastest tree
    assert len(root.findall(f".//{SVG}path[@class='frontier']")) == 2
    assert "prefers-color-scheme: dark" in svg


def test_labels_avoid_each_other() -> None:
    close = [point(generation_tps=30.0 + i * 0.1, quality=40.0) for i in range(6)]

    root = ET.fromstring(render([Panel("P", close, 0, 50, 10)], title="t", x_label="x"))

    spots = {
        (t.get("x"), t.get("y"))
        for t in root.iter(f"{SVG}text")
        if t.get("class") == "label" and t.text == "Q4_0"
    }
    assert len(spots) == 6


def test_cli_plot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BONSAI_HOME", str(tmp_path))
    repo = Path(__file__).resolve().parents[1]
    db = tmp_path / "j.sqlite"
    built = runner.invoke(
        app, ["journal", "build", "--db", str(db), "--root", str(repo)]
    )
    out = tmp_path / "plot.svg"

    result = runner.invoke(
        app,
        ["journal", "plot", "--db", str(db), "--out", str(out),
         "--pot", str(repo / "configs" / "pots" / "pi5-4gb.toml")],
    )  # fmt: skip

    assert built.exit_code == 0, built.output
    assert result.exit_code == 0, result.output
    assert "trees plotted in" in plain(result.output)
    ET.fromstring(out.read_text(encoding="utf-8"))


def test_cli_plot_needs_a_journal(tmp_path: Path) -> None:
    result = runner.invoke(app, ["journal", "plot", "--db", str(tmp_path / "x")])

    assert result.exit_code == 1
    assert "bonsai journal build" in plain(result.output)
