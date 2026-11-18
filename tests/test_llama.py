import sys
from pathlib import Path

import pytest

from bonsai.llama import (
    LlamaError,
    default_bin_dir,
    parse_kl_divergence,
    parse_multiple_choice,
    parse_perplexity,
    read_env_file,
    run_tool,
    tool,
)

# Trimmed output of real llama-perplexity runs (b12359, Qwen2.5-0.5B Q4_0).
DATA = Path(__file__).parent / "data" / "llama"


def fixture(name: str) -> str:
    return (DATA / name).read_text(encoding="utf-8")


def test_parse_perplexity() -> None:
    estimate = parse_perplexity(fixture("perplexity.txt"))

    assert (estimate.value, estimate.stderr) == (17.4968, 0.13529)


def test_parse_multiple_choice() -> None:
    estimate = parse_multiple_choice(fixture("multiple-choice.txt"))

    assert (estimate.value, estimate.stderr) == (48.7917, 1.6966)


def test_parse_kl_divergence() -> None:
    divergence = parse_kl_divergence(fixture("kl-divergence.txt"))

    assert divergence.mean_kld.value == 0.116853
    assert divergence.mean_kld.stderr == 0.001938
    assert divergence.same_top.value == 82.635
    assert divergence.ppl_ratio.value == 1.125725
    assert divergence.kld_99 == 0.832003


@pytest.mark.parametrize(
    "parse", [parse_perplexity, parse_multiple_choice, parse_kl_divergence]
)
def test_missing_results_are_reported(parse: object) -> None:
    with pytest.raises(LlamaError, match="no "):
        parse("the tool crashed before printing anything")  # type: ignore[operator]


def test_kl_divergence_needs_its_percentiles() -> None:
    text = "\n".join(
        line
        for line in fixture("kl-divergence.txt").splitlines()
        if "%   KLD" not in line
    )

    with pytest.raises(LlamaError, match="percentiles"):
        parse_kl_divergence(text)


# Finding the tools


def test_env_file_is_read(tmp_path: Path) -> None:
    path = tmp_path / "pin.env"
    path.write_text("# comment\nLLAMA_CPP_BUILD=b1\nnot a pair\n", encoding="utf-8")

    assert read_env_file(path) == {"LLAMA_CPP_BUILD": "b1"}


def test_bin_dir_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BONSAI_LLAMA_BIN", "/opt/llama")

    assert default_bin_dir() == Path("/opt/llama")


def test_bin_dir_from_the_pinned_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BONSAI_LLAMA_BIN", raising=False)
    pin = tmp_path / "llama-cpp.env"
    pin.write_text("LLAMA_CPP_BUILD=b12359\n", encoding="utf-8")

    assert default_bin_dir(pin, tmp_path) == tmp_path / "llama.cpp-bin" / "b12359"


def test_bin_dir_falls_back_to_a_source_build(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("BONSAI_LLAMA_BIN", raising=False)

    found = default_bin_dir(tmp_path / "missing.env", tmp_path)

    assert found == tmp_path / "llama.cpp" / "build" / "bin"


def test_tool_finds_executables(tmp_path: Path) -> None:
    name = "llama-perplexity.exe" if sys.platform == "win32" else "llama-perplexity"
    (tmp_path / name).write_text("", encoding="utf-8")

    assert tool(tmp_path, "llama-perplexity") == tmp_path / name


def test_missing_tool_is_reported(tmp_path: Path) -> None:
    with pytest.raises(LlamaError, match="llama-server not found"):
        tool(tmp_path, "llama-server")


# Running the tools


def test_run_tool_combines_output() -> None:
    script = "import sys; print('out'); print('err', file=sys.stderr)"

    output = run_tool([sys.executable, "-c", script])

    assert "out" in output
    assert "err" in output


def test_run_tool_reports_failures() -> None:
    script = "print('last words'); raise SystemExit(3)"

    with pytest.raises(LlamaError, match="exited with 3:\nlast words"):
        run_tool([sys.executable, "-c", script])


def test_run_tool_reports_missing_programs() -> None:
    with pytest.raises(LlamaError, match="not found"):
        run_tool(["bonsai-no-such-tool"])
