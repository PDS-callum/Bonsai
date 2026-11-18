import json
from pathlib import Path

import pytest

from bonsai.gsm8k import (
    Gsm8kError,
    Problem,
    extract_answer,
    is_correct,
    load,
    parse_number,
    subset,
    wilson_interval,
)


def write_problems(path: Path, *answers: str) -> Path:
    lines = [
        json.dumps({"question": f"Q{i}?", "answer": f"Work.\n#### {a}"})
        for i, a in enumerate(answers)
    ]
    path.write_text("\n".join(lines) + "\n\n", encoding="utf-8")
    return path


def test_load_reads_answers_after_the_marker(tmp_path: Path) -> None:
    problems = load(write_problems(tmp_path / "t.jsonl", "18", "1,200"))

    assert problems == [Problem(0, "Q0?", 18.0), Problem(1, "Q1?", 1200.0)]


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ("not json", "not a GSM8K record"),
        ('{"question": "q"}', "not a GSM8K record"),
        ('{"question": "q", "answer": "no marker"}', "not a GSM8K record"),
        ('{"question": "q", "answer": "#### lots"}', "answer is not a number"),
    ],
)
def test_malformed_data_is_reported(tmp_path: Path, line: str, message: str) -> None:
    path = tmp_path / "t.jsonl"
    path.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(Gsm8kError, match=message):
        load(path)


def test_subset_is_fixed_and_ordered() -> None:
    problems = [Problem(i, f"Q{i}", float(i)) for i in range(100)]

    first = subset(problems, 10, seed=0)

    assert first == subset(problems, 10, seed=0)
    assert first != subset(problems, 10, seed=1)
    assert [p.index for p in first] == sorted(p.index for p in first)
    assert len(first) == 10


def test_subset_larger_than_the_set_takes_everything() -> None:
    problems = [Problem(i, "Q", 1.0) for i in range(3)]

    assert subset(problems, 10) == problems


@pytest.mark.parametrize(
    ("text", "value"),
    [("18", 18.0), ("$1,200", 1200.0), ("-3", -3.0), ("4.50.", 4.5), ("x", None)],
)
def test_parse_number(text: str, value: float | None) -> None:
    assert parse_number(text) == value


@pytest.mark.parametrize(
    ("reply", "value", "rule"),
    [
        ("so 5 + 13 = 18.\nThe answer is: 18", 18.0, "answer-is"),
        ("The answer is 7. Wait, the answer is: $1,050.", 1050.0, "answer-is"),
        ("The answer is: **42**", 42.0, "answer-is"),
        ("The answer is \\(12\\).", 12.0, "answer-is"),
        ("Total: 3 + 4.\n\\boxed{7}", 7.0, "boxed"),
        ("She has 3 apples and buys 9 more, 12 in all.", 12.0, "last-number"),
        ("I cannot solve this.", None, "none"),
    ],
)
def test_extract_answer(reply: str, value: float | None, rule: str) -> None:
    assert extract_answer(reply) == (value, rule)


def test_is_correct() -> None:
    assert is_correct(18.0, 18.0)
    assert not is_correct(18.5, 18.0)
    assert not is_correct(None, 18.0)


def test_wilson_interval_brackets_the_rate() -> None:
    low, high = wilson_interval(120, 400)

    assert low < 0.3 < high
    assert high - low == pytest.approx(0.09, abs=0.005)


@pytest.mark.parametrize(("correct", "total"), [(0, 50), (50, 50)])
def test_wilson_interval_stays_within_bounds(correct: int, total: int) -> None:
    low, high = wilson_interval(correct, total)

    assert 0.0 <= low < high <= 1.0


def test_wilson_interval_with_no_data() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)
