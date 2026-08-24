from pathlib import Path

import pytest

from prompt_research.benchmark import load_benchmark, validate_answer
from prompt_research.evaluation import Evaluator
from prompt_research.models import BenchmarkCase


ROOT = Path(__file__).parents[1]


def test_benchmark_shape_and_balance() -> None:
    cases = load_benchmark(ROOT / "benchmark.jsonl")
    assert len(cases) == 36
    assert sum(case.split == "screen" for case in cases) == 12
    assert sum(case.split == "validation" for case in cases) == 24
    for split in ("screen", "validation"):
        assert sum(case.split == split and case.language == "en" for case in cases) == sum(
            case.split == split and case.language == "tr" for case in cases
        )


@pytest.mark.parametrize(
    ("validator", "answer", "passed"),
    [
        ({"type": "exact", "value": "MAVİ"}, "mavi", True),
        ({"type": "contains_all", "values": ["good morning", "how are you"]}, "Good morning, how are you?", True),
        ({"type": "contains_any", "values": ["Ankara"]}, "İstanbul", False),
        (
            {"type": "contains_groups", "groups": [["mixed"], ["apple"], ["orange"]]},
            "Draw from mixed; an apple or orange identifies it.",
            True,
        ),
        ({"type": "regex", "pattern": r"\b56\b"}, "The answer is 56.", True),
        ({"type": "judge", "min_chars": 5}, "short enough", True),
    ],
)
def test_validators(validator: dict, answer: str, passed: bool) -> None:
    case = BenchmarkCase("x", "screen", "en", "test", [{"role": "user", "content": "x"}], validator, "x")
    assert validate_answer(case, answer)[0] is passed


def test_empty_answer_always_fails() -> None:
    case = BenchmarkCase("x", "screen", "en", "test", [{"role": "user", "content": "x"}], {"type": "judge"}, "x")
    assert not validate_answer(case, "   ")[0]


def test_fast_subset_is_balanced() -> None:
    cases = load_benchmark(ROOT / "benchmark.jsonl")
    evaluator = Evaluator(
        ollama=None,
        cases=cases,
        metric_config={},
        case_selection={
            "screen_case_ids": ["s-en-06", "s-tr-06", "s-en-03", "s-tr-03"],
            "validation_case_ids": ["v-en-10", "v-tr-10", "v-en-03", "v-tr-03"],
        },
    )
    selected = evaluator.selected_cases("screen") + evaluator.selected_cases("validation")
    assert len(selected) == 8
    assert sum(case.language == "en" for case in selected) == 4
    assert sum(case.language == "tr" for case in selected) == 4
