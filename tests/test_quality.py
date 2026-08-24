from prompt_research.evaluation import compare_quality
from prompt_research.models import (
    ApiUsage,
    BenchmarkCase,
    EvaluationReport,
    GenerationResult,
    JudgeDecision,
)


def sample(content: str, passed: bool = True) -> GenerationResult:
    return GenerationResult(
        case_id="open",
        split="validation",
        language="en",
        category="test",
        seed=1,
        thinking="",
        content=content,
        done_reason="stop",
        eval_count=1,
        prompt_eval_count=1,
        duration_seconds=0.1,
        deterministic_pass=passed,
    )


def report(row: GenerationResult, pass_rate: float = 1.0) -> EvaluationReport:
    return EvaluationReport("validation", "x", "hash", 0, 0, 0, 0, pass_rate, [row])


class Judge:
    def __init__(self, critical: bool = False, preference: str = "tie") -> None:
        self.critical = critical
        self.preference = preference

    def judge_pair(self, **kwargs) -> JudgeDecision:
        return JudgeDecision(
            "open", 1, 4, 4, self.preference, self.critical, "test", ApiUsage(1, 1, 0.1)
        )


def test_quality_floor_passes_tie() -> None:
    case = BenchmarkCase("open", "validation", "en", "test", [{"role": "user", "content": "x"}], {"type": "judge"}, "x")
    quality, usage = compare_quality(
        cases=[case],
        baseline=report(sample("baseline")),
        candidate=report(sample("candidate")),
        judge=Judge(),
        judge_max_tokens=10,
        max_loss_rate=0.1,
        quality_tolerance=0.1,
    )
    assert quality.passed
    assert usage.estimated_cost_usd == 0.1


def test_critical_or_wrong_language_judge_failure_blocks_candidate() -> None:
    case = BenchmarkCase("open", "validation", "tr", "test", [{"role": "user", "content": "Türkçe cevapla"}], {"type": "judge"}, "Türkçe olmalı")
    quality, _ = compare_quality(
        cases=[case],
        baseline=report(sample("Türkçe cevap")),
        candidate=report(sample("English answer")),
        judge=Judge(critical=True, preference="baseline"),
        judge_max_tokens=10,
        max_loss_rate=0.1,
        quality_tolerance=0.1,
    )
    assert not quality.passed


def test_deterministic_regression_skips_paid_judge() -> None:
    class MustNotRun:
        def judge_pair(self, **kwargs):
            raise AssertionError("judge should be skipped")

    case = BenchmarkCase(
        "open",
        "validation",
        "en",
        "test",
        [{"role": "user", "content": "x"}],
        {"type": "judge"},
        "x",
    )
    quality, usage = compare_quality(
        cases=[case],
        baseline=report(sample("baseline"), pass_rate=1.0),
        candidate=report(sample("candidate", passed=False), pass_rate=0.0),
        judge=MustNotRun(),
        judge_max_tokens=10,
        max_loss_rate=0.1,
        quality_tolerance=0.1,
    )
    assert not quality.passed
    assert quality.failure_reason == "deterministic_pass_rate_regressed"
    assert usage.estimated_cost_usd == 0.0
