from prompt_research.metrics import (
    aggregate_scores,
    has_repeated_sentence,
    normalize_text,
    paired_bootstrap_upper_bound,
    repeated_ngram_coverage,
    score_generation,
    tokenize,
)
from prompt_research.models import GenerationResult


def result(
    case: str,
    seed: int,
    thinking: str,
    done: str = "stop",
    *,
    content: str = "ok",
    eval_count: int = 20,
) -> GenerationResult:
    row = GenerationResult(
        case_id=case,
        split="validation",
        language="tr",
        category="test",
        seed=seed,
        thinking=thinking,
        content=content,
        done_reason=done,
        eval_count=eval_count,
        prompt_eval_count=1,
        duration_seconds=0.1,
    )
    return score_generation(
        row,
        ngram_size=4,
        loop_fraction_threshold=0.1,
        sentence_min_tokens=4,
        output_token_limit=100,
    )


def test_unicode_normalization_and_turkish_tokenization() -> None:
    assert normalize_text("MAVİ") == normalize_text("Mavi")
    assert "çalışma" in tokenize("Çalışma planı")


def test_repeated_ngram_coverage_unions_overlaps() -> None:
    tokens = "one two three four five six one two three four five six".split()
    coverage = repeated_ngram_coverage(tokens, n=4)
    assert 0.45 <= coverage <= 0.55


def test_sentence_loop_detection() -> None:
    text = "This is a sufficiently long repeated sentence. This is a sufficiently long repeated sentence."
    assert has_repeated_sentence(text, min_tokens=6)
    assert not has_repeated_sentence("First distinct sentence here. Another unrelated line here.", 3)


def test_aggregate_metric_weights() -> None:
    looping = result("a", 1, "a b c d a b c d", "length")
    clean = result("b", 1, "a b c d e f g h")
    score = aggregate_scores(
        [looping, clean],
        loop_fraction_weight=0.6,
        loop_incidence_weight=0.2,
        truncation_weight=0.1,
        reasoning_efficiency_weight=0.1,
    )
    assert 0 < score.repetition_score <= 1
    assert score.loop_incidence == 0.5
    assert score.truncation_rate == 0.5
    assert 0 < score.mean_reasoning_efficiency_penalty < 1


def test_reasoning_efficiency_penalizes_long_thinking_and_empty_answers() -> None:
    short = result("short", 1, "brief thought", eval_count=20)
    long = result(
        "long",
        1,
        "one two three four five six seven eight nine ten " * 8,
        eval_count=90,
    )
    empty = result("empty", 1, "brief thought", content="", eval_count=20)
    assert short.reasoning_efficiency_penalty < long.reasoning_efficiency_penalty
    assert empty.reasoning_efficiency_penalty == 1.0
    assert empty.empty_answer


def test_paired_bootstrap_supports_consistent_improvement() -> None:
    baseline = [result(str(index), 1, "a b c d a b c d") for index in range(12)]
    candidate = [result(str(index), 1, "a b c d e f g h") for index in range(12)]
    upper = paired_bootstrap_upper_bound(
        baseline,
        candidate,
        samples=200,
        confidence=0.95,
        weights=(0.6, 0.2, 0.1, 0.1),
    )
    assert upper < 0


def test_bootstrap_rejects_unpaired_inputs() -> None:
    baseline = [result("a", 1, "a b c d a b c d")]
    candidate = [result("b", 1, "a b c d e f g h")]
    try:
        paired_bootstrap_upper_bound(
            baseline,
            candidate,
            samples=10,
            confidence=0.95,
            weights=(0.6, 0.2, 0.1, 0.1),
        )
    except ValueError as exc:
        assert "identical" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("unpaired bootstrap should fail")
