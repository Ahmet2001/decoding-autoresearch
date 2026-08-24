from __future__ import annotations

import math
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from .models import GenerationResult


TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)
SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)", re.UNICODE)


def normalize_text(text: str) -> str:
    # Unicode casefold maps Turkish capital dotted İ to ``i`` + combining dot.
    # Removing only that redundant combining mark makes exact checks stable while
    # preserving Turkish letters such as ş, ğ, ç, ö, and ü.
    return unicodedata.normalize("NFKC", text).casefold().replace("\u0307", "")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def repeated_ngram_coverage(tokens: list[str], n: int = 8) -> float:
    """Fraction covered by second-or-later occurrences of exact n-grams.

    Positions are unioned so a long repeated passage is not counted repeatedly by its
    overlapping n-grams.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if len(tokens) < n:
        return 0.0
    first_seen: dict[tuple[str, ...], int] = {}
    covered: set[int] = set()
    for start in range(len(tokens) - n + 1):
        gram = tuple(tokens[start : start + n])
        if gram in first_seen:
            covered.update(range(start, start + n))
        else:
            first_seen[gram] = start
    return len(covered) / len(tokens)


def has_repeated_sentence(text: str, min_tokens: int = 6) -> bool:
    seen: set[tuple[str, ...]] = set()
    for match in SENTENCE_RE.finditer(normalize_text(text)):
        sentence = tuple(tokenize(match.group(0).strip(" .!?\n\t")))
        if len(sentence) < min_tokens:
            continue
        if sentence in seen:
            return True
        seen.add(sentence)
    return False


def score_generation(
    result: GenerationResult,
    *,
    ngram_size: int,
    loop_fraction_threshold: float,
    sentence_min_tokens: int,
    output_token_limit: int | None = None,
) -> GenerationResult:
    result.loop_fraction = repeated_ngram_coverage(tokenize(result.thinking), ngram_size)
    result.sentence_loop = has_repeated_sentence(result.thinking, sentence_min_tokens)
    result.loop_detected = (
        result.loop_fraction >= loop_fraction_threshold or result.sentence_loop
    )
    result.truncated = result.done_reason in {"length", "max_tokens"}
    result.empty_answer = not result.content.strip()
    if output_token_limit is not None and output_token_limit <= 0:
        raise ValueError("output_token_limit must be positive")
    thinking_units = len(tokenize(result.thinking))
    content_units = len(tokenize(result.content))
    total_units = thinking_units + content_units
    generated_tokens = result.eval_count if result.eval_count > 0 else total_units
    result.estimated_thinking_tokens = (
        generated_tokens * thinking_units / total_units if total_units else 0.0
    )
    # A missing final answer receives the maximum penalty. Otherwise, estimate the
    # thinking share of Ollama's real generated-token count and normalize it by the
    # fixed output cap. The downstream quality gate prevents short wrong answers
    # from gaming this efficiency term.
    if result.empty_answer:
        result.reasoning_efficiency_penalty = 1.0
    elif output_token_limit is None:
        result.reasoning_efficiency_penalty = 0.0
    else:
        result.reasoning_efficiency_penalty = min(
            1.0, result.estimated_thinking_tokens / output_token_limit
        )
    return result


@dataclass(frozen=True)
class AggregateScore:
    repetition_score: float
    mean_loop_fraction: float
    loop_incidence: float
    truncation_rate: float
    mean_reasoning_efficiency_penalty: float
    empty_answer_rate: float


def aggregate_scores(
    samples: Iterable[GenerationResult],
    *,
    loop_fraction_weight: float,
    loop_incidence_weight: float,
    truncation_weight: float,
    reasoning_efficiency_weight: float,
) -> AggregateScore:
    rows = list(samples)
    if not rows:
        raise ValueError("Cannot aggregate an empty sample set")
    mean_loop = sum(row.loop_fraction for row in rows) / len(rows)
    incidence = sum(row.loop_detected for row in rows) / len(rows)
    truncation = sum(row.truncated for row in rows) / len(rows)
    inefficiency = sum(row.reasoning_efficiency_penalty for row in rows) / len(rows)
    empty_answers = sum(row.empty_answer for row in rows) / len(rows)
    score = (
        loop_fraction_weight * mean_loop
        + loop_incidence_weight * incidence
        + truncation_weight * truncation
        + reasoning_efficiency_weight * inefficiency
    )
    return AggregateScore(
        score, mean_loop, incidence, truncation, inefficiency, empty_answers
    )


def per_sample_penalty(
    sample: GenerationResult,
    *,
    loop_fraction_weight: float,
    loop_incidence_weight: float,
    truncation_weight: float,
    reasoning_efficiency_weight: float,
) -> float:
    return (
        loop_fraction_weight * sample.loop_fraction
        + loop_incidence_weight * float(sample.loop_detected)
        + truncation_weight * float(sample.truncated)
        + reasoning_efficiency_weight * sample.reasoning_efficiency_penalty
    )


def paired_bootstrap_upper_bound(
    baseline: Iterable[GenerationResult],
    candidate: Iterable[GenerationResult],
    *,
    samples: int,
    confidence: float,
    weights: tuple[float, float, float, float],
    seed: int = 20260714,
) -> float:
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be between 0.5 and 1")
    baseline_map = {(row.case_id, row.seed): row for row in baseline}
    candidate_map = {(row.case_id, row.seed): row for row in candidate}
    if baseline_map.keys() != candidate_map.keys():
        raise ValueError("Bootstrap inputs must contain identical case/seed pairs")
    loop_w, incidence_w, truncation_w, efficiency_w = weights
    differences = [
        per_sample_penalty(
            candidate_map[key],
            loop_fraction_weight=loop_w,
            loop_incidence_weight=incidence_w,
            truncation_weight=truncation_w,
            reasoning_efficiency_weight=efficiency_w,
        )
        - per_sample_penalty(
            baseline_map[key],
            loop_fraction_weight=loop_w,
            loop_incidence_weight=incidence_w,
            truncation_weight=truncation_w,
            reasoning_efficiency_weight=efficiency_w,
        )
        for key in sorted(baseline_map)
    ]
    if not differences:
        raise ValueError("No paired samples available")
    rng = random.Random(seed)
    boot_means = []
    for _ in range(samples):
        boot_means.append(
            sum(rng.choice(differences) for _ in differences) / len(differences)
        )
    boot_means.sort()
    index = min(len(boot_means) - 1, math.ceil(confidence * len(boot_means)) - 1)
    return boot_means[index]


def repeated_span_examples(
    samples: Iterable[GenerationResult], *, ngram_size: int = 8, limit: int = 4
) -> list[dict[str, str | float]]:
    """Return short diagnostics from visible samples only."""
    examples: list[dict[str, str | float]] = []
    for sample in sorted(samples, key=lambda row: row.loop_fraction, reverse=True):
        if not sample.loop_detected:
            continue
        tokens = tokenize(sample.thinking)
        positions: dict[tuple[str, ...], list[int]] = defaultdict(list)
        for start in range(max(0, len(tokens) - ngram_size + 1)):
            positions[tuple(tokens[start : start + ngram_size])].append(start)
        repeated = next((gram for gram, starts in positions.items() if len(starts) > 1), ())
        snippet = " ".join(repeated[:ngram_size])
        examples.append(
            {
                "case_id": sample.case_id,
                "language": sample.language,
                "loop_fraction": round(sample.loop_fraction, 6),
                "repeated_span": snippet,
            }
        )
        if len(examples) >= limit:
            break
    return examples
