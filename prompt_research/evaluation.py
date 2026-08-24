from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Iterable

from .benchmark import cases_for_split, validate_answer
from .deepseek import DeepSeekClient
from .config import decoding_hash
from .io_utils import atomic_write_json, prompt_hash, save_report
from .metrics import aggregate_scores, score_generation
from .models import (
    ApiUsage,
    BenchmarkCase,
    EvaluationReport,
    GenerationResult,
    QualityReport,
    DecodingConfig,
)
from .ollama import OllamaClient


class Evaluator:
    def __init__(
        self,
        *,
        ollama: OllamaClient,
        cases: list[BenchmarkCase],
        metric_config: dict,
        case_selection: dict | None = None,
        output_tokens_by_split: dict[str, int] | None = None,
    ) -> None:
        self.ollama = ollama
        self.cases = cases
        self.metric_config = metric_config
        self.case_selection = case_selection or {}
        self.output_tokens_by_split = output_tokens_by_split or {}
        # Validate configured subsets immediately instead of discovering a typo after
        # an expensive generation run has started.
        self.selected_cases("screen")
        self.selected_cases("validation")

    def selected_cases(self, split: str) -> list[BenchmarkCase]:
        available = cases_for_split(self.cases, split)
        configured = self.case_selection.get(f"{split}_case_ids")
        if not configured:
            return available
        if len(configured) != len(set(configured)):
            raise ValueError(f"Duplicate case id in configured {split} subset")
        by_id = {case.id: case for case in available}
        missing = [case_id for case_id in configured if case_id not in by_id]
        if missing:
            raise ValueError(f"Unknown {split} case ids: {missing}")
        selected = [by_id[case_id] for case_id in configured]
        languages = {language: sum(case.language == language for case in selected) for language in ("en", "tr")}
        if languages["en"] != languages["tr"]:
            raise ValueError(f"Configured {split} subset must be English/Turkish balanced")
        return selected

    def evaluate(
        self,
        *,
        system_prompt: str,
        decoding: DecodingConfig,
        split: str,
        seeds: Iterable[int],
        output_dir: Path,
        stop_check: Callable[[], bool] | None = None,
    ) -> EvaluationReport:
        selected_cases = self.selected_cases(split)
        output_token_limit = self.output_tokens_by_split.get(split)
        output_dir.mkdir(parents=True, exist_ok=True)
        response_path = output_dir / "responses.jsonl"
        samples: list[GenerationResult] = []
        with response_path.open("w", encoding="utf-8") as response_file:
            for case in selected_cases:
                for seed in seeds:
                    if stop_check is not None and stop_check():
                        raise EvaluationStopped("Evaluation stopped by the configured run limit")
                    result = self.ollama.generate(
                        case=case,
                        system_prompt=system_prompt,
                        decoding=decoding,
                        seed=int(seed),
                        output_tokens=output_token_limit,
                    )
                    passed, reason = validate_answer(case, result.content)
                    result.deterministic_pass = passed
                    result.deterministic_reason = reason
                    score_generation(
                        result,
                        ngram_size=int(self.metric_config["ngram_size"]),
                        loop_fraction_threshold=float(
                            self.metric_config["loop_fraction_threshold"]
                        ),
                        sentence_min_tokens=int(self.metric_config["sentence_min_tokens"]),
                        output_token_limit=int(
                            output_token_limit or self.ollama.output_tokens
                        ),
                    )
                    samples.append(result)
                    response_file.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")
                    response_file.flush()
        aggregate = aggregate_scores(
            samples,
            loop_fraction_weight=float(self.metric_config["loop_fraction_weight"]),
            loop_incidence_weight=float(self.metric_config["loop_incidence_weight"]),
            truncation_weight=float(self.metric_config["truncation_weight"]),
            reasoning_efficiency_weight=float(
                self.metric_config["reasoning_efficiency_weight"]
            ),
        )
        report = EvaluationReport(
            split=split,
            decoding_sha256=decoding_hash(decoding),
            prompt_sha256=prompt_hash(system_prompt),
            repetition_score=aggregate.repetition_score,
            mean_loop_fraction=aggregate.mean_loop_fraction,
            loop_incidence=aggregate.loop_incidence,
            truncation_rate=aggregate.truncation_rate,
            deterministic_pass_rate=sum(sample.deterministic_pass for sample in samples)
            / len(samples),
            samples=samples,
            output_token_limit=int(output_token_limit or self.ollama.output_tokens),
            mean_reasoning_efficiency_penalty=(
                aggregate.mean_reasoning_efficiency_penalty
            ),
            empty_answer_rate=aggregate.empty_answer_rate,
        )
        save_report(output_dir / "report.json", report)
        return report


class EvaluationStopped(RuntimeError):
    pass


def compare_quality(
    *,
    cases: list[BenchmarkCase],
    baseline: EvaluationReport,
    candidate: EvaluationReport,
    judge: DeepSeekClient,
    judge_max_tokens: int,
    max_loss_rate: float,
    quality_tolerance: float,
    output_path: Path | None = None,
    stop_check: Callable[[], bool] | None = None,
) -> tuple[QualityReport, ApiUsage]:
    if baseline.split != candidate.split:
        raise ValueError("Quality reports must use the same split")
    if candidate.deterministic_pass_rate < baseline.deterministic_pass_rate:
        report = QualityReport(
            passed=False,
            deterministic_pass_rate=candidate.deterministic_pass_rate,
            baseline_deterministic_pass_rate=baseline.deterministic_pass_rate,
            mean_quality=0.0,
            baseline_mean_quality=0.0,
            loss_rate=0.0,
            critical_failures=0,
            decisions=[],
            failure_reason="deterministic_pass_rate_regressed",
        )
        if output_path is not None:
            atomic_write_json(output_path, report.to_dict())
        return report, ApiUsage()
    baseline_map = {(sample.case_id, sample.seed): sample for sample in baseline.samples}
    candidate_map = {(sample.case_id, sample.seed): sample for sample in candidate.samples}
    if baseline_map.keys() != candidate_map.keys():
        raise ValueError("Quality reports must contain identical case/seed pairs")
    case_map = {case.id: case for case in cases}
    decisions = []
    usage = ApiUsage()
    for key in sorted(candidate_map):
        if stop_check is not None and stop_check():
            raise EvaluationStopped("Quality judging stopped by the configured run limit")
        case = case_map[key[0]]
        if case.validator.get("type") != "judge":
            continue
        decision = judge.judge_pair(
            case=case,
            baseline=baseline_map[key],
            candidate=candidate_map[key],
            max_tokens=judge_max_tokens,
        )
        decisions.append(decision)
        usage = usage + decision.usage
    if decisions:
        baseline_quality = sum(row.baseline_score for row in decisions) / len(decisions)
        candidate_quality = sum(row.candidate_score for row in decisions) / len(decisions)
        loss_rate = sum(row.preference == "baseline" for row in decisions) / len(decisions)
        critical = sum(row.critical_failure for row in decisions)
    else:
        baseline_quality = candidate_quality = 5.0
        loss_rate = 0.0
        critical = 0
    passed = (
        candidate.deterministic_pass_rate >= baseline.deterministic_pass_rate
        and critical == 0
        and loss_rate <= max_loss_rate
        and candidate_quality >= baseline_quality - quality_tolerance
    )
    report = QualityReport(
        passed=passed,
        deterministic_pass_rate=candidate.deterministic_pass_rate,
        baseline_deterministic_pass_rate=baseline.deterministic_pass_rate,
        mean_quality=candidate_quality,
        baseline_mean_quality=baseline_quality,
        loss_rate=loss_rate,
        critical_failures=critical,
        decisions=decisions,
    )
    if output_path is not None:
        atomic_write_json(output_path, report.to_dict())
    return report, usage
