from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna
from optuna.distributions import FloatDistribution, IntDistribution
from optuna.samplers import TPESampler
from optuna.trial import TrialState, create_trial

from .benchmark import load_benchmark
from .config import (
    AppConfig,
    decoding_hash,
    render_decoding_config,
    require_deepseek_key,
    validate_decoding_values,
)
from .deepseek import BudgetExceeded, CostBudget, DeepSeekClient, ProposalRejected
from .evaluation import EvaluationStopped, Evaluator, compare_quality
from .git_ops import commit_decoding_config
from .io_utils import (
    atomic_write_json,
    atomic_write_text,
    load_report,
    load_state,
    prompt_hash,
    save_state,
)
from .metrics import paired_bootstrap_upper_bound, repeated_span_examples
from .models import ApiUsage, DecodingConfig, EvaluationReport, QualityReport
from .ollama import OllamaClient


RESULT_COLUMNS = [
    "run_id",
    "commit",
    "candidate",
    "screen_score",
    "validation_score",
    "quality_score",
    "loop_incidence",
    "truncation_rate",
    "reasoning_efficiency_penalty",
    "empty_answer_rate",
    "status",
    "boss_tokens",
    "estimated_cost_usd",
    "cumulative_cost_usd",
    "decoding_config",
    "description",
]


def utc_run_id(prefix: str = "run") -> str:
    return f"{prefix}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


class ResearchRunner:
    def __init__(
        self,
        config: AppConfig,
        *,
        benchmark_path: Path | None = None,
        ollama: OllamaClient | None = None,
        deepseek_factory: Any | None = None,
    ) -> None:
        self.config = config
        self.root = config.root
        self.benchmark_path = benchmark_path or self.root / "benchmark.jsonl"
        self.cases = load_benchmark(self.benchmark_path)
        ollama_config = config.ollama
        self.ollama = ollama or OllamaClient(
            base_url=str(ollama_config["base_url"]),
            model=str(ollama_config["model"]),
            context_length=int(ollama_config["context_length"]),
            output_tokens=int(ollama_config["output_tokens"]),
            timeout=float(ollama_config["timeout_seconds"]),
        )
        self.evaluator = Evaluator(
            ollama=self.ollama,
            cases=self.cases,
            metric_config=config.metrics,
            case_selection=config.benchmark,
            output_tokens_by_split={
                "screen": int(ollama_config["output_tokens"]),
                "validation": int(
                    ollama_config.get("validation_output_tokens", ollama_config["output_tokens"])
                ),
            },
        )
        self.deepseek_factory = deepseek_factory
        self.state_path = self.root / ".autoresearch" / "decoding_state.json"
        self.state = load_state(self.state_path)
        self.runs_dir = self.root / "runs"
        self.results_path = self.root / "results.tsv"
        self._ensure_results_header()

    def close(self) -> None:
        self.ollama.close()

    def _ensure_results_header(self) -> None:
        if self.results_path.exists():
            return
        self.results_path.write_text("\t".join(RESULT_COLUMNS) + "\n", encoding="utf-8")

    def _append_result(self, values: dict[str, Any]) -> None:
        sanitized = {
            key: str(values.get(key, "")).replace("\t", " ").replace("\n", " ")
            for key in RESULT_COLUMNS
        }
        with self.results_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, RESULT_COLUMNS, delimiter="\t", lineterminator="\n")
            writer.writerow(sanitized)

    def _history(self) -> list[dict[str, Any]]:
        if not self.results_path.exists():
            return []
        with self.results_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        limit = int(self.config.run["history_limit"])
        candidates = [
            row
            for row in rows
            if row.get("candidate") not in {"", "baseline"}
            and row.get("screen_score")
            and row.get("decoding_config")
        ]

        def score(row: dict[str, str], field: str) -> float:
            try:
                return float(row.get(field, ""))
            except ValueError:
                return float("inf")

        ranked: list[tuple[dict[str, str], str]] = []
        validated = [row for row in candidates if row.get("validation_score")]
        ranked.extend(
            (row, "best_validation")
            for row in sorted(validated, key=lambda item: score(item, "validation_score"))[:4]
        )
        ranked.extend(
            (row, "best_screen")
            for row in sorted(candidates, key=lambda item: score(item, "screen_score"))[:4]
        )
        ranked.extend((row, "recent") for row in reversed(candidates))

        selected: list[dict[str, Any]] = []
        seen: set[str] = set()
        for row, selection_reason in ranked:
            key = f"{row.get('run_id', '')}/{row.get('candidate', '')}"
            if key in seen:
                continue
            seen.add(key)
            item: dict[str, Any] = {
                "candidate": key,
                "selected_as": selection_reason,
                "screen_score": row.get("screen_score", ""),
                "validation_score": row.get("validation_score", ""),
                "quality_score": row.get("quality_score", ""),
                "loop_incidence": row.get("loop_incidence", ""),
                "truncation_rate": row.get("truncation_rate", ""),
                "reasoning_efficiency_penalty": row.get(
                    "reasoning_efficiency_penalty", ""
                ),
                "empty_answer_rate": row.get("empty_answer_rate", ""),
                "status": row.get("status", ""),
                "decoding_config": row.get("decoding_config", ""),
                "description": row.get("description", ""),
            }
            quality_path = (
                self.runs_dir
                / row.get("run_id", "")
                / row.get("candidate", "")
                / "quality.json"
            )
            if quality_path.exists():
                try:
                    quality = json.loads(quality_path.read_text(encoding="utf-8"))
                    item["quality_gate"] = {
                        field: quality.get(field)
                        for field in (
                            "passed",
                            "deterministic_pass_rate",
                            "baseline_deterministic_pass_rate",
                            "mean_quality",
                            "baseline_mean_quality",
                            "loss_rate",
                            "critical_failures",
                        )
                    }
                except (OSError, json.JSONDecodeError):
                    pass
            selected.append(item)
            if len(selected) >= limit:
                break
        return selected

    def _tried_decoding_configurations(self) -> list[dict[str, float | int]]:
        if not self.results_path.exists():
            return []
        tried: list[dict[str, float | int]] = []
        seen: set[str] = set()
        with self.results_path.open(encoding="utf-8", newline="") as handle:
            rows = csv.DictReader(handle, delimiter="\t")
            for row in rows:
                raw = row.get("decoding_config", "")
                if not raw:
                    continue
                try:
                    decoding = validate_decoding_values(
                        json.loads(raw), self.config.decoding_search
                    )
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                key = decoding_hash(decoding)
                if key not in seen:
                    seen.add(key)
                    tried.append(decoding.to_dict())
        return tried

    def _interaction_anchor_configurations(self) -> list[dict[str, float | int]]:
        if not self.results_path.exists():
            return []
        ranked: list[tuple[float, DecodingConfig]] = []
        current = self._decoding()
        with self.results_path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                raw_config = row.get("decoding_config", "")
                raw_score = row.get("validation_score", "")
                if not raw_config or not raw_score or row.get("candidate") == "baseline":
                    continue
                try:
                    decoding = validate_decoding_values(
                        json.loads(raw_config), self.config.decoding_search
                    )
                    validation_score = float(raw_score)
                except (json.JSONDecodeError, TypeError, ValueError):
                    continue
                if decoding != current:
                    ranked.append((validation_score, decoding))
        anchors: list[dict[str, float | int]] = []
        seen: set[str] = set()
        for _score, decoding in sorted(ranked, key=lambda item: item[0]):
            key = decoding_hash(decoding)
            if key in seen:
                continue
            seen.add(key)
            anchors.append(decoding.to_dict())
            if len(anchors) == 2:
                break
        return anchors

    def _decoding(self) -> DecodingConfig:
        return self.config.load_decoding()

    def _make_deepseek(self, maximum_total_cost: float) -> DeepSeekClient:
        deepseek_config = self.config.deepseek
        budget = CostBudget(
            maximum_usd=maximum_total_cost,
            used_usd=self.state.total_estimated_cost_usd,
            input_rate=float(deepseek_config["input_usd_per_million_tokens"]),
            output_rate=float(deepseek_config["output_usd_per_million_tokens"]),
        )
        if self.deepseek_factory is not None:
            return self.deepseek_factory(budget)
        return DeepSeekClient(
            base_url=str(deepseek_config["base_url"]),
            api_key=require_deepseek_key(),
            model=str(deepseek_config["model"]),
            temperature=float(deepseek_config["temperature"]),
            input_rate=float(deepseek_config["input_usd_per_million_tokens"]),
            output_rate=float(deepseek_config["output_usd_per_million_tokens"]),
            budget=budget,
            usage_callback=self._record_usage,
        )

    def _record_usage(self, usage: ApiUsage) -> None:
        self.state.add_usage(usage)
        save_state(self.state_path, self.state)

    def preflight(self, *, require_gpu: bool = True) -> dict[str, Any]:
        report = self.ollama.preflight(
            decoding=self._decoding(),
            minimum_gpu_fraction=float(self.config.ollama["minimum_gpu_fraction"]),
            require_gpu=require_gpu,
        )
        atomic_write_json(self.root / ".autoresearch" / "preflight.json", report)
        return report

    def calibrate(self, *, require_gpu: bool = True, max_cost_usd: float | None = None) -> dict:
        # Decoding is now the research variable, so calibration must not pre-select a winner.
        # It only verifies that the complete mutable configuration works on this Ollama build.
        report = self.preflight(require_gpu=require_gpu)
        decoding = self._decoding()
        self.state.calibrated_decoding_sha256 = decoding_hash(decoding)
        save_state(self.state_path, self.state)
        return {
            "status": "ready",
            "decoding_sha256": self.state.calibrated_decoding_sha256,
            "decoding_config": decoding.to_dict(),
            "preflight": report,
        }

    def baseline(self, *, require_gpu: bool = True) -> dict[str, Any]:
        decoding = self._decoding()
        if self.state.calibrated_decoding_sha256 != decoding_hash(decoding):
            raise RuntimeError("Run `research.py calibrate` before establishing the baseline")
        self.preflight(require_gpu=require_gpu)
        prompt = (self.root / "system_prompt.md").read_text(encoding="utf-8")
        run_dir = self.runs_dir / utc_run_id("baseline")
        screen = self.evaluator.evaluate(
            system_prompt=prompt,
            decoding=decoding,
            split="screen",
            seeds=self.config.ollama["screen_seeds"],
            output_dir=run_dir / "screen",
        )
        validation = self.evaluator.evaluate(
            system_prompt=prompt,
            decoding=decoding,
            split="validation",
            seeds=self.config.ollama["validation_seeds"],
            output_dir=run_dir / "validation",
        )
        self.state.baseline_screen_path = str(run_dir / "screen" / "report.json")
        self.state.baseline_validation_path = str(run_dir / "validation" / "report.json")
        self.state.accepted_screen_path = self.state.baseline_screen_path
        self.state.accepted_validation_path = self.state.baseline_validation_path
        self.state.fixed_prompt_sha256 = prompt_hash(prompt)
        self.state.current_decoding_sha256 = decoding_hash(decoding)
        # A fresh baseline starts a new comparable research series while preserving
        # cumulative API cost accounting from calibration and earlier runs.
        self.state.run_id = None
        self.state.run_start_estimated_cost_usd = None
        self.state.iteration = 0
        self.state.boss_calls = 0
        self.state.accepted_count = 0
        save_state(self.state_path, self.state)
        self._append_result(
            {
                "run_id": run_dir.name,
                "commit": "baseline",
                "candidate": "baseline",
                "screen_score": f"{screen.repetition_score:.6f}",
                "validation_score": f"{validation.repetition_score:.6f}",
                "quality_score": "",
                "loop_incidence": f"{validation.loop_incidence:.6f}",
                "truncation_rate": f"{validation.truncation_rate:.6f}",
                "reasoning_efficiency_penalty": (
                    f"{validation.mean_reasoning_efficiency_penalty:.6f}"
                ),
                "empty_answer_rate": f"{validation.empty_answer_rate:.6f}",
                "status": "baseline",
                "boss_tokens": 0,
                "estimated_cost_usd": "0.000000",
                "cumulative_cost_usd": f"{self.state.total_estimated_cost_usd:.6f}",
                "decoding_config": json.dumps(decoding.to_dict(), sort_keys=True),
                "description": f"decoding baseline {self.state.current_decoding_sha256[:12]}",
            }
        )
        return {"screen": screen.summary(), "validation": validation.summary()}

    def _require_baseline(self) -> tuple[EvaluationReport, EvaluationReport]:
        if not self.state.baseline_validation_path or not self.state.accepted_screen_path:
            raise RuntimeError("Run `research.py baseline` before autonomous research")
        if not self.state.accepted_validation_path:
            raise RuntimeError("Accepted validation state is missing")
        return (
            load_report(Path(self.state.baseline_validation_path)),
            load_report(Path(self.state.accepted_validation_path)),
        )

    def run_with_boss(
        self,
        *,
        max_hours: float,
        max_cost_usd: float,
        require_gpu: bool = True,
    ) -> dict[str, Any]:
        baseline_validation, accepted_validation = self._require_baseline()
        if not self.state.accepted_screen_path:
            raise RuntimeError("Accepted screen state is missing")
        fixed_prompt = (self.root / "system_prompt.md").read_text(encoding="utf-8")
        current_decoding = self._decoding()
        if prompt_hash(fixed_prompt) != self.state.fixed_prompt_sha256:
            raise RuntimeError(
                "Fixed system_prompt.md changed; re-run calibrate and baseline or restore it"
            )
        if decoding_hash(current_decoding) != self.state.current_decoding_sha256:
            raise RuntimeError(
                "decoding_config.toml changed outside the research loop; re-run calibrate and baseline"
            )
        self.preflight(require_gpu=require_gpu)
        if self.state.run_id is None:
            self.state.run_id = utc_run_id("research")
            self.state.run_start_estimated_cost_usd = (
                self.state.total_estimated_cost_usd
            )
            self.state.iteration = 0
            self.state.boss_calls = 0
            self.state.accepted_count = 0
        elif self.state.run_start_estimated_cost_usd is None:
            # Compatibility for a process interrupted under schema v3.
            self.state.run_start_estimated_cost_usd = (
                self.state.total_estimated_cost_usd
            )
        run_id = self.state.run_id
        run_start_cost = float(self.state.run_start_estimated_cost_usd)
        run_cost_ceiling = run_start_cost + max_cost_usd
        save_state(self.state_path, self.state)
        boss = self._make_deepseek(run_cost_ceiling)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + max_hours * 3600
        max_boss_calls = int(self.config.run["max_boss_calls"])
        max_failures = int(self.config.run["max_consecutive_failures"])
        failures = 0
        stop_reason = "completed"

        def limits_reached() -> bool:
            return (
                time.monotonic() >= deadline
                or self.state.total_estimated_cost_usd >= run_cost_ceiling
                or self.state.boss_calls >= max_boss_calls
            )

        def active_candidate_limit_reached() -> bool:
            # The boss-call cap prevents starting another candidate; it must not abort
            # evaluation of the final candidate already paid for and proposed.
            return (
                time.monotonic() >= deadline
                or self.state.total_estimated_cost_usd >= run_cost_ceiling
            )

        try:
            while not limits_reached():
                self.state.iteration += 1
                candidate_id = f"candidate-{self.state.iteration:04d}"
                attempt_dir = run_dir / candidate_id
                attempt_dir.mkdir(parents=True, exist_ok=True)
                accepted_screen = load_report(Path(self.state.accepted_screen_path))
                boss_usage = ApiUsage()
                quality_report: QualityReport | None = None
                candidate_screen: EvaluationReport | None = None
                candidate_validation: EvaluationReport | None = None
                candidate = None
                try:
                    tried_configurations = self._tried_decoding_configurations()
                    interaction_threshold = int(
                        self.config.run.get("interaction_after_unique_configs", 9)
                    )
                    quality_recovery_threshold = int(
                        self.config.run.get("quality_recovery_after_unique_configs", 16)
                    )
                    if len(tried_configurations) < interaction_threshold:
                        required_changed_fields = 1
                    elif len(tried_configurations) < quality_recovery_threshold:
                        required_changed_fields = 2
                    else:
                        required_changed_fields = 3
                    # Count proposal cycles before the call so malformed/duplicate
                    # outputs cannot bypass the configured boss-call ceiling.
                    self.state.boss_calls += 1
                    save_state(self.state_path, self.state)
                    candidate, boss_usage = boss.propose(
                        current_decoding=current_decoding,
                        decoding_bounds=self.config.decoding_search,
                        screen_summary=accepted_screen.summary(),
                        loop_examples=repeated_span_examples(
                            accepted_screen.samples,
                            ngram_size=int(self.config.metrics["ngram_size"]),
                        ),
                        history=self._history(),
                        tried_configurations=tried_configurations,
                        required_changed_fields=required_changed_fields,
                        interaction_anchor_configurations=(
                            self._interaction_anchor_configurations()
                            if required_changed_fields > 1
                            else []
                        ),
                        max_tokens=int(self.config.deepseek["boss_max_tokens"]),
                        max_description_characters=int(
                            self.config.deepseek["max_description_characters"]
                        ),
                    )
                    atomic_write_text(
                        attempt_dir / "candidate_decoding_config.toml",
                        render_decoding_config(candidate.decoding_config),
                    )
                    atomic_write_json(attempt_dir / "proposal.json", asdict(candidate))
                    candidate_screen = self.evaluator.evaluate(
                        system_prompt=fixed_prompt,
                        decoding=candidate.decoding_config,
                        split="screen",
                        seeds=self.config.ollama["screen_seeds"],
                        output_dir=attempt_dir / "screen",
                        stop_check=active_candidate_limit_reached,
                    )
                    denominator = max(accepted_screen.repetition_score, 1e-12)
                    relative_improvement = (
                        accepted_screen.repetition_score - candidate_screen.repetition_score
                    ) / denominator
                    screen_pass = (
                        candidate_screen.deterministic_pass_rate
                        >= accepted_screen.deterministic_pass_rate
                        and relative_improvement
                        >= float(self.config.quality["screen_relative_improvement"])
                    )
                    if not screen_pass:
                        status = "discard"
                        description = (
                            f"{candidate.hypothesis}; screen improvement "
                            f"{relative_improvement:.2%} or deterministic quality below floor"
                        )
                    else:
                        candidate_validation = self.evaluator.evaluate(
                            system_prompt=fixed_prompt,
                            decoding=candidate.decoding_config,
                            split="validation",
                            seeds=self.config.ollama["validation_seeds"],
                            output_dir=attempt_dir / "validation",
                            stop_check=active_candidate_limit_reached,
                        )
                        quality_report, _judge_usage = compare_quality(
                            cases=self.cases,
                            baseline=baseline_validation,
                            candidate=candidate_validation,
                            judge=boss,
                            judge_max_tokens=int(self.config.deepseek["judge_max_tokens"]),
                            max_loss_rate=float(self.config.quality["max_judge_loss_rate"]),
                            quality_tolerance=float(self.config.quality["quality_tolerance"]),
                            output_path=attempt_dir / "quality.json",
                            stop_check=active_candidate_limit_reached,
                        )
                        improvement = (
                            accepted_validation.repetition_score
                            - candidate_validation.repetition_score
                        )
                        upper_bound = paired_bootstrap_upper_bound(
                            accepted_validation.samples,
                            candidate_validation.samples,
                            samples=int(self.config.quality["bootstrap_samples"]),
                            confidence=float(self.config.quality["bootstrap_confidence"]),
                            weights=(
                                float(self.config.metrics["loop_fraction_weight"]),
                                float(self.config.metrics["loop_incidence_weight"]),
                                float(self.config.metrics["truncation_weight"]),
                                float(
                                    self.config.metrics[
                                        "reasoning_efficiency_weight"
                                    ]
                                ),
                            ),
                        )
                        keep = (
                            quality_report.passed
                            and improvement
                            >= float(
                                self.config.quality["validation_absolute_improvement"]
                            )
                            and upper_bound < 0
                        )
                        if keep:
                            atomic_write_text(
                                self.root / "decoding_config.toml",
                                render_decoding_config(candidate.decoding_config),
                            )
                            commit = commit_decoding_config(self.root, candidate_id)
                            current_decoding = candidate.decoding_config
                            accepted_screen = candidate_screen
                            accepted_validation = candidate_validation
                            self.state.accepted_screen_path = str(
                                attempt_dir / "screen" / "report.json"
                            )
                            self.state.accepted_validation_path = str(
                                attempt_dir / "validation" / "report.json"
                            )
                            self.state.current_decoding_sha256 = decoding_hash(current_decoding)
                            self.state.accepted_count += 1
                            status = "keep"
                            description = (
                                f"{candidate.hypothesis}; validation improvement "
                                f"{improvement:.6f}, bootstrap upper {upper_bound:.6f}"
                            )
                        else:
                            commit = ""
                            status = "discard"
                            description = (
                                f"{candidate.hypothesis}; validation improvement "
                                f"{improvement:.6f}, bootstrap upper {upper_bound:.6f}, "
                                f"quality_pass={quality_report.passed}"
                            )
                    if status != "keep":
                        commit = ""
                    failures = 0
                    self._append_result(
                        {
                            "run_id": run_id,
                            "commit": commit,
                            "candidate": candidate_id,
                            "screen_score": f"{candidate_screen.repetition_score:.6f}",
                            "validation_score": (
                                f"{candidate_validation.repetition_score:.6f}"
                                if candidate_validation
                                else ""
                            ),
                            "quality_score": (
                                f"{quality_report.mean_quality:.3f}" if quality_report else ""
                            ),
                            "loop_incidence": (
                                f"{candidate_validation.loop_incidence:.6f}"
                                if candidate_validation
                                else f"{candidate_screen.loop_incidence:.6f}"
                            ),
                            "truncation_rate": (
                                f"{candidate_validation.truncation_rate:.6f}"
                                if candidate_validation
                                else f"{candidate_screen.truncation_rate:.6f}"
                            ),
                            "reasoning_efficiency_penalty": (
                                f"{candidate_validation.mean_reasoning_efficiency_penalty:.6f}"
                                if candidate_validation
                                else f"{candidate_screen.mean_reasoning_efficiency_penalty:.6f}"
                            ),
                            "empty_answer_rate": (
                                f"{candidate_validation.empty_answer_rate:.6f}"
                                if candidate_validation
                                else f"{candidate_screen.empty_answer_rate:.6f}"
                            ),
                            "status": status,
                            "boss_tokens": (
                                boss_usage.prompt_tokens + boss_usage.completion_tokens
                            ),
                            "estimated_cost_usd": (
                                f"{self.state.total_estimated_cost_usd - run_start_cost:.6f}"
                            ),
                            "cumulative_cost_usd": (
                                f"{self.state.total_estimated_cost_usd:.6f}"
                            ),
                            "decoding_config": json.dumps(
                                candidate.decoding_config.to_dict(), sort_keys=True
                            ),
                            "description": description,
                        }
                    )
                    save_state(self.state_path, self.state)
                except Exception as exc:
                    # KeyboardInterrupt and SystemExit remain unhandled by design.
                    if isinstance(exc, ProposalRejected):
                        # A reachable API returning unusable research proposals is a
                        # boss/search failure, not an Ollama or network outage. The
                        # cost and boss-call limits already bound it safely.
                        failures = 0
                    else:
                        failures += 1
                    self._append_result(
                        {
                            "run_id": run_id,
                            "commit": "",
                            "candidate": candidate_id,
                            "screen_score": (
                                f"{candidate_screen.repetition_score:.6f}"
                                if candidate_screen
                                else ""
                            ),
                            "validation_score": "",
                            "quality_score": "",
                            "loop_incidence": "",
                            "truncation_rate": "",
                            "reasoning_efficiency_penalty": "",
                            "empty_answer_rate": "",
                            "status": "crash",
                            "boss_tokens": boss_usage.prompt_tokens
                            + boss_usage.completion_tokens,
                            "estimated_cost_usd": (
                                f"{self.state.total_estimated_cost_usd - run_start_cost:.6f}"
                            ),
                            "cumulative_cost_usd": (
                                f"{self.state.total_estimated_cost_usd:.6f}"
                            ),
                            "decoding_config": (
                                json.dumps(candidate.decoding_config.to_dict(), sort_keys=True)
                                if candidate is not None
                                else ""
                            ),
                            "description": f"{type(exc).__name__}: {exc}",
                        }
                    )
                    save_state(self.state_path, self.state)
                    if isinstance(exc, (BudgetExceeded, EvaluationStopped)):
                        stop_reason = str(exc)
                        break
                    if failures >= max_failures:
                        stop_reason = f"{failures} consecutive infrastructure failures"
                        break
            if limits_reached() and stop_reason == "completed":
                if time.monotonic() >= deadline:
                    stop_reason = "time limit reached"
                elif self.state.total_estimated_cost_usd >= run_cost_ceiling:
                    stop_reason = "cost limit reached"
                else:
                    stop_reason = "boss-call limit reached"
        except KeyboardInterrupt:
            stop_reason = "manual interrupt"
        finally:
            save_state(self.state_path, self.state)
            boss.close()
        summary = {
            "run_id": run_id,
            "stop_reason": stop_reason,
            "iterations": self.state.iteration,
            "accepted": self.state.accepted_count,
            "boss_calls": self.state.boss_calls,
            "estimated_cost_usd": (
                self.state.total_estimated_cost_usd - run_start_cost
            ),
            "cumulative_cost_usd": self.state.total_estimated_cost_usd,
            "current_decoding_sha256": self.state.current_decoding_sha256,
            "decoding_config": current_decoding.to_dict(),
        }
        atomic_write_json(run_dir / "summary.json", summary)
        # A normally completed CLI invocation is one budgeted run. If the process is
        # killed before this point, the persisted run id/start cost allow safe resume
        # against the same remaining budget.
        self.state.run_id = None
        self.state.run_start_estimated_cost_usd = None
        save_state(self.state_path, self.state)
        return summary

    def _algorithm_study_name(self, fixed_prompt: str) -> str:
        payload = {
            "model": self.config.ollama["model"],
            "prompt_sha256": prompt_hash(fixed_prompt),
            "benchmark_sha256": hashlib.sha256(
                self.benchmark_path.read_bytes()
            ).hexdigest(),
            "case_selection": self.config.benchmark,
            "metrics": self.config.metrics,
            "search": self.config.decoding_search,
            "algorithm": self.config.algorithm,
        }
        digest = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:16]
        return f"qwen-decoding-{digest}"

    def _optuna_distributions(self) -> dict[str, Any]:
        bounds = self.config.decoding_search
        algorithm = self.config.algorithm
        return {
            "temperature": FloatDistribution(
                float(bounds["temperature"]["min"]),
                float(bounds["temperature"]["max"]),
                step=float(algorithm["temperature_step"]),
            ),
            "top_p": FloatDistribution(
                float(bounds["top_p"]["min"]),
                float(bounds["top_p"]["max"]),
                step=float(algorithm["top_p_step"]),
            ),
            "top_k": IntDistribution(
                int(bounds["top_k"]["min"]),
                int(bounds["top_k"]["max"]),
                step=int(algorithm["top_k_step"]),
            ),
            "min_p": FloatDistribution(
                float(bounds["min_p"]["min"]),
                float(bounds["min_p"]["max"]),
                step=float(algorithm["min_p_step"]),
            ),
        }

    def run(
        self,
        *,
        max_hours: float,
        max_trials: int,
        require_gpu: bool = True,
    ) -> dict[str, Any]:
        """Run one exclusive algorithmic optimizer process."""
        lock_path = self.root / ".autoresearch" / "run.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_handle = lock_path.open("a+", encoding="utf-8")
        try:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise RuntimeError(
                    "Another research.py run process is already active; "
                    "wait for it or interrupt it before starting a second run"
                ) from exc
            lock_handle.seek(0)
            lock_handle.truncate()
            lock_handle.write(f"pid={os.getpid()}\n")
            lock_handle.flush()
            return self._run_algorithm(
                max_hours=max_hours,
                max_trials=max_trials,
                require_gpu=require_gpu,
            )
        finally:
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            finally:
                lock_handle.close()

    def _run_algorithm(
        self,
        *,
        max_hours: float,
        max_trials: int,
        require_gpu: bool = True,
    ) -> dict[str, Any]:
        """Optimize decoding with constrained multivariate TPE and no boss LLM."""
        baseline_validation, accepted_validation = self._require_baseline()
        if not self.state.baseline_screen_path or not self.state.accepted_screen_path:
            raise RuntimeError("Baseline screen state is missing")
        baseline_screen = load_report(Path(self.state.baseline_screen_path))
        accepted_screen = load_report(Path(self.state.accepted_screen_path))
        fixed_prompt = (self.root / "system_prompt.md").read_text(encoding="utf-8")
        current_decoding = self._decoding()
        if prompt_hash(fixed_prompt) != self.state.fixed_prompt_sha256:
            raise RuntimeError(
                "Fixed system_prompt.md changed; re-run calibrate and baseline or restore it"
            )
        if decoding_hash(current_decoding) != self.state.current_decoding_sha256:
            raise RuntimeError(
                "decoding_config.toml changed outside the research loop; re-run calibrate and baseline"
            )
        self.preflight(require_gpu=require_gpu)
        if self.state.run_id is None:
            self.state.run_id = utc_run_id("algorithm")
            self.state.iteration = 0
            self.state.boss_calls = 0
            self.state.accepted_count = 0
        run_id = self.state.run_id
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        save_state(self.state_path, self.state)
        deadline = time.monotonic() + max_hours * 3600

        def constraints_func(trial: optuna.trial.FrozenTrial) -> tuple[float, ...]:
            raw = trial.user_attrs.get("constraints")
            if not isinstance(raw, (list, tuple)) or len(raw) != 3:
                return (1.0, 1.0, 1.0)
            return tuple(float(value) for value in raw)

        def make_sampler(seed: int) -> TPESampler:
            return TPESampler(
                seed=seed,
                n_startup_trials=int(self.config.algorithm["startup_trials"]),
                multivariate=True,
                constraints_func=constraints_func,
            )

        base_sampler_seed = int(self.config.algorithm["sampler_seed"])
        storage_path = (self.root / ".autoresearch" / "optuna.db").resolve()
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_url = f"sqlite:///{storage_path}"
        study_name = self._algorithm_study_name(fixed_prompt)
        study = optuna.create_study(
            study_name=study_name,
            storage=storage_url,
            load_if_exists=True,
            direction="minimize",
            sampler=make_sampler(base_sampler_seed),
        )
        distributions = self._optuna_distributions()
        initial_trials = study.get_trials(deepcopy=False)
        if not initial_trials:
            study.add_trial(
                create_trial(
                    params=current_decoding.to_dict(),
                    distributions=distributions,
                    value=baseline_screen.repetition_score,
                    user_attrs={
                        "constraints": [0.0, 0.0, 0.0],
                        "source": "baseline",
                        "screen_score": baseline_screen.repetition_score,
                        "validation_score": baseline_validation.repetition_score,
                    },
                    system_attrs={"constraints": [0.0, 0.0, 0.0]},
                )
            )
        else:
            legacy_baseline = next(
                (
                    trial
                    for trial in initial_trials
                    if trial.user_attrs.get("source") == "baseline"
                    and "constraints" not in trial.system_attrs
                ),
                None,
            )
            already_migrated = any(
                trial.user_attrs.get("source") == "baseline_constraint_migration"
                for trial in initial_trials
            )
            if legacy_baseline is not None and not already_migrated:
                # Finished Optuna trials are immutable. Add one corrected baseline
                # observation instead of attempting an unsafe in-place update.
                study.add_trial(
                    create_trial(
                        params=legacy_baseline.params,
                        distributions=distributions,
                        value=float(legacy_baseline.value),
                        user_attrs={
                            **legacy_baseline.user_attrs,
                            "source": "baseline_constraint_migration",
                        },
                        system_attrs={"constraints": [0.0, 0.0, 0.0]},
                    )
                )
        existing_trial_count = len(study.get_trials(deepcopy=False))
        if existing_trial_count:
            # Optuna persists trials but not the sampler RNG state. Advancing the
            # deterministic seed on resume prevents replaying the same startup
            # sequence; the exact-config cache remains a second line of defense.
            study = optuna.load_study(
                study_name=study_name,
                storage=storage_url,
                sampler=make_sampler(base_sampler_seed + existing_trial_count),
            )
        starting_trial_count = len(study.get_trials(deepcopy=False))
        stop_reason = "trial limit reached"

        def stopped() -> bool:
            return time.monotonic() >= deadline

        def objective(trial: optuna.Trial) -> float:
            nonlocal current_decoding, accepted_screen, accepted_validation
            values = {
                "temperature": trial.suggest_float(
                    "temperature",
                    distributions["temperature"].low,
                    distributions["temperature"].high,
                    step=distributions["temperature"].step,
                ),
                "top_p": trial.suggest_float(
                    "top_p",
                    distributions["top_p"].low,
                    distributions["top_p"].high,
                    step=distributions["top_p"].step,
                ),
                "top_k": trial.suggest_int(
                    "top_k",
                    distributions["top_k"].low,
                    distributions["top_k"].high,
                    step=distributions["top_k"].step,
                ),
                "min_p": trial.suggest_float(
                    "min_p",
                    distributions["min_p"].low,
                    distributions["min_p"].high,
                    step=distributions["min_p"].step,
                ),
            }
            decoding = validate_decoding_values(values, self.config.decoding_search)
            candidate_id = f"trial-{trial.number:04d}"
            attempt_dir = run_dir / candidate_id
            attempt_dir.mkdir(parents=True, exist_ok=True)
            self.state.iteration += 1
            save_state(self.state_path, self.state)

            for previous in study.get_trials(
                deepcopy=False, states=(TrialState.COMPLETE,)
            ):
                if previous.number == trial.number or previous.params != trial.params:
                    continue
                previous_constraints = previous.user_attrs.get(
                    "constraints", [1.0, 1.0, 1.0]
                )
                trial.set_user_attr("constraints", previous_constraints)
                trial.set_user_attr("duplicate_of", previous.number)
                self._append_result(
                    {
                        "run_id": run_id,
                        "candidate": candidate_id,
                        "screen_score": f"{float(previous.value):.6f}",
                        "status": "duplicate",
                        "boss_tokens": 0,
                        "estimated_cost_usd": "0.000000",
                        "cumulative_cost_usd": (
                            f"{self.state.total_estimated_cost_usd:.6f}"
                        ),
                        "decoding_config": json.dumps(
                            decoding.to_dict(), sort_keys=True
                        ),
                        "description": f"Optuna duplicate of trial {previous.number}",
                    }
                )
                return float(previous.value)

            atomic_write_text(
                attempt_dir / "candidate_decoding_config.toml",
                render_decoding_config(decoding),
            )
            candidate_screen: EvaluationReport | None = None
            candidate_validation: EvaluationReport | None = None
            try:
                candidate_screen = self.evaluator.evaluate(
                    system_prompt=fixed_prompt,
                    decoding=decoding,
                    split="screen",
                    seeds=self.config.ollama["screen_seeds"],
                    output_dir=attempt_dir / "screen",
                    stop_check=stopped,
                )
                screen_constraints = [
                    baseline_screen.deterministic_pass_rate
                    - candidate_screen.deterministic_pass_rate,
                    candidate_screen.empty_answer_rate
                    - baseline_screen.empty_answer_rate,
                    candidate_screen.truncation_rate - baseline_screen.truncation_rate,
                ]
                trial.set_user_attr("constraints", screen_constraints)
                trial.set_user_attr(
                    "screen_summary", candidate_screen.summary()
                )
                relative_improvement = (
                    baseline_screen.repetition_score
                    - candidate_screen.repetition_score
                ) / max(baseline_screen.repetition_score, 1e-12)
                screen_pass = all(value <= 0 for value in screen_constraints) and (
                    relative_improvement
                    >= float(self.config.quality["screen_relative_improvement"])
                )
                status = "screen_discard"
                description = (
                    f"TPE trial; screen improvement {relative_improvement:.2%}; "
                    f"constraints={screen_constraints}"
                )
                commit = ""
                upper_bound: float | None = None
                if screen_pass:
                    candidate_validation = self.evaluator.evaluate(
                        system_prompt=fixed_prompt,
                        decoding=decoding,
                        split="validation",
                        seeds=self.config.ollama["validation_seeds"],
                        output_dir=attempt_dir / "validation",
                        stop_check=stopped,
                    )
                    validation_constraints = [
                        baseline_validation.deterministic_pass_rate
                        - candidate_validation.deterministic_pass_rate,
                        candidate_validation.empty_answer_rate
                        - baseline_validation.empty_answer_rate,
                        candidate_validation.truncation_rate
                        - baseline_validation.truncation_rate,
                    ]
                    combined_constraints = [
                        max(screen_value, validation_value)
                        for screen_value, validation_value in zip(
                            screen_constraints, validation_constraints
                        )
                    ]
                    trial.set_user_attr("constraints", combined_constraints)
                    trial.set_user_attr(
                        "validation_summary", candidate_validation.summary()
                    )
                    improvement = (
                        accepted_validation.repetition_score
                        - candidate_validation.repetition_score
                    )
                    upper_bound = paired_bootstrap_upper_bound(
                        accepted_validation.samples,
                        candidate_validation.samples,
                        samples=int(self.config.quality["bootstrap_samples"]),
                        confidence=float(
                            self.config.quality["bootstrap_confidence"]
                        ),
                        weights=(
                            float(self.config.metrics["loop_fraction_weight"]),
                            float(self.config.metrics["loop_incidence_weight"]),
                            float(self.config.metrics["truncation_weight"]),
                            float(
                                self.config.metrics[
                                    "reasoning_efficiency_weight"
                                ]
                            ),
                        ),
                    )
                    keep = (
                        all(value <= 0 for value in combined_constraints)
                        and improvement
                        >= float(
                            self.config.quality[
                                "validation_absolute_improvement"
                            ]
                        )
                        and upper_bound < 0
                    )
                    if keep:
                        atomic_write_text(
                            self.root / "decoding_config.toml",
                            render_decoding_config(decoding),
                        )
                        commit = commit_decoding_config(self.root, candidate_id)
                        current_decoding = decoding
                        accepted_screen = candidate_screen
                        accepted_validation = candidate_validation
                        self.state.accepted_screen_path = str(
                            attempt_dir / "screen" / "report.json"
                        )
                        self.state.accepted_validation_path = str(
                            attempt_dir / "validation" / "report.json"
                        )
                        self.state.current_decoding_sha256 = decoding_hash(decoding)
                        self.state.accepted_count += 1
                        status = "keep"
                    else:
                        status = "discard"
                    description = (
                        f"TPE trial; validation improvement {improvement:.6f}; "
                        f"bootstrap upper {upper_bound:.6f}; "
                        f"constraints={combined_constraints}"
                    )
                report = candidate_validation or candidate_screen
                self._append_result(
                    {
                        "run_id": run_id,
                        "commit": commit,
                        "candidate": candidate_id,
                        "screen_score": f"{candidate_screen.repetition_score:.6f}",
                        "validation_score": (
                            f"{candidate_validation.repetition_score:.6f}"
                            if candidate_validation
                            else ""
                        ),
                        "quality_score": f"{report.deterministic_pass_rate:.3f}",
                        "loop_incidence": f"{report.loop_incidence:.6f}",
                        "truncation_rate": f"{report.truncation_rate:.6f}",
                        "reasoning_efficiency_penalty": (
                            f"{report.mean_reasoning_efficiency_penalty:.6f}"
                        ),
                        "empty_answer_rate": f"{report.empty_answer_rate:.6f}",
                        "status": status,
                        "boss_tokens": 0,
                        "estimated_cost_usd": "0.000000",
                        "cumulative_cost_usd": (
                            f"{self.state.total_estimated_cost_usd:.6f}"
                        ),
                        "decoding_config": json.dumps(
                            decoding.to_dict(), sort_keys=True
                        ),
                        "description": description,
                    }
                )
                save_state(self.state_path, self.state)
                return candidate_screen.repetition_score
            except Exception as exc:
                self._append_result(
                    {
                        "run_id": run_id,
                        "candidate": candidate_id,
                        "screen_score": (
                            f"{candidate_screen.repetition_score:.6f}"
                            if candidate_screen
                            else ""
                        ),
                        "status": "crash",
                        "boss_tokens": 0,
                        "estimated_cost_usd": "0.000000",
                        "cumulative_cost_usd": (
                            f"{self.state.total_estimated_cost_usd:.6f}"
                        ),
                        "decoding_config": json.dumps(
                            decoding.to_dict(), sort_keys=True
                        ),
                        "description": f"{type(exc).__name__}: {exc}",
                    }
                )
                save_state(self.state_path, self.state)
                raise

        try:
            study.optimize(
                objective,
                n_trials=max_trials,
                timeout=max_hours * 3600,
                catch=(Exception,),
                show_progress_bar=False,
            )
            if time.monotonic() >= deadline:
                stop_reason = "time limit reached"
        except KeyboardInterrupt:
            stop_reason = "manual interrupt"

        completed = [
            trial
            for trial in study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
            if all(value <= 0 for value in constraints_func(trial))
        ]
        best = min(completed, key=lambda row: float(row.value)) if completed else None
        summary = {
            "run_id": run_id,
            "optimizer": "optuna-multivariate-tpe",
            "stop_reason": stop_reason,
            "new_trials": len(study.get_trials(deepcopy=False)) - starting_trial_count,
            "total_study_trials": len(study.get_trials(deepcopy=False)),
            "accepted": self.state.accepted_count,
            "api_cost_usd": 0.0,
            "best_feasible_screen_score": float(best.value) if best else None,
            "current_decoding_sha256": self.state.current_decoding_sha256,
            "decoding_config": current_decoding.to_dict(),
            "study_name": study.study_name,
        }
        atomic_write_json(run_dir / "summary.json", summary)
        self.state.run_id = None
        save_state(self.state_path, self.state)
        return summary

    def validate(self, *, require_gpu: bool = True) -> dict:
        baseline_validation, accepted_validation = self._require_baseline()
        self.preflight(require_gpu=require_gpu)
        decoding = self._decoding()
        prompt = (self.root / "system_prompt.md").read_text(encoding="utf-8")
        if prompt_hash(prompt) != self.state.fixed_prompt_sha256:
            raise RuntimeError("Fixed system_prompt.md no longer matches the baseline")
        if decoding_hash(decoding) != self.state.current_decoding_sha256:
            raise RuntimeError("decoding_config.toml no longer matches the accepted state")
        run_dir = self.runs_dir / utc_run_id("validation")
        fresh = self.evaluator.evaluate(
            system_prompt=prompt,
            decoding=decoding,
            split="validation",
            seeds=self.config.ollama["validation_seeds"],
            output_dir=run_dir,
        )
        quality_constraints = {
            "deterministic_pass_regression": (
                baseline_validation.deterministic_pass_rate
                - fresh.deterministic_pass_rate
            ),
            "empty_answer_regression": (
                fresh.empty_answer_rate - baseline_validation.empty_answer_rate
            ),
            "truncation_regression": (
                fresh.truncation_rate - baseline_validation.truncation_rate
            ),
        }
        quality_passed = all(value <= 0 for value in quality_constraints.values())
        upper = paired_bootstrap_upper_bound(
            baseline_validation.samples,
            fresh.samples,
            samples=int(self.config.quality["bootstrap_samples"]),
            confidence=float(self.config.quality["bootstrap_confidence"]),
            weights=(
                float(self.config.metrics["loop_fraction_weight"]),
                float(self.config.metrics["loop_incidence_weight"]),
                float(self.config.metrics["truncation_weight"]),
                float(self.config.metrics["reasoning_efficiency_weight"]),
            ),
        )
        summary = {
            "fresh": fresh.summary(),
            "previous_accepted": accepted_validation.summary(),
            "original_baseline": baseline_validation.summary(),
            "quality": {
                "passed": quality_passed,
                "constraints": quality_constraints,
                "api_judge_used": False,
            },
            "bootstrap_upper_vs_baseline": upper,
        }
        atomic_write_json(run_dir / "validation_summary.json", summary)
        return summary
