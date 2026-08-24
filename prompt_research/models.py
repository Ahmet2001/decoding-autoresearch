from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DecodingConfig:
    temperature: float
    top_p: float
    top_k: int
    min_p: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


@dataclass(frozen=True)
class BenchmarkCase:
    id: str
    split: str
    language: str
    category: str
    messages: list[dict[str, str]]
    validator: dict[str, Any]
    rubric: str


@dataclass
class ApiUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    estimated_cost_usd: float = 0.0

    def __add__(self, other: "ApiUsage") -> "ApiUsage":
        return ApiUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.estimated_cost_usd + other.estimated_cost_usd,
        )


@dataclass(frozen=True)
class Candidate:
    hypothesis: str
    decoding_config: DecodingConfig
    expected_effect: str


@dataclass
class GenerationResult:
    case_id: str
    split: str
    language: str
    category: str
    seed: int
    thinking: str
    content: str
    done_reason: str
    eval_count: int
    prompt_eval_count: int
    duration_seconds: float
    deterministic_pass: bool = False
    deterministic_reason: str = ""
    loop_fraction: float = 0.0
    sentence_loop: bool = False
    loop_detected: bool = False
    truncated: bool = False
    empty_answer: bool = False
    estimated_thinking_tokens: float = 0.0
    reasoning_efficiency_penalty: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvaluationReport:
    split: str
    decoding_sha256: str
    prompt_sha256: str
    repetition_score: float
    mean_loop_fraction: float
    loop_incidence: float
    truncation_rate: float
    deterministic_pass_rate: float
    samples: list[GenerationResult] = field(default_factory=list)
    output_token_limit: int = 0
    mean_reasoning_efficiency_penalty: float = 0.0
    empty_answer_rate: float = 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "split": self.split,
            "decoding_sha256": self.decoding_sha256,
            "prompt_sha256": self.prompt_sha256,
            "repetition_score": self.repetition_score,
            "mean_loop_fraction": self.mean_loop_fraction,
            "loop_incidence": self.loop_incidence,
            "truncation_rate": self.truncation_rate,
            "mean_reasoning_efficiency_penalty": self.mean_reasoning_efficiency_penalty,
            "empty_answer_rate": self.empty_answer_rate,
            "deterministic_pass_rate": self.deterministic_pass_rate,
            "output_token_limit": self.output_token_limit,
            "sample_count": len(self.samples),
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.summary()
        data["samples"] = [sample.to_dict() for sample in self.samples]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationReport":
        return cls(
            split=data["split"],
            decoding_sha256=data.get("decoding_sha256", data.get("profile", "legacy")),
            prompt_sha256=data["prompt_sha256"],
            repetition_score=float(data["repetition_score"]),
            mean_loop_fraction=float(data["mean_loop_fraction"]),
            loop_incidence=float(data["loop_incidence"]),
            truncation_rate=float(data["truncation_rate"]),
            deterministic_pass_rate=float(data["deterministic_pass_rate"]),
            samples=[GenerationResult(**sample) for sample in data.get("samples", [])],
            output_token_limit=int(data.get("output_token_limit", 0)),
            mean_reasoning_efficiency_penalty=float(
                data.get("mean_reasoning_efficiency_penalty", 0.0)
            ),
            empty_answer_rate=float(data.get("empty_answer_rate", 0.0)),
        )


@dataclass
class JudgeDecision:
    case_id: str
    seed: int
    baseline_score: float
    candidate_score: float
    preference: str
    critical_failure: bool
    reason: str
    usage: ApiUsage = field(default_factory=ApiUsage)


@dataclass
class QualityReport:
    passed: bool
    deterministic_pass_rate: float
    baseline_deterministic_pass_rate: float
    mean_quality: float
    baseline_mean_quality: float
    loss_rate: float
    critical_failures: int
    decisions: list[JudgeDecision] = field(default_factory=list)
    failure_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class ResearchState:
    schema_version: int = 4
    research_variable: str = "decoding_config"
    calibrated_decoding_sha256: str | None = None
    baseline_screen_path: str | None = None
    baseline_validation_path: str | None = None
    accepted_screen_path: str | None = None
    accepted_validation_path: str | None = None
    fixed_prompt_sha256: str | None = None
    current_decoding_sha256: str | None = None
    total_estimated_cost_usd: float = 0.0
    run_start_estimated_cost_usd: float | None = None
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    boss_calls: int = 0
    iteration: int = 0
    accepted_count: int = 0
    run_id: str | None = None

    def add_usage(self, usage: ApiUsage) -> None:
        self.total_estimated_cost_usd += usage.estimated_cost_usd
        self.total_prompt_tokens += usage.prompt_tokens
        self.total_completion_tokens += usage.completion_tokens

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ResearchState":
        known = cls.__dataclass_fields__
        return cls(**{key: value for key, value in data.items() if key in known})
