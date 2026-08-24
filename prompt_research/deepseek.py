from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from typing import Any, Callable

import httpx

from .config import DECODING_FIELDS, validate_decoding_values
from .models import (
    ApiUsage,
    BenchmarkCase,
    Candidate,
    DecodingConfig,
    GenerationResult,
    JudgeDecision,
)


class DeepSeekError(RuntimeError):
    pass


class BudgetExceeded(DeepSeekError):
    pass


class ProposalRejected(DeepSeekError):
    """The API worked, but the boss exhausted its structured-candidate retries."""

    pass


@dataclass
class CostBudget:
    maximum_usd: float
    used_usd: float
    input_rate: float
    output_rate: float

    @property
    def remaining_usd(self) -> float:
        return max(0.0, self.maximum_usd - self.used_usd)

    def reserve_check(self, messages: list[dict[str, str]], max_tokens: int) -> None:
        # Four characters per token is intentionally conservative for mixed EN/TR JSON prompts.
        input_upper = sum(len(message.get("content", "")) for message in messages) / 3.0 + 200
        estimated = (
            input_upper * self.input_rate / 1_000_000
            + max_tokens * self.output_rate / 1_000_000
        )
        if estimated > self.remaining_usd:
            raise BudgetExceeded(
                f"Next API call could cost ${estimated:.4f}, but only "
                f"${self.remaining_usd:.4f} remains in the configured budget"
            )

    def charge(self, usage: ApiUsage) -> None:
        self.used_usd += usage.estimated_cost_usd


def parse_candidate(
    payload: str,
    *,
    bounds: dict[str, Any],
    max_description_characters: int,
) -> Candidate:
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise DeepSeekError(f"Boss returned invalid JSON: {exc}") from exc
    expected = {"hypothesis", "decoding_config", "expected_effect"}
    if not isinstance(data, dict) or set(data) != expected:
        raise DeepSeekError(f"Boss JSON must contain exactly: {', '.join(sorted(expected))}")
    for field in ("hypothesis", "expected_effect"):
        if not isinstance(data[field], str) or not data[field].strip():
            raise DeepSeekError(f"Boss JSON field {field} must be a non-empty string")
        if len(data[field]) > max_description_characters:
            raise DeepSeekError(f"Boss JSON field {field} is too long")
    try:
        decoding = validate_decoding_values(data["decoding_config"], bounds)
    except ValueError as exc:
        raise DeepSeekError(f"Invalid candidate decoding config: {exc}") from exc
    return Candidate(
        data["hypothesis"].strip(), decoding, data["expected_effect"].strip()
    )


def build_boss_messages(
    *,
    current_decoding: DecodingConfig,
    decoding_bounds: dict[str, Any],
    screen_summary: dict[str, Any],
    loop_examples: list[dict[str, Any]],
    history: list[dict[str, Any]],
    tried_configurations: list[dict[str, float | int]],
    required_changed_fields: int,
    interaction_anchor_configurations: list[dict[str, float | int]],
) -> list[dict[str, str]]:
    """Build boss context only from explicitly supplied visible-screen information."""
    if required_changed_fields == 1:
        phase_instruction = (
            "This is the single-coordinate phase. Change exactly one field so the result is "
            "attributable."
        )
    elif required_changed_fields == 2:
        phase_instruction = (
            "The single-coordinate sweep did not satisfy both repetition and quality. This is the "
            "interaction phase: change exactly two fields. Use one field to retain a demonstrated "
            "repetition improvement and the other to recover termination, Turkish/English quality, "
            "or instruction adherence. The candidate must preserve at least one non-baseline "
            "field/value from interaction_anchor_configurations."
        )
    else:
        phase_instruction = (
            "The one- and two-coordinate sweeps found repetition gains but did not preserve answer "
            "completion and quality. This is the quality-recovery phase: change exactly three "
            "fields. Retain at least one demonstrated non-baseline value from "
            "interaction_anchor_configurations, and use the other fields to recover termination, "
            "non-empty final answers, Turkish/English quality, and instruction adherence."
        )
    system = (
        "You are the research boss in an autonomous decoding-optimization experiment. "
        "Your only variables are the four sampling parameters supplied below for Qwen3.5 2B. "
        "The system prompt, model, benchmark, seed, context size, and output cap are fixed. The target is "
        "English/Turkish general chat, and the primary defect is exact repetition in its hidden "
        "thinking trace. The score also lightly penalizes overly long thinking and gives the "
        "maximum efficiency penalty when no final answer is produced. Preserve correctness, "
        "helpfulness, language, safety, and instruction "
        "following. Low-entropy decoding can trap the model in deterministic loops, so prioritize "
        "controlled exploration through temperature and the probability filters. Higher temperature, "
        "higher top_p, and higher top_k generally expand the sampling pool; higher min_p filters more "
        "low-probability tokens and narrows it. "
        + phase_instruction
        + " The local Ollama "
        "backend was empirically shown to ignore presence, frequency, and repeat penalties in the "
        "thinking trace; never propose those fields. Do not repeat a configuration from recent "
        "experiments. Do not discuss or alter prompts and do not encode benchmark answers. Return one "
        "generalizable experiment as strict JSON with exactly the string keys hypothesis, "
        "expected_effect, plus decoding_config. decoding_config must contain exactly: "
        + ", ".join(DECODING_FIELDS)
        + ". Return the complete configuration, not a partial patch."
    )
    experiment = {
        "current_decoding_config": current_decoding.to_dict(),
        "allowed_ranges_inclusive": decoding_bounds,
        "visible_screen_metrics": screen_summary,
        "visible_loop_diagnostics": loop_examples,
        "recent_experiments": history,
        "all_previously_tried_configurations": tried_configurations,
        "required_number_of_changed_fields": required_changed_fields,
        "interaction_anchor_configurations": interaction_anchor_configurations,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(experiment, ensure_ascii=False, indent=2)},
    ]


class DeepSeekClient:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float,
        input_rate: float,
        output_rate: float,
        budget: CostBudget,
        usage_callback: Callable[[ApiUsage], None] | None = None,
        timeout: float = 120.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.input_rate = input_rate
        self.output_rate = output_rate
        self.budget = budget
        self.usage_callback = usage_callback
        self.http = httpx.Client(
            base_url=base_url.rstrip("/"),
            timeout=timeout,
            transport=transport,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )

    def close(self) -> None:
        self.http.close()

    def _chat(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float | None = None,
    ) -> tuple[str, ApiUsage]:
        self.budget.reserve_check(messages, max_tokens)
        try:
            response = self.http.post(
                "/chat/completions",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": self.temperature if temperature is None else temperature,
                    "max_tokens": max_tokens,
                    "response_format": {"type": "json_object"},
                    "thinking": {"type": "disabled"},
                    "stream": False,
                },
            )
            response.raise_for_status()
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            raw_usage = data.get("usage", {})
        except (httpx.HTTPError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            raise DeepSeekError(f"DeepSeek API request failed: {exc}") from exc
        prompt_tokens = int(raw_usage.get("prompt_tokens", 0))
        completion_tokens = int(raw_usage.get("completion_tokens", 0))
        cost = (
            prompt_tokens * self.input_rate / 1_000_000
            + completion_tokens * self.output_rate / 1_000_000
        )
        usage = ApiUsage(prompt_tokens, completion_tokens, cost)
        self.budget.charge(usage)
        if self.usage_callback is not None:
            self.usage_callback(usage)
        if not isinstance(content, str):
            raise DeepSeekError("DeepSeek returned non-text content")
        return content, usage

    def propose(
        self,
        *,
        current_decoding: DecodingConfig,
        decoding_bounds: dict[str, Any],
        screen_summary: dict[str, Any],
        loop_examples: list[dict[str, Any]],
        history: list[dict[str, Any]],
        tried_configurations: list[dict[str, float | int]],
        required_changed_fields: int,
        interaction_anchor_configurations: list[dict[str, float | int]],
        max_tokens: int,
        max_description_characters: int,
    ) -> tuple[Candidate, ApiUsage]:
        messages = build_boss_messages(
            current_decoding=current_decoding,
            decoding_bounds=decoding_bounds,
            screen_summary=screen_summary,
            loop_examples=loop_examples,
            history=history,
            tried_configurations=tried_configurations,
            required_changed_fields=required_changed_fields,
            interaction_anchor_configurations=interaction_anchor_configurations,
        )
        usage = ApiUsage()
        last_error: DeepSeekError | None = None
        max_attempts = 3
        for attempt in range(max_attempts):
            retry_messages = list(messages)
            if attempt:
                rejection = f" Rejection reason: {last_error}." if last_error else ""
                retry_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Your previous response was not valid strict JSON. Try again. Return only "
                            "one JSON object with exactly hypothesis, decoding_config, and "
                            "expected_effect. Include all four sampling fields and change exactly "
                            f"{required_changed_fields} value(s) from the current configuration. "
                            "Do not reuse a configuration "
                            "listed in all_previously_tried_configurations."
                            + rejection
                        ),
                    }
                )
            content, call_usage = self._chat(retry_messages, max_tokens=max_tokens)
            usage = usage + call_usage
            try:
                candidate = parse_candidate(
                    content,
                    bounds=decoding_bounds,
                    max_description_characters=max_description_characters,
                )
                if candidate.decoding_config == current_decoding:
                    raise DeepSeekError("Candidate is identical to the current configuration")
                candidate_key = json.dumps(
                    candidate.decoding_config.to_dict(), sort_keys=True, separators=(",", ":")
                )
                tried_keys = {
                    json.dumps(item, sort_keys=True, separators=(",", ":"))
                    for item in tried_configurations
                }
                if candidate_key in tried_keys:
                    raise DeepSeekError("Candidate configuration was already evaluated")
                changed_fields = [
                    field
                    for field in DECODING_FIELDS
                    if getattr(candidate.decoding_config, field)
                    != getattr(current_decoding, field)
                ]
                if len(changed_fields) != required_changed_fields:
                    raise DeepSeekError(
                        f"Candidate must change exactly {required_changed_fields} field(s); changed "
                        + ", ".join(changed_fields)
                    )
                if required_changed_fields > 1 and interaction_anchor_configurations:
                    retains_anchor = any(
                        any(
                            anchor[field] != getattr(current_decoding, field)
                            and getattr(candidate.decoding_config, field) == anchor[field]
                            for field in DECODING_FIELDS
                        )
                        for anchor in interaction_anchor_configurations
                    )
                    if not retains_anchor:
                        raise DeepSeekError(
                            "Interaction candidate must retain a demonstrated non-baseline value "
                            "from interaction_anchor_configurations"
                        )
                return candidate, usage
            except DeepSeekError as exc:
                last_error = exc
        raise ProposalRejected(
            f"Boss returned invalid structured output after {max_attempts} attempts: {last_error}"
        )

    def judge_pair(
        self,
        *,
        case: BenchmarkCase,
        baseline: GenerationResult,
        candidate: GenerationResult,
        max_tokens: int,
    ) -> JudgeDecision:
        stable_seed = int.from_bytes(
            hashlib.sha256(f"{case.id}:{candidate.seed}".encode()).digest()[:8], "big"
        )
        baseline_is_a = random.Random(stable_seed).choice([True, False])
        answer_a = baseline.content if baseline_is_a else candidate.content
        answer_b = candidate.content if baseline_is_a else baseline.content
        system = (
            "You are a strict blinded evaluator. Compare two answers to the same bilingual chat "
            "request. Score each from 1 to 5 for correctness, relevance, instruction adherence, "
            "language quality, helpfulness, safety, and concision. A critical failure is an empty, "
            "unsafe, nonsensical, wrong-language, or clearly instruction-violating answer. Return "
            "strict JSON with exactly: score_a (number), score_b (number), preference "
            "(A, B, or tie), critical_failure_a (boolean), critical_failure_b (boolean), reason "
            "(short string). Do not favor verbosity or references to hidden reasoning."
        )
        user = {
            "conversation": case.messages,
            "rubric": case.rubric,
            "answer_A": answer_a,
            "answer_B": answer_b,
        }
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(user, ensure_ascii=False)},
        ]
        usage = ApiUsage()
        last_error: Exception | None = None
        data: dict[str, Any] | None = None
        score_a = score_b = 0.0
        required = {
            "score_a",
            "score_b",
            "preference",
            "critical_failure_a",
            "critical_failure_b",
            "reason",
        }
        for attempt in range(3):
            retry_messages = list(messages)
            if attempt:
                retry_messages.append(
                    {
                        "role": "user",
                        "content": (
                            "The previous output was malformed. Return only valid JSON. Escape all "
                            "quotes inside reason and use exactly the required six keys."
                        ),
                    }
                )
            payload, call_usage = self._chat(
                retry_messages, max_tokens=max_tokens, temperature=0.0
            )
            usage = usage + call_usage
            try:
                parsed = json.loads(payload)
                if not isinstance(parsed, dict) or set(parsed) != required:
                    raise ValueError("wrong judge schema")
                score_a = float(parsed["score_a"])
                score_b = float(parsed["score_b"])
                if not 1 <= score_a <= 5 or not 1 <= score_b <= 5:
                    raise ValueError("scores must be between 1 and 5")
                if parsed["preference"] not in {"A", "B", "tie"}:
                    raise ValueError("invalid preference")
                if not isinstance(parsed["critical_failure_a"], bool) or not isinstance(
                    parsed["critical_failure_b"], bool
                ):
                    raise ValueError("critical failure fields must be booleans")
                if not isinstance(parsed["reason"], str):
                    raise ValueError("reason must be a string")
                data = parsed
                break
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                last_error = exc
        if data is None:
            raise DeepSeekError(
                f"Invalid judge response after 3 attempts: {last_error}"
            ) from last_error
        baseline_score = score_a if baseline_is_a else score_b
        candidate_score = score_b if baseline_is_a else score_a
        raw_preference = data["preference"]
        if raw_preference == "tie":
            preference = "tie"
        elif (raw_preference == "A") == baseline_is_a:
            preference = "baseline"
        else:
            preference = "candidate"
        candidate_critical = bool(
            data["critical_failure_b"] if baseline_is_a else data["critical_failure_a"]
        )
        return JudgeDecision(
            case_id=case.id,
            seed=candidate.seed,
            baseline_score=baseline_score,
            candidate_score=candidate_score,
            preference=preference,
            critical_failure=candidate_critical,
            reason=str(data["reason"]),
            usage=usage,
        )


def api_key_from_environment() -> str:
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        raise DeepSeekError("DEEPSEEK_API_KEY is not set")
    return key
