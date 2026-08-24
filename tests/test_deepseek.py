import json

import httpx
import pytest

from pathlib import Path

from prompt_research.config import load_config
from prompt_research.deepseek import (
    BudgetExceeded,
    CostBudget,
    DeepSeekClient,
    DeepSeekError,
    build_boss_messages,
    parse_candidate,
)
from prompt_research.models import BenchmarkCase, DecodingConfig, GenerationResult


ROOT = Path(__file__).parents[1]
BOUNDS = load_config(ROOT / "config.toml").decoding_search
DECODING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 20,
    "min_p": 0.0,
}


def test_candidate_schema_and_length() -> None:
    payload = json.dumps(
        {"hypothesis": "h", "decoding_config": DECODING, "expected_effect": "e"}
    )
    candidate = parse_candidate(
        payload, bounds=BOUNDS, max_description_characters=100
    )
    assert candidate.decoding_config.top_k == 20
    with pytest.raises(DeepSeekError):
        parse_candidate("not json", bounds=BOUNDS, max_description_characters=100)
    with pytest.raises(DeepSeekError):
        parse_candidate(
            json.dumps({"hypothesis": "h", "decoding_config": DECODING}),
            bounds=BOUNDS,
            max_description_characters=100,
        )
    with pytest.raises(DeepSeekError):
        parse_candidate(payload, bounds=BOUNDS, max_description_characters=0)
    invalid = dict(DECODING, temperature=3.0)
    with pytest.raises(DeepSeekError, match="outside"):
        parse_candidate(
            json.dumps({"hypothesis": "h", "decoding_config": invalid, "expected_effect": "e"}),
            bounds=BOUNDS,
            max_description_characters=100,
        )


def test_boss_context_has_no_implicit_hidden_data() -> None:
    messages = build_boss_messages(
        current_decoding=DecodingConfig(**DECODING),
        decoding_bounds=BOUNDS,
        screen_summary={"split": "screen", "score": 0.5},
        loop_examples=[{"case_id": "s-en-01", "repeated_span": "x"}],
        history=[],
        tried_configurations=[DECODING],
        required_changed_fields=1,
        interaction_anchor_configurations=[],
    )
    rendered = json.dumps(messages)
    assert "v-en-10" not in rendered
    assert "validation" not in json.loads(messages[1]["content"])


def test_budget_refuses_call_before_crossing_cap() -> None:
    budget = CostBudget(0.001, 0.0, input_rate=10.0, output_rate=10.0)
    with pytest.raises(BudgetExceeded):
        budget.reserve_check([{"role": "user", "content": "x" * 1000}], 1000)


def test_usage_cost_accounting() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": '{"ok":true}'}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            },
        )

    budget = CostBudget(10, 0, input_rate=2, output_rate=8)
    client = DeepSeekClient(
        base_url="https://example.test",
        api_key="test",
        model="test",
        temperature=0,
        input_rate=2,
        output_rate=8,
        budget=budget,
        transport=httpx.MockTransport(handler),
    )
    _, usage = client._chat([{"role": "user", "content": "x"}], max_tokens=10)
    client.close()
    assert usage.estimated_cost_usd == pytest.approx(0.006)
    assert budget.used_usd == pytest.approx(0.006)


def test_boss_retries_until_exactly_one_sampling_field_changes() -> None:
    responses = iter(
        [
            {**DECODING, "temperature": 1.2, "top_k": 10},
            {**DECODING, "temperature": 1.2},
            {**DECODING, "temperature": 1.1},
        ]
    )
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        proposal = {
            "hypothesis": "test one coordinate",
            "decoding_config": next(responses),
            "expected_effect": "test",
        }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(proposal)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    budget = CostBudget(10, 0, input_rate=2, output_rate=8)
    client = DeepSeekClient(
        base_url="https://example.test",
        api_key="test",
        model="test",
        temperature=0.7,
        input_rate=2,
        output_rate=8,
        budget=budget,
        transport=httpx.MockTransport(handler),
    )
    candidate, _ = client.propose(
        current_decoding=DecodingConfig(**DECODING),
        decoding_bounds=BOUNDS,
        screen_summary={"repetition_score": 0.2},
        loop_examples=[],
        history=[],
        tried_configurations=[{**DECODING, "temperature": 1.2}],
        required_changed_fields=1,
        interaction_anchor_configurations=[],
        max_tokens=200,
        max_description_characters=100,
    )
    client.close()
    assert len(calls) == 3
    assert candidate.decoding_config.temperature == 1.1
    assert candidate.decoding_config.top_k == 20


def test_interaction_candidate_must_retain_an_anchor_value() -> None:
    responses = iter(
        [
            {**DECODING, "temperature": 1.15, "min_p": 0.05},
            {**DECODING, "temperature": 1.3, "min_p": 0.05},
        ]
    )
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        proposal = {
            "hypothesis": "interaction",
            "decoding_config": next(responses),
            "expected_effect": "retain repetition gain and recover quality",
        }
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(proposal)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    client = DeepSeekClient(
        base_url="https://example.test",
        api_key="test",
        model="test",
        temperature=0.7,
        input_rate=2,
        output_rate=8,
        budget=CostBudget(10, 0, input_rate=2, output_rate=8),
        transport=httpx.MockTransport(handler),
    )
    candidate, _ = client.propose(
        current_decoding=DecodingConfig(**DECODING),
        decoding_bounds=BOUNDS,
        screen_summary={"repetition_score": 0.3},
        loop_examples=[],
        history=[],
        tried_configurations=[],
        required_changed_fields=2,
        interaction_anchor_configurations=[{**DECODING, "temperature": 1.3}],
        max_tokens=200,
        max_description_characters=100,
    )
    client.close()
    assert len(calls) == 2
    assert candidate.decoding_config.temperature == 1.3
    assert candidate.decoding_config.min_p == 0.05


def test_judge_retries_malformed_json_deterministically() -> None:
    responses = iter(
        [
            '{"score_a": 4, "reason": "broken "quote"}',
            json.dumps(
                {
                    "score_a": 4,
                    "score_b": 4,
                    "preference": "tie",
                    "critical_failure_a": False,
                    "critical_failure_b": False,
                    "reason": "Equivalent answers.",
                }
            ),
        ]
    )
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": next(responses)}}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10},
            },
        )

    budget = CostBudget(10, 0, input_rate=2, output_rate=8)
    client = DeepSeekClient(
        base_url="https://example.test",
        api_key="test",
        model="test",
        temperature=0.7,
        input_rate=2,
        output_rate=8,
        budget=budget,
        transport=httpx.MockTransport(handler),
    )
    case = BenchmarkCase(
        "open",
        "validation",
        "en",
        "test",
        [{"role": "user", "content": "Answer."}],
        {"type": "judge"},
        "Be correct.",
    )
    row = GenerationResult(
        "open", "validation", "en", "test", 1, "", "answer", "stop", 1, 1, 0.1
    )
    decision = client.judge_pair(case=case, baseline=row, candidate=row, max_tokens=100)
    client.close()
    assert decision.preference == "tie"
    assert len(calls) == 2
    assert all(request.read() and json.loads(request.content)["temperature"] == 0.0 for request in calls)
    assert all(
        json.loads(request.content)["thinking"] == {"type": "disabled"} for request in calls
    )
