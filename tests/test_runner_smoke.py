from __future__ import annotations

import shutil
from pathlib import Path

from prompt_research.config import load_config
from prompt_research.io_utils import save_state
from prompt_research.models import (
    ApiUsage,
    Candidate,
    DecodingConfig,
    GenerationResult,
    JudgeDecision,
)
from prompt_research.runner import ResearchRunner


ROOT = Path(__file__).parents[1]


class FakeOllama:
    def __init__(self):
        self.output_limits = []

    def preflight(self, **kwargs):
        return {"ollama_version": "test", "model": "fake", "gpu_fraction": 1.0, "thinking_field": True}

    def generate(self, *, case, system_prompt, decoding, seed, output_tokens=None):
        self.output_limits.append((case.split, output_tokens))
        answers = {
            "s-en-01": "Paris is the capital of France.", "s-en-02": "9 remain.", "s-en-03": "BLUE",
            "s-en-05": "Thursday.", "s-tr-01": "Türkiye'nin başkenti Ankara'dır.", "s-tr-02": "7 kalem kalır.",
            "s-tr-03": "MAVİ", "s-tr-05": "Ayran.", "v-en-01": "Tokyo", "v-en-02": "56",
            "v-en-03": "red green blue", "v-en-07": "Mira", "v-en-08": "Good morning, how are you?",
            "v-tr-01": "Tokyo", "v-tr-02": "54", "v-tr-03": "kırmızı yeşil mavi", "v-tr-07": "Mira",
            "v-tr-08": "İyi akşamlar, nasılsın?",
        }
        content = answers.get(case.id, "This is a sufficiently useful answer for the requested task.")
        if decoding.temperature > 1.0:
            thinking = "one two three four five six seven eight nine ten eleven twelve"
        else:
            thinking = "one two three four five six seven eight one two three four five six seven eight"
        return GenerationResult(
            case.id, case.split, case.language, case.category, seed, thinking, content,
            "stop", 12, 4, 0.01,
        )

    def close(self):
        pass


class FakeDeepSeek:
    def __init__(self, budget):
        self.budget = budget
        self.proposals = 0

    def propose(self, **kwargs):
        self.proposals += 1
        usage = ApiUsage(10, 10, 0.0001)
        self.budget.charge(usage)
        if self.proposals == 1:
            return Candidate(
                "No effective change", DecodingConfig(0.9, 0.95, 20, 0.0), "none"
            ), usage
        return Candidate(
            "Increase exploration", DecodingConfig(1.2, 0.95, 20, 0.0), "fewer loops"
        ), usage

    def judge_pair(self, *, case, baseline, candidate, max_tokens):
        usage = ApiUsage(5, 5, 0.00005)
        self.budget.charge(usage)
        return JudgeDecision(case.id, candidate.seed, 4, 4, "tie", False, "equal", usage)

    def close(self):
        pass


def test_two_candidate_autonomous_run_and_resume_state(tmp_path: Path) -> None:
    config_text = (ROOT / "config.toml").read_text(encoding="utf-8").replace(
        "max_boss_calls = 40", "max_boss_calls = 2"
    )
    (tmp_path / "config.toml").write_text(config_text, encoding="utf-8")
    shutil.copy(ROOT / "system_prompt.md", tmp_path / "system_prompt.md")
    shutil.copy(ROOT / "decoding_config.toml", tmp_path / "decoding_config.toml")
    config = load_config(tmp_path / "config.toml")
    fake_boss = None

    def factory(budget):
        nonlocal fake_boss
        fake_boss = FakeDeepSeek(budget)
        return fake_boss

    runner = ResearchRunner(
        config,
        benchmark_path=ROOT / "benchmark.jsonl",
        ollama=FakeOllama(),
        deepseek_factory=factory,
    )
    runner.calibrate(require_gpu=False)
    runner.baseline(require_gpu=False)
    # A new run receives its own additional budget even when earlier runs have
    # already accumulated spend.
    runner.state.total_estimated_cost_usd = 0.25
    save_state(runner.state_path, runner.state)
    summary = runner.run_with_boss(
        max_hours=1, max_cost_usd=1, require_gpu=False
    )
    runner.close()

    assert summary["accepted"] == 1
    assert summary["boss_calls"] == 2
    assert fake_boss.budget.maximum_usd == 1.25
    assert summary["estimated_cost_usd"] == 0.0
    assert summary["cumulative_cost_usd"] == 0.25
    assert runner.state.run_id is None
    assert "temperature = 1.2" in (tmp_path / "decoding_config.toml").read_text()
    assert (tmp_path / "system_prompt.md").read_text() == (ROOT / "system_prompt.md").read_text()
    results = (tmp_path / "results.tsv").read_text()
    assert "candidate-0001" in results and "discard" in results
    assert "candidate-0002" in results and "keep" in results
    assert runner.state_path.exists()
    assert runner.state.accepted_validation_path
    assert ("screen", 768) in runner.ollama.output_limits
    assert ("validation", 2048) in runner.ollama.output_limits

    resumed = ResearchRunner(
        config,
        benchmark_path=ROOT / "benchmark.jsonl",
        ollama=FakeOllama(),
        deepseek_factory=factory,
    )
    assert resumed.state.accepted_count == 1
    assert resumed.state.current_decoding_sha256 == summary["current_decoding_sha256"]
    resumed.close()


def test_algorithmic_run_uses_tpe_without_deepseek(tmp_path: Path) -> None:
    shutil.copy(ROOT / "config.toml", tmp_path / "config.toml")
    shutil.copy(ROOT / "system_prompt.md", tmp_path / "system_prompt.md")
    shutil.copy(ROOT / "decoding_config.toml", tmp_path / "decoding_config.toml")
    config = load_config(tmp_path / "config.toml")

    def forbidden_factory(_budget):
        raise AssertionError("algorithmic run must not construct a DeepSeek client")

    runner = ResearchRunner(
        config,
        benchmark_path=ROOT / "benchmark.jsonl",
        ollama=FakeOllama(),
        deepseek_factory=forbidden_factory,
    )
    runner.calibrate(require_gpu=False)
    runner.baseline(require_gpu=False)
    summary = runner.run(max_hours=1, max_trials=1, require_gpu=False)
    resumed_summary = runner.run(max_hours=1, max_trials=1, require_gpu=False)
    runner.close()

    assert summary["optimizer"] == "optuna-multivariate-tpe"
    assert summary["api_cost_usd"] == 0.0
    assert summary["new_trials"] == 1
    assert resumed_summary["new_trials"] == 1
    results = (tmp_path / "results.tsv").read_text(encoding="utf-8")
    assert "trial-" in results
    assert "\tduplicate\t" not in results
