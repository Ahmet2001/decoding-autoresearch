# Qwen Decoding Autoresearch

Autonomous, closed-loop experimentation on decoding parameters to reduce chain-of-thought
repetition loops in **Qwen3.5:2b (q4_K_M)**, run locally through Ollama. Adapts
[Andrej Karpathy's "autoresearch"](https://x.com/karpathy) idea — an agent that proposes an
experiment, runs it, measures it, and keeps or discards the result — to a single, tightly
constrained artifact: `decoding_config.toml`.

This is a companion experiment to
[SLMOrchestration](https://github.com/Ahmet2001/SLMOrchestration/tree/cp1), where the same
repetition problem showed up in very small models during CoT generation.

> **TL;DR result:** across two different optimizer strategies and 90+ evaluated configurations,
> no candidate decoding configuration reliably beat the baseline on held-out data. See
> [Result](#result) below — this is a negative-result writeup, published as-is for auditability.

## Table of contents

- [Motivation](#motivation)
- [Result](#result)
- [How it works](#how-it-works)
- [Two optimizer strategies](#two-optimizer-strategies)
- [Scoring](#scoring)
- [Keep / discard policy](#keep--discard-policy)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration reference](#configuration-reference)
- [Project structure](#project-structure)
- [Limitations & conclusion](#limitations--conclusion)
- [References](#references)

## Motivation

Small reasoning models frequently fall into **exact repetition loops** inside their hidden
`message.thinking` trace — restating the same span of tokens instead of converging on an answer.
This wastes the output budget and can crowd out the final answer entirely.

The hypothesis tested here: could **decoding-time sampling parameters alone**
(`temperature`, `top_p`, `top_k`, `min_p`) meaningfully reduce this, without touching the prompt,
the model weights, or the benchmark? Everything except those four fields is held fixed for the
whole research series — model, system prompt, benchmark, seed, context size, output cap, metrics,
and acceptance gates — so any accepted change is attributable to decoding alone.

## Result

**No configuration was accepted.** The system is designed to be conservative by construction — a
candidate only replaces `decoding_config.toml` if it passes every quality gate *and* is
statistically better than the current configuration — and across every experiment run, nothing
cleared that bar.

| Approach | Candidates evaluated | Accepted | Notes |
|---|---|---|---|
| LLM-proposed (DeepSeek boss) | 18 | 0 | 1 crash from malformed/duplicate structured output |
| Algorithmic (Optuna multivariate TPE) | 68 trials (74 in the persisted study) | 0 | best *screen-only* score improved ~16% but never passed the held-out validation + bootstrap gate together |

Baseline decoding (`temperature=1.0, top_p=0.95, top_k=20, min_p=0.0`) on the held-out validation
split:

- `loop_incidence = 1.00` — every sampled response looped at least once
- `truncation_rate = 0.50`
- `empty_answer_rate = 0.50`
- `mean_reasoning_efficiency_penalty = 0.86`

The best *feasible* screen-only score found by TPE was `0.396` vs. a baseline of `0.434` (~16%
better on the fast screening subset), but none of those candidates simultaneously held up on the
held-out validation split, the deterministic pass-rate/empty-answer/truncation constraints, and the
paired-bootstrap significance test — so nothing was promoted.

**Working conclusion:** the repetition behavior observed here looks more like a pretraining-time
property of this model size than something four sampling knobs can reliably correct.
Decoding-parameter search can trade one failure mode for another (e.g. less exact repetition at the
cost of more truncation or empty answers) but didn't find a configuration that improved everything
at once, within the tested ranges and budget.

## How it works

```
              ┌──────────────────────┐
              │  system_prompt.md    │  (fixed)
              │  benchmark.jsonl     │  (fixed)
              │  decoding_config.toml│  (the only mutable artifact)
              └──────────┬───────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │   optimizer proposes a       │
          │   candidate decoding config  │   (TPE, or historically DeepSeek)
          └──────────────┬───────────────┘
                         │
                         ▼
          ┌─────────────────────────────┐
          │  Ollama generates responses  │  qwen3.5:2b-q4_K_M, message.thinking
          │  on the fast "screen" split  │  captured separately from the answer
          └──────────────┬───────────────┘
               screen improves enough? ──No──▶ discard, log to results.tsv
                         │ Yes
                         ▼
          ┌─────────────────────────────┐
          │  full held-out "validation"  │
          │  split + paired bootstrap    │
          └──────────────┬───────────────┘
             passes quality gates + ──No──▶ discard, log to results.tsv
             bootstrap upper bound < 0?
                         │ Yes
                         ▼
          ┌─────────────────────────────┐
          │ decoding_config.toml updated │
          │ + git commit                 │
          └───────────────────────────────┘
```

Every trial — kept, discarded, or crashed — is logged to `results.tsv` and archived under `runs/`,
so the full search history is auditable even though the working config rarely changes.

## Two optimizer strategies

### 1. LLM-proposed candidates (DeepSeek, retired)

The first version used DeepSeek (`deepseek-v4-flash`) as a "research boss": it read the current
config, screen metrics, and history, and proposed one new candidate as structured JSON, one changed
field at a time and later two or three fields once single-coordinate search stalled
(`prompt_research/deepseek.py`). It was also used as a blind pairwise judge for answer quality.

This was abandoned for the active research path:

- **Reliability** — structured JSON output occasionally failed validation or repeated an
  already-tried configuration; one run crashed after exhausting all retries
  (`ProposalRejected`).
- **Cost** — every candidate cost real API tokens ($0.002–$0.02 per proposal), which caps how much
  of the search space can realistically be explored.
- **No real optimization signal** — proposals were plausible-sounding heuristics
  ("increasing temperature should reduce repetition because...") rather than a model that learns
  correlations from prior trials.
- **Judge/proposer conflict of interest** — the same LLM proposing candidates was also scoring
  candidate vs. baseline answer quality.

The DeepSeek path is still in the codebase (`prompt_research/deepseek.py`,
`ResearchRunner.run_with_boss`) for archival/comparison purposes, and its historical runs live
under `runs/legacy-deepseek-boss-final-20260714/`, but it is **not required** and not used by
`research.py run`.

### 2. Algorithmic search (Optuna multivariate TPE, active)

The active path replaces the LLM proposer entirely with
[Optuna](https://optuna.org/)'s constrained multivariate TPE sampler:

- No API calls, no cost, fully reproducible from a fixed sampler seed.
- Learns correlations across `temperature` / `top_p` / `top_k` / `min_p` from every prior trial
  persisted in a SQLite study (`.autoresearch/optuna.db`).
- The search space is quantized (fixed step sizes) so trials can't differ by imperceptible
  amounts, and exact-duplicate configurations are served from the study without spending another
  Ollama evaluation.
- Screen-fail, empty-answer, and truncation regressions are enforced as hard Optuna *constraints*,
  not just objective terms — a candidate can't win by trading correctness for a lower score.

## Scoring

The lower-is-better repetition/repair score is:

```
0.60 × exact-loop coverage (repeated 8-gram span)
+ 0.20 × loop incidence (fraction of samples that looped at all)
+ 0.10 × truncation rate
+ 0.10 × reasoning inefficiency
```

*Reasoning inefficiency* estimates how much of Ollama's real generated-token count fell inside
`message.thinking`, normalized by the fixed output cap; a response with no final answer gets the
maximum penalty. This prevents an optimizer from "winning" by cutting `message.thinking` short
without actually producing a usable answer.

Correctness (deterministic exact-match checks) and answer quality are enforced as **separate
gates**, described next — a short wrong answer can't win purely by gaming the repetition score.

## Keep / discard policy

1. A candidate is generated on the fast **screen** split (768-token cap, one seed, loop-stress +
   exact-completion cases).
2. It's only promoted to the **validation** split (2048-token cap, four fully held-out cases) if it
   improves the screen score by ≥5% *and* doesn't regress deterministic pass rate.
3. On validation, the candidate must pass quality constraints (empty-answer rate, truncation rate,
   deterministic pass rate not regressing) **and** the improvement must clear a paired-bootstrap
   significance test: the upper bound of the bootstrap confidence interval on the score difference
   must be below zero.
4. Only then does it overwrite `decoding_config.toml`, with an automatic git commit recording the
   change.

Only one `research.py run` process may touch the local model and Optuna study at a time; a
non-blocking `flock` on `.autoresearch/run.lock` rejects accidental concurrent starts before any
GPU work begins.

## Installation

Requires Python ≥3.11, [uv](https://docs.astral.sh/uv/), and a local
[Ollama](https://ollama.com/) instance with `qwen3.5:2b-q4_K_M` pulled.

```bash
git clone https://github.com/Ahmet2001/qwen-decoding-autoresearch.git
cd qwen-decoding-autoresearch
uv sync --dev
ollama pull qwen3.5:2b-q4_K_M
```

`DEEPSEEK_API_KEY` in `.env` is **not required** for the active commands below — it's only needed
to re-read the archived legacy DeepSeek-boss experiments.

## Usage

```bash
uv run python research.py calibrate                       # compatibility preflight
uv run python research.py baseline                         # record the reference score
uv run python research.py run --max-hours 4 --max-trials 40  # TPE search loop
uv run python research.py validate                          # re-check the accepted config
uv run pytest                                                # test suite
```

- `calibrate` verifies Ollama connectivity, GPU offload, the separate `message.thinking` field, and
  the full current decoding config. It does not call any external API or pick a "winning" profile.
- `baseline` runs the fixed config on both splits and records the reference scores that every future
  candidate is measured against.
- `run` executes the TPE loop, respecting `--max-hours` / `--max-trials`, and stops early on a
  manual interrupt (`Ctrl+C`), safely persisting study state either way.
- `validate` re-runs the currently accepted config on fresh samples to confirm it still holds up.

## Configuration reference

All fixed research parameters — model, benchmark case selection, metric weights, quality
thresholds, and the TPE search space/step sizes — live in `config.toml`. The search space
(inclusive bounds) is grounded in the entropy-collapse and controlled-exploration findings from:

- *Wait, Wait, Wait… Why Do Reasoning Models Loop?*
- *Circular Reasoning*
- *Neural Text Generation with Unlikelihood Training*
- *Repetition In, Repetition Out*

Direct local probes showed Ollama 0.24 / Qwen3.5 applies these four sampling controls to the
thinking trace but ignores presence-, frequency-, and repeat-penalty options there, so those
no-op parameters are intentionally excluded from the search space.

## Project structure

```
research.py               CLI entrypoint (calibrate / baseline / run / validate)
config.toml                Fixed research config, metric weights, TPE search space
decoding_config.toml       The only file the optimizer is allowed to overwrite
system_prompt.md           Fixed system prompt for the whole research series
benchmark.jsonl            Bilingual (EN/TR) screen + held-out validation cases
prompt_research/
  runner.py                 ResearchRunner: calibrate/baseline/run/validate orchestration
  evaluation.py              Runs a benchmark split through Ollama, aggregates scores
  metrics.py                 Loop detection, scoring, paired bootstrap
  ollama.py                  Ollama /api/chat client + GPU-offload preflight
  deepseek.py                Legacy LLM-proposer/judge (not on the active path)
  config.py, models.py, io_utils.py, git_ops.py, benchmark.py
tests/                     pytest suite for the modules above
runs/                      Per-run artifacts: proposals, responses, reports, summaries
results.tsv                 Flat log of every trial ever evaluated
.autoresearch/              Optuna SQLite study, run state, run lock (git-ignored)
```

## Limitations & conclusion

- The search only covers four sampling parameters within fixed, conservative bounds — it says
  nothing about whether a wider search space, a different quantization, or more trials would
  eventually find a working point.
- The benchmark is deliberately small and fast (screen: 4 cases × 1 seed; validation: 4 held-out
  cases × 1 seed) to keep iteration cheap; a larger, multi-seed benchmark would give tighter
  confidence intervals but cost proportionally more compute.
- Both optimizer strategies converged on the same outcome — no accepted candidate — which is weak
  but non-trivial evidence that, for this model at this size, the repetition problem is not
  primarily a decoding-parameter problem.

## References

- Andrej Karpathy's autoresearch loop concept
- [SLMOrchestration](https://github.com/Ahmet2001/SLMOrchestration/tree/cp1) — the companion
  project where this repetition issue was first observed
- *Wait, Wait, Wait… Why Do Reasoning Models Loop?*
- *Circular Reasoning*
- *Neural Text Generation with Unlikelihood Training*
- *Repetition In, Repetition Out*
