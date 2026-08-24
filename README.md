# Qwen Decoding Autoresearch

This project adapts Karpathy's autoresearch loop to one controlled artifact:
`decoding_config.toml`. A constrained multivariate TPE optimizer proposes decoding configurations,
the local `qwen3.5:2b-q4_K_M` model runs through Ollama, and a fixed bilingual benchmark measures
exact loops and reasoning efficiency in `message.thinking`. No LLM proposes candidates or judges
answers on the active research path.

`system_prompt.md`, the model, benchmark, seed, context size, output limit, metrics, and acceptance
gates are fixed during a research series. The optimizer can change only:

- `temperature`, `top_p`, `top_k`, and `min_p`

The inclusive safety ranges live in fixed `config.toml`. Candidates with missing fields, extra
fields, non-finite values, or out-of-range values are rejected before local inference. The search
space is based on the entropy-collapse and controlled-exploration findings discussed in *Wait, Wait,
Wait… Why Do Reasoning Models Loop?*, *Circular Reasoning*, *Neural Text Generation with
Unlikelihood Training*, and *Repetition In, Repetition Out*. Direct local probes showed that Ollama
0.24/Qwen3.5 applies these four sampling controls to the thinking trace but ignores its presence,
frequency, and repeat-penalty options there, so no-op parameters are intentionally excluded.

Responses, Optuna's SQLite study, state, and result logs are ignored by Git. `DEEPSEEK_API_KEY` is
not required for `calibrate`, `baseline`, `run`, or `validate`; the DeepSeek configuration remains
only for reading archived legacy boss experiments.

## Commands

```bash
uv sync --dev
uv run python research.py calibrate
uv run python research.py baseline
uv run python research.py run --max-hours 4 --max-trials 40
uv run python research.py validate
uv run pytest
```

`calibrate` is now a fast compatibility preflight: it checks Ollama, GPU offload, the separate
thinking field, and the complete current decoding configuration. It does not call DeepSeek or pick a
winning profile. `baseline` then records the official Qwen-style starting configuration from
`decoding_config.toml`.

The default fast benchmark uses a 768-token screening cap, one fixed seed, two bilingual loop-stress
cases plus two exact-instruction completion checks, and four held-out bilingual validation cases with
a 2048-token cap. The longer cap is paid only by candidates that improve screening by at least 5%
without violating completion constraints. `run` stores its TPE study in `.autoresearch/optuna.db`
and stops on the trial count, time limit, or manual interruption.

The lower-is-better research score is now `0.60 × exact-loop coverage + 0.20 × loop
incidence + 0.10 × truncation + 0.10 × reasoning inefficiency`. Reasoning inefficiency
estimates how much of Ollama's generated-token count belongs to `message.thinking`, normalized by
the fixed split cap. A missing final answer receives the maximum efficiency penalty. Correctness
and answer quality are still enforced separately, so a short wrong answer cannot win by gaming
length alone.

Algorithmic runs make no paid API calls, so `estimated_cost_usd` is always zero. Historical DeepSeek
spend remains in `cumulative_cost_usd` for auditability but never affects the optimizer.

## Keep/discard policy

Optuna begins with seeded space-filling trials, then multivariate TPE models correlations among
temperature, top-p, top-k, and min-p. The search space is quantized to prevent meaningless microscopic
changes and exact duplicates are served from the persisted study without another Ollama evaluation.
The objective is visible-screen repetition score; constraints require deterministic pass rate, empty
answer rate, and truncation rate not to regress from baseline. Promising candidates are evaluated on
held-out cases. A candidate is kept only when held-out quality constraints pass, repetition improves
by at least the configured absolute margin, and the paired-bootstrap upper bound is below zero.
Accepted configurations replace `decoding_config.toml`; all trials remain under `runs/` and in
`results.tsv` for auditability.

Only one `research.py run` process may use the local model and Optuna study at a time. A non-blocking
filesystem lock rejects accidental concurrent starts before inference, preventing GPU contention and
state races; the operating system releases the lock automatically if the process exits.

Older prompt, no-op penalty, DeepSeek-boss, and diagnostic smoke states are preserved under the
`runs/legacy-*` directories and are not imported into the active TPE study.
