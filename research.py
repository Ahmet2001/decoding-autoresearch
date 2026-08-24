#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prompt_research.config import load_config, load_dotenv
from prompt_research.runner import ResearchRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Autonomous decoding-parameter research for Qwen3.5 2B"
    )
    parser.add_argument("--config", type=Path, default=Path("config.toml"))
    parser.add_argument("--benchmark", type=Path, default=Path("benchmark.jsonl"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    for name in ("preflight", "calibrate", "baseline", "validate"):
        command = subparsers.add_parser(name)
        command.add_argument(
            "--allow-cpu",
            action="store_true",
            help="Allow CPU inference for diagnostics (not recommended for research runs)",
        )
        if name == "calibrate":
            command.add_argument("--max-cost-usd", type=float)

    run = subparsers.add_parser("run")
    run.add_argument("--max-hours", type=float, default=4.0)
    run.add_argument("--max-trials", type=int, default=40)
    run.add_argument("--allow-cpu", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config_path = args.config.resolve()
    load_dotenv(config_path.parent / ".env")
    config = load_config(config_path)
    runner = ResearchRunner(config, benchmark_path=args.benchmark.resolve())
    try:
        require_gpu = not args.allow_cpu
        if args.command == "preflight":
            result = runner.preflight(require_gpu=require_gpu)
        elif args.command == "calibrate":
            result = runner.calibrate(
                require_gpu=require_gpu, max_cost_usd=args.max_cost_usd
            )
        elif args.command == "baseline":
            result = runner.baseline(require_gpu=require_gpu)
        elif args.command == "run":
            if args.max_hours <= 0 or args.max_trials <= 0:
                raise ValueError("--max-hours and --max-trials must be positive")
            result = runner.run(
                max_hours=args.max_hours,
                max_trials=args.max_trials,
                require_gpu=require_gpu,
            )
        elif args.command == "validate":
            result = runner.validate(require_gpu=require_gpu)
        else:  # pragma: no cover
            raise AssertionError(args.command)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    finally:
        runner.close()


if __name__ == "__main__":
    raise SystemExit(main())
