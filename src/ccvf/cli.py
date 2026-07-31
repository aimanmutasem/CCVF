from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

import yaml

from ccvf.schemas import ExperimentConfig


def load_config(path: str | Path) -> ExperimentConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return ExperimentConfig.model_validate(raw)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ccvf")
    sub = parser.add_subparsers(dest="cmd")

    run = sub.add_parser("run", help="Run a CCVF/CAVER experiment")
    run.add_argument("--config", required=True, type=Path)
    run.add_argument("--run-id")
    run.add_argument("--dataset")
    run.add_argument("--limit", type=int)
    run.add_argument("--overwrite", action="store_true")
    run.add_argument("--no-resume", action="store_true")

    args = parser.parse_args(argv)
    if args.cmd is None:
        argv = ["run", *(argv or [])]
        return main(argv)
    if args.cmd == "run":
        from ccvf.runner import ExperimentRunner, InfrastructureAbort

        cfg = load_config(args.config)
        if args.run_id:
            cfg.run.run_id = args.run_id
        if args.dataset:
            cfg.dataset.path = args.dataset
        if args.limit is not None:
            cfg.dataset.limit = args.limit
        if args.overwrite:
            cfg.run.overwrite = True
        if args.no_resume:
            cfg.run.resume = False
        try:
            run_dir = asyncio.run(ExperimentRunner(cfg, args.config.resolve()).run())
        except InfrastructureAbort as exc:
            print(f"CCVF run aborted: {exc}", flush=True)
            return 3
        print(f"CCVF run complete: {run_dir}", flush=True)
        return 0
    parser.error(f"unknown command: {args.cmd}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
