"""Run every gate check in a benchmark against Jev and record the raw answers.

    export TYPESAFE_API_KEY=...
    python -m eval.run                              # both datasets, jev-latest
    python -m eval.run --dataset rjudge --limit 20  # quick smoke test
    python -m eval.run --model jev-preview
    python -m eval.run --dataset rjudge --no-thought   # ablation: hide agent reasoning
    python -m eval.run --dataset rjudge --tag repeat   # second run, to measure stability

All six built-in policies are asked on every check (they share one request), and
their raw answers are stored, so every configuration in the report - the default
gate, block-only, single policies, other thresholds - is computed afterwards
without calling the API again.

Results go to eval/results/<model>/<dataset>.jsonl, one line per check, holding
ids and numbers only - no dataset text. Re-running resumes: checks that already
succeeded are skipped and failed ones are retried. Committed results are gzipped
(`gzip eval/results/*/*.jsonl`); both forms are read.
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

from actionreflex import AsyncGate, GateUnavailable
from actionreflex.policies import (
    contains_pii,
    destructive_action,
    follows_injected_instructions,
    intent_mismatch,
    risk_tier,
    scope_of_impact,
)

from .datasets import LOADERS, SOURCES, Case, Check

RESULTS = Path(__file__).parent / "results"


def all_policies():
    return [
        destructive_action(),
        intent_mismatch(),
        follows_injected_instructions(),
        contains_pii(),
        scope_of_impact(),
        risk_tier(),
    ]


def raw_answer(answer: Any) -> dict[str, Any]:
    if hasattr(answer, "noul"):
        return {"noul": answer.noul}
    if hasattr(answer, "choice"):
        return {
            "choice": answer.choice,
            "probabilities": dict(answer.probabilities),
            "confidence": answer.confidence,
        }
    return {
        "score": answer.score,
        "probabilities": {str(k): v for k, v in answer.probabilities.items()},
        "confidence": answer.confidence,
    }


def iter_rows(path: Path):
    """Rows from `<name>.jsonl.gz` (committed results) and `<name>.jsonl` (a local run)."""
    gz = path.with_suffix(".jsonl.gz")
    if gz.exists():
        with gzip.open(gz, "rt") as f:
            yield from (json.loads(line) for line in f)
    if path.exists():
        with path.open() as f:
            yield from (json.loads(line) for line in f)


def load_done(path: Path) -> set[str]:
    return {row["key"] for row in iter_rows(path) if "error" not in row}


def run_dir(model: str, no_thought: bool, tag: str | None = None) -> Path:
    name = model + ("+no-thought" if no_thought else "") + (f"+{tag}" if tag else "")
    return RESULTS / name


async def run_dataset(
    name: str,
    model: str,
    concurrency: int,
    limit: int | None,
    no_thought: bool,
    tag: str | None = None,
) -> None:
    cases: list[Case] = LOADERS[name]()
    if limit:
        cases = cases[:limit]
    if no_thought:
        # Many agent frameworks never expose the model's reasoning before a tool call,
        # so measure the gate without it.
        for case in cases:
            for check in case.checks:
                if isinstance(check.action.context, dict):
                    check.action.context.pop("agent_thought", None)
                    for step in check.action.context.get("history", []):
                        step.pop("thought", None)
    out = run_dir(model, no_thought, tag) / f"{name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    done = load_done(out)
    todo: list[tuple[Case, Check]] = [
        (case, check) for case in cases for check in case.checks if check.key not in done
    ]
    total = sum(len(c.checks) for c in cases)
    print(f"[{name}] {len(cases)} cases, {total} checks, {len(todo)} to run ({model})")
    if not todo:
        return

    semaphore = asyncio.Semaphore(concurrency)
    completed = errors = 0
    started = time.perf_counter()

    async with AsyncGate(all_policies(), model=model) as gate:
        with out.open("a") as sink:

            async def one(case: Case, check: Check) -> None:
                nonlocal completed, errors
                async with semaphore:
                    try:
                        verdict = await gate.check(check.action)
                    except GateUnavailable as exc:
                        row = {"key": check.key, "case_id": case.id, "error": repr(exc.cause)}
                        errors += 1
                    else:
                        row = {
                            "key": check.key,
                            "case_id": case.id,
                            "model": verdict.model,
                            "latency_ms": round(verdict.latency_ms, 1),
                            "input_tokens": verdict.usage.input_tokens if verdict.usage else None,
                            "output_tokens": verdict.usage.output_tokens if verdict.usage else None,
                            "decision": verdict.decision,
                            "policies": {
                                r.policy_id: {
                                    "triggered": r.triggered,
                                    "on_trigger": r.on_trigger,
                                    **raw_answer(r.raw_answer),
                                }
                                for r in verdict.results
                            },
                        }
                sink.write(json.dumps(row) + "\n")
                sink.flush()
                completed += 1
                if completed % 100 == 0 or completed == len(todo):
                    rate = completed / (time.perf_counter() - started)
                    print(f"  {completed}/{len(todo)}  ({rate:.1f}/s, {errors} errors)")

            await asyncio.gather(*(one(case, check) for case, check in todo))


def write_manifest(
    model: str, datasets: list[str], concurrency: int, no_thought: bool, tag: str | None = None
) -> None:
    path = run_dir(model, no_thought, tag) / "manifest.json"
    manifest = {
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_requested": model,
        "agent_thought_visible": not no_thought,
        "datasets": {
            n: {"repo": SOURCES[k][0], "commit": SOURCES[k][1]}
            for n, k in (("rjudge", "R-Judge"), ("injecagent", "InjecAgent"))
            if n in datasets
        },
        "policies": [
            {"id": p.id, "on_trigger": p.on_trigger, "question": type(p.question).__name__}
            for p in all_policies()
        ],
        "concurrency": concurrency,
        "actionreflex": version("actionreflex"),
        "git": _git_state(),
        "typesafe_sdk": version("typesafe-sdk"),
        "python": platform.python_version(),
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def _git_state() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "src"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{sha}+uncommitted-src-changes" if dirty else sha


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--dataset", choices=[*LOADERS, "all"], default="all")
    parser.add_argument("--model", default="jev-latest")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int, help="only the first N cases of each dataset")
    parser.add_argument(
        "--no-thought", action="store_true", help="hide the agent's reasoning from the gate"
    )
    parser.add_argument("--tag", help="suffix for the results folder, e.g. 'repeat'")
    args = parser.parse_args()

    datasets = list(LOADERS) if args.dataset == "all" else [args.dataset]
    for name in datasets:
        asyncio.run(
            run_dataset(name, args.model, args.concurrency, args.limit, args.no_thought, args.tag)
        )
    write_manifest(args.model, datasets, args.concurrency, args.no_thought, args.tag)


if __name__ == "__main__":
    main()
