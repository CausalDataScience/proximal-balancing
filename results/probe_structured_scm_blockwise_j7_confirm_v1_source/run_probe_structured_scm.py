"""CLI for the frozen small-data structured SCM experiment."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from probe_structured_scm import (
    SCMSpec,
    run_moment_replicate,
    run_brier_diagnostic_replicate,
    run_rotating_replicate,
    source_sha256,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--phase", choices=("diagnostic", "dev", "confirm"), required=True)
    parser.add_argument("--settings", nargs="+", required=True)
    parser.add_argument("--replicates", type=int, required=True)
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--mode", choices=("hard4", "rotate4"), required=True)
    parser.add_argument("--learner", choices=("moment", "brier"), default="moment")
    parser.add_argument("--primary-budget-m", type=int, default=None)
    parser.add_argument("--budget-sweep", default="")
    args = parser.parse_args()
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    current_sources = {}
    for filename, expected in protocol["source_sha256"].items():
        source_path = Path(__file__).with_name(filename)
        if not source_path.exists():
            raise SystemExit(f"required frozen source is missing: {filename}")
        current_sources[filename] = source_sha256(source_path)
        if current_sources[filename] != expected:
            raise SystemExit(f"source hash differs from the frozen protocol: {filename}")
    if args.phase not in protocol["allowed_phases"]:
        raise SystemExit("phase absent from frozen protocol")
    if args.mode not in protocol.get("allowed_modes", ["hard4"]):
        raise SystemExit("mode absent from frozen protocol")
    sweep = tuple(int(x) for x in args.budget_sweep.split(",") if x.strip())
    requested = {
        "phase": args.phase,
        "mode": args.mode,
        "mode_scope": (
            "production_operational_crossfit" if args.mode == "rotate4" else "one_shot_diagnostic"
        ),
        "learner": args.learner,
        "replicates": args.replicates,
        "seed_start": args.seed_start,
        "settings": args.settings,
        "primary_budget_m": args.primary_budget_m,
        "budget_sweep": list(sweep),
    }
    allowed_runs = protocol.get("runs")
    if allowed_runs is not None and requested not in allowed_runs:
        raise SystemExit(f"CLI does not match an exact frozen run: {requested}")
    rows = []
    for setting in args.settings:
        if setting not in protocol["specs"]:
            raise SystemExit(f"unfrozen setting: {setting}")
        spec = SCMSpec(**protocol["specs"][setting])
        for rep in range(args.replicates):
            seed_base = args.seed_start + 10 * rep
            if args.learner == "brier":
                run = (
                    (lambda chosen_spec, chosen_n, chosen_seed: run_rotating_replicate(
                        chosen_spec, chosen_n, chosen_seed, learner="brier",
                        primary_budget_m=args.primary_budget_m, budget_sweep=sweep,
                    ))
                    if args.mode == "rotate4"
                    else run_brier_diagnostic_replicate
                )
            else:
                run = run_rotating_replicate if args.mode == "rotate4" else run_moment_replicate
            rows.append(run(spec, protocol["total_n"], seed_base))
            print(
                json.dumps({"setting": setting, "completed": rep + 1, "total": args.replicates}),
                flush=True,
            )
    payload = {
        "complete": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": source_sha256(protocol_path),
        "source_sha256": current_sources,
        "settings": args.settings,
        "replicates": args.replicates,
        "seed_start": args.seed_start,
        "mode": args.mode,
        "learner": args.learner,
        "primary_budget_m": args.primary_budget_m,
        "budget_sweep": list(sweep),
        "rows": rows,
    }
    write_json_atomic(args.output, payload)


if __name__ == "__main__":
    main()
