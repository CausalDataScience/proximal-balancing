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
    parser.add_argument("--mode", choices=("hard4", "rotate4"), default="hard4")
    parser.add_argument("--learner", choices=("moment", "brier"), default="moment")
    args = parser.parse_args()
    protocol_path = Path(args.protocol)
    protocol = json.loads(protocol_path.read_text())
    expected = protocol["source_sha256"]["probe_structured_scm.py"]
    current = source_sha256(Path(__file__).with_name("probe_structured_scm.py"))
    if current != expected:
        raise SystemExit("source hash differs from the frozen protocol")
    if args.phase not in protocol["allowed_phases"]:
        raise SystemExit("phase absent from frozen protocol")
    if args.mode not in protocol.get("allowed_modes", ["hard4"]):
        raise SystemExit("mode absent from frozen protocol")
    requested = {
        "phase": args.phase,
        "mode": args.mode,
        "learner": args.learner,
        "replicates": args.replicates,
        "seed_start": args.seed_start,
        "settings": args.settings,
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
                run = run_brier_diagnostic_replicate
            else:
                run = run_rotating_replicate if args.mode == "rotate4" else run_moment_replicate
            rows.append(run(spec, protocol["total_n"], seed_base))
    payload = {
        "complete": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase": args.phase,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_sha256": source_sha256(protocol_path),
        "source_sha256": current,
        "settings": args.settings,
        "replicates": args.replicates,
        "seed_start": args.seed_start,
        "mode": args.mode,
        "learner": args.learner,
        "rows": rows,
    }
    write_json_atomic(args.output, payload)


if __name__ == "__main__":
    main()
