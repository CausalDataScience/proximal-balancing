"""Run the frozen E9 H11/H12/H21/H22 component ablation."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from probe_structured_scm import (
    SCMSpec,
    run_e9_component_replicate,
    source_sha256,
    write_json_atomic,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    protocol_path = Path(args.protocol).resolve()
    protocol = json.loads(protocol_path.read_text())
    if protocol.get("protocol_id") != "e9-hard4-component-ablation-v1":
        raise SystemExit("unexpected protocol_id")
    current_sources = {}
    for filename, expected in protocol["source_sha256"].items():
        path = Path(__file__).with_name(filename)
        if not path.exists():
            raise SystemExit(f"required frozen source is missing: {filename}")
        current_sources[filename] = source_sha256(path)
        if current_sources[filename] != expected:
            raise SystemExit(f"source hash differs from frozen protocol: {filename}")
    references = {}
    for role, record in protocol["reference_results"].items():
        path = (protocol_path.parent.parent / record["path"]).resolve()
        if source_sha256(path) != record["sha256"]:
            raise SystemExit(f"reference result hash mismatch: {role}")
        payload = json.loads(path.read_text())
        if not payload.get("complete"):
            raise SystemExit(f"reference result is incomplete: {role}")
        references[role] = payload
    run = protocol["run"]
    spec = SCMSpec(**protocol["spec"])
    rows = []
    for rep in range(run["replicates"]):
        seed_base = run["seed_start"] + 10 * rep
        row = run_e9_component_replicate(spec, protocol["total_n"], seed_base)
        for role, payload in references.items():
            expected = payload["rows"][rep]
            hashes = expected.get("fold_sha256", expected.get("role_sha256"))
            if row["fold_sha256"] != hashes:
                raise SystemExit(f"fold hash mismatch for {role}, replicate {rep}")
        rows.append(row)
        print(json.dumps({"completed": rep + 1, "total": run["replicates"]}), flush=True)
    output = {
        "complete": True,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "protocol_path": str(protocol_path),
        "protocol_sha256": source_sha256(protocol_path),
        "source_sha256": current_sources,
        "reference_result_sha256": {
            role: record["sha256"] for role, record in protocol["reference_results"].items()
        },
        "total_n": protocol["total_n"],
        "replicates": run["replicates"],
        "seed_start": run["seed_start"],
        "rows": rows,
        "claim_boundary": protocol["claim_boundary"],
    }
    write_json_atomic(args.output, output)


if __name__ == "__main__":
    main()
