"""Code integrity for the KSPC comparator runs: every generator file each frozen protocol hashed must still
hash the same, so that the same seed rebuilds exactly the sealed data.  This closes the one gap the COCA
check leaves, a constant added to Y, which COCA cannot see and KSPC's intercept-free ridge can.

    python3 -B kspc_comparator_integrity.py
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import run_family_v2 as rf

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
OUT = R / "kspc_comparator" / "code_integrity_v1.json"


def recorded() -> dict[str, dict[str, str]]:
    v3 = json.loads((R / "v3_confirmation_protocol_v2.json").read_text())["manifests"]
    return {"E1": json.loads((R / "family_v2/confirm_protocol_v1.json").read_text())["source_sha256"],
            "E2": json.loads((R / "family_v2/scm4/confirm_protocol_v1.json").read_text())["source_sha256"],
            "E5": v3["twins"]["expected"]["11"]["source_sha256"],
            "E6": v3["ihdp"]["expected"]["12"]["source_sha256"],
            "E7": json.loads((R / "acic/replicate_2.json").read_text())["source_sha256"]}


def check() -> dict:
    out = {}
    for exp, rec in recorded().items():
        now = {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest() for f in rec}
        out[exp] = {"files": len(rec), "differ": [f for f in rec if now[f] != rec[f]]}
    out["pass"] = all(not v["differ"] for v in out.values() if isinstance(v, dict))
    return out


if __name__ == "__main__":
    if OUT.exists():
        raise SystemExit(f"{OUT} exists; write-once")
    res = {"artifact": "kspc_comparator_code_integrity", "generated_utc": rf.now(), **check(),
           "E3": "the RHC file is checked against its sha256 by rhc_data.load"}
    OUT.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res, indent=1))
