"""The replacement gate of amendment 3 (memo/2026-09-23-replicate-extension-prd-v2.md), applied to sealed items
rerun on DeltaAI.  Three checks per item: the rebuilt data equal the sealed data (COCA, or for E3 and E8 a
deterministic comparator, within 1e-8 relative; SCM-4 within 1e-3 because its reader runs on CUDA); PROBE's
reported output differs from the sealed one by less than half the standard deviation of that experiment's
sealed outputs; the item finished, and for SCM-4 the pixel CNN returned within its cap.

    python3 -B extension_v1_gate.py <folder with the rerun shards> <name>    writes results/extension_v1/gate_<name>.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parent / "results"
HALF_SD = {"E1": 0.0113, "E2": 0.0126, "E5": 0.0046, "E6": 0.247, "E3": 0.133, "E8": 0.0059}
ITEMS = {"E1": ("E1/SCM-1_seed97200000.json", R / "family_v2/confirm/SCM-1_seed97200000.json"),
         "E2": ("E2/SCM-4_seed97406000.json", R / "family_v2/scm4/confirm/SCM-4_seed97406000.json"),
         "E3": ("E3/boot_1.json", R / "rhc_bootstrap/boot_1.json"),
         "E5": ("E5/replicate_11.json", R / "twins/replicate_11.json"),
         "E6": ("E6/replicate_12.json", R / "ihdp/replicate_12.json"),
         "E8": ("E8/replicate_1.json", R / "kspc_design/replicate_1.json")}


def probe_output(exp: str, s: dict):
    import noise_scaled as ns
    if exp in ("E1", "E2"):
        return s["probe"]["estimate"]
    if exp == "E3":
        return s["probe_v3"]["output"]
    if exp == "E8":
        return s["probe"]["readings"]["significance"]["output"]
    return ns.decide(s["probe"], s["n"], s["seed"])["output"]


def data_gap(exp: str, sealed: dict, mine: dict) -> float:
    rel = lambda a, b: abs(a - b) / max(1.0, abs(a))
    if exp == "E3":
        return rel(sealed["comparators"]["naive"], mine["comparators"]["naive"])
    if exp == "E8":
        return max(rel(sealed["comparators"][k], mine["comparators"][k]) for k in ("naive", "coca_W1_ate"))
    return max(rel(sealed["proximal"][k], mine["proximal"][k])
               for k in ("park_spc_clean_proxy", "park_spc_contaminated_proxy"))


def gate(folder: Path) -> dict:
    out = {}
    for exp, (rel_path, sealed_path) in ITEMS.items():
        path = folder / rel_path
        if not path.exists():
            out[exp] = {"present": False}
            continue
        sealed, mine = json.loads(sealed_path.read_text()), json.loads(path.read_text())
        a, b = probe_output(exp, sealed), probe_output(exp, mine)
        gap = None if (a is None or b is None) else abs(a - b)
        dgap = data_gap(exp, sealed, mine)
        row = {"present": True, "data_gap": dgap, "data_ok": dgap <= (1e-3 if exp == "E2" else 1e-8),
               "probe_sealed": a, "probe_rerun": b, "probe_gap": gap, "limit": HALF_SD[exp],
               "probe_ok": (a is None and b is None) or (gap is not None and gap < HALF_SD[exp])}
        if exp == "E2":
            row["pixel_cnn"] = mine["pixel_cnn"].get("ate")
            row["pixel_cnn_seconds"] = mine["pixel_cnn"].get("seconds")
            row["pixel_ok"] = mine["pixel_cnn"].get("finite", False)
        row["pass"] = row["data_ok"] and row["probe_ok"] and row.get("pixel_ok", True)
        out[exp] = row
    return out


def main(argv: list[str]) -> int:
    folder, name = Path(argv[1]), argv[2]
    result = gate(folder)
    for exp, row in result.items():
        if not row["present"]:
            print(f"{exp}: not in this folder")
            continue
        extra = f", pixel CNN {row['pixel_cnn']} in {row['pixel_cnn_seconds']} s" if exp == "E2" else ""
        print(f"{exp}: data gap {row['data_gap']:.1e} | PROBE {row['probe_sealed']} vs {row['probe_rerun']}, gap "
              f"{row['probe_gap']} against {row['limit']}{extra} -> {'PASS' if row['pass'] else 'FAIL'}")
    out = R / "extension_v1" / f"gate_{name}.json"
    with open(out, "x") as handle:
        handle.write(json.dumps({"folder": str(folder), "items": result}, indent=1, default=str) + "\n")
    print(f"written {out}")
    return 0 if all(r["pass"] for r in result.values() if r["present"]) else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
