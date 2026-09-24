"""Summaries of the replicate extension, on the sealed set, on the added replicates and pooled
(memo/2026-09-23-replicate-extension-prd-v2.md, section 6).

Every number is computed from the same fields, by the same functions, as the frozen index computed it:
E1 and E2 from the shard fields `make_section5_v2_figures.collect` reads, SCM-4's sealed pixel CNN from the
amended summary; E3 from the bootstrap shards; E5 and E6 by `noise_scaled.decide` and
`v3_fresh_scorer.comparator`, COCA as an ATE by `park_ate.park_parts` on the clean block; E8 by the fields
`run_kspc_design.summarise` reads.  The sealed block must reproduce the frozen index (test_extension_v1.py).

    python3 -B summarise_extension_v1.py            print
    python3 -B summarise_extension_v1.py --write    write results/extension_v1/summary_v1.json, write-once
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
EXT = RESULTS / "extension_v1"
SEALED_IDS = {"E5": range(11, 21), "E6": range(12, 22)}
REAL = {"E5": "twins", "E6": "ihdp"}


def load(p: Path) -> dict:
    return json.loads(Path(p).read_text())


def error_stats(pairs: list[tuple[float | None, float]]) -> dict:
    """pairs of (estimate, truth); an estimate of None is a data set on which the method returned nothing."""
    got = [(v, t) for v, t in pairs if v is not None and np.isfinite(v)]
    e = np.array([abs(v - t) for v, t in got], dtype=float)
    return {"data_sets": len(pairs), "returned": len(got),
            "mae": float(e.mean()) if len(e) else None,
            "mc_se": float(e.std(ddof=1) / np.sqrt(len(e))) if len(e) > 1 else None,
            "mean_error": float(np.mean([v - t for v, t in got])) if got else None}


def spread_stats(values: list[float | None]) -> dict:
    got = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if not len(got):
        return {"data_sets": len(values), "returned": 0}
    return {"data_sets": len(values), "returned": int(len(got)), "mean": float(got.mean()),
            "sd": float(got.std(ddof=1)) if len(got) > 1 else None,
            "q05": float(np.quantile(got, 0.05)), "q95": float(np.quantile(got, 0.95))}


# ----------------------------------------------------------------------------------------------- E1, E2
def scm_pairs(level: str, source: str) -> dict[str, list]:
    exp = "E2" if level == "SCM-4" else "E1"
    if source == "sealed":
        folder = RESULTS / "family_v2" / ("scm4/confirm" if exp == "E2" else "confirm")
        kfolder = RESULTS / "kspc_comparator" / exp
    else:
        folder, kfolder = EXT / exp, EXT / "kspc" / exp
    shards = [load(p) for p in sorted(folder.glob(f"{level}_seed*.json"))]
    fields = {"probe": lambda s: s["probe"]["estimate"], "cevae": lambda s: s["cevae"]["ate"],
              "p2sls_given": lambda s: s["proximal"]["p2sls_given_roles"],
              "p2sls_swapped": lambda s: s["proximal"]["p2sls_swapped_roles"],
              "park_clean": lambda s: s["proximal"]["park_spc_clean_proxy"],
              "park_contaminated": lambda s: s["proximal"]["park_spc_contaminated_proxy"],
              "raw": lambda s: s["baselines"]["raw"], "x": lambda s: s["baselines"]["X"],
              "oracle": lambda s: s["baselines"]["oracle"]}
    out = {k: [(f(s), 1.0) for s in shards] for k, f in fields.items()}
    if level == "SCM-4":
        if source == "sealed":
            amended = load(RESULTS / "family_v2" / "scm4" / "confirm_summary_v1_amended.json")
            filled = dict(zip([r["seed"] for r in amended["replay"]], amended["comparators"]["pixel_cnn"]["values"]))
            out["pixel"] = [(filled[s["task"]["seed"]], 1.0) for s in shards]
        else:
            out["pixel"] = [(s["pixel_cnn"]["ate"], 1.0) for s in shards]
    for key, block in (("kspc_clean", "W1"), ("kspc_contaminated", "W4")):
        vals = []
        for s in shards:
            p = kfolder / f"{level}_replicate_{s['task']['rep']}.json"
            vals.append((load(p)["kspc_skpv_with_x"][block]["ate"] if p.exists() else None, 1.0))
        out[key] = vals
    return out


# ----------------------------------------------------------------------------------------------- E5, E6
def real_pairs(exp: str, source: str) -> dict[str, list]:
    import noise_scaled as ns
    import v3_fresh_scorer as vs
    name = REAL[exp]
    if source == "sealed":
        paths = [RESULTS / name / f"replicate_{r}.json" for r in SEALED_IDS[exp]]
        kfolder = RESULTS / "kspc_comparator" / exp
    else:
        paths = sorted((EXT / exp).glob("replicate_*.json"), key=lambda p: int(p.stem.split("_")[1]))
        kfolder = EXT / "kspc" / exp
    shards = [load(p) for p in paths]
    out: dict[str, list] = {"probe": [(ns.decide(s["probe"], s["n"], s["seed"])["output"], s["tau"]) for s in shards]}
    for key in vs.COMPARATORS:
        out[key] = [(vs.comparator(s, key), s["tau"]) for s in shards]
    out["coca_ate"] = [(coca_ate(name, s["replicate"], source), s["tau"]) for s in shards]
    for key, block in (("kspc_clean", "W1"), ("kspc_contaminated", "W4")):
        vals = []
        for s in shards:
            p = kfolder / f"{name}_replicate_{s['replicate']}.json"
            vals.append((load(p)["kspc_skpv_with_x"][block]["ate"] if p.exists() else None, s["tau"]))
        out[key] = vals
    return out


_COCA = None


def coca_ate(name: str, r: int, source: str) -> float:
    """Sealed replicates: the frozen coca_ate_all_v1.json value.  Added ones: the same construction,
    park_ate.park_parts on the clean block of the rebuilt view."""
    global _COCA
    if source == "sealed":
        if _COCA is None:
            _COCA = load(RESULTS / "coca_ate_all_v1.json")["datasets"]
        return next(x["ate"] for x in _COCA[name]["replicates"] if x["replicate"] == r)
    import coca_ate_all as ca
    import park_ate as pa
    view, _ = ca.views_for(name, [r])[r]
    return float(pa.park_parts(view, ca.BLOCK)["ate"])


# ----------------------------------------------------------------------------------------------- E3, E8
def e3_values(source: str) -> dict[str, list]:
    """Proximal 2SLS is reported only with Cui et al.'s specification (amendment 1), from
    results/rhc_p2sls_cui/v1.json; the runner's own 2SLS, without the six other measures in X, is not."""
    folder = RESULTS / "rhc_bootstrap" if source == "sealed" else EXT / "E3"
    boots = [load(p) for p in sorted(folder.glob("boot_*.json"))]
    cui = load(RESULTS / "rhc_p2sls_cui" / "v1.json")["resamples"]
    out = {"probe_v3": [b["probe_v3"]["output"] for b in boots]}
    for k in (boots[0]["comparators"] if boots else {}):
        if k == "p2sls_cui_roles":
            out["p2sls_cui_specification"] = [cui[str(b["b"])] for b in boots]
        else:
            out[k] = [b["comparators"][k] for b in boots]
    return out


def e8_pairs(source: str) -> dict[str, list]:
    folder = RESULTS / "kspc_design" if source == "sealed" else EXT / "E8"
    rows = [load(p) for p in sorted(folder.glob("replicate_*.json"))]
    out = {"PROBE, significance": [(s["probe"]["readings"]["significance"]["output"], s["tau"]) for s in rows],
           "PROBE, v3": [(s["probe"]["readings"]["noise_floor"]["output"], s["tau"]) for s in rows]}
    for k in (rows[0]["comparators"] if rows else {}):
        out[k] = [(s["comparators"][k], s["tau"]) for s in rows]
    for k in (rows[0]["kspc"] if rows else {}):
        out[f"KSPC {k}"] = [(s["kspc"][k]["ate"], s["tau"]) for s in rows]
    return out


# ----------------------------------------------------------------------------------------------- assemble
def three_ways(get) -> dict:
    sealed, added = get("sealed"), get("extension")
    keys = list(sealed) + [k for k in added if k not in sealed]
    return {k: {"sealed": sealed.get(k, []), "extension": added.get(k, [])} for k in keys}


def summary() -> dict:
    out = {"label": "extension, run on DeltaAI after the protocol was frozen", "experiments": {}}
    for level in ("SCM-1", "SCM-2", "SCM-3", "SCM-4"):
        block = three_ways(lambda src, lv=level: scm_pairs(lv, src))
        out["experiments"][level] = {k: {"sealed": error_stats(v["sealed"]), "extension": error_stats(v["extension"]),
                                         "pooled": error_stats(v["sealed"] + v["extension"])}
                                     for k, v in block.items()}
    for exp in ("E5", "E6"):
        block = three_ways(lambda src, e=exp: real_pairs(e, src))
        out["experiments"][exp] = {k: {"sealed": error_stats(v["sealed"]), "extension": error_stats(v["extension"]),
                                       "pooled": error_stats(v["sealed"] + v["extension"])}
                                   for k, v in block.items()}
    block = three_ways(e8_pairs)
    out["experiments"]["E8"] = {k: {"sealed": error_stats(v["sealed"]), "extension": error_stats(v["extension"]),
                                    "pooled": error_stats(v["sealed"] + v["extension"])} for k, v in block.items()}
    block = three_ways(e3_values)
    out["experiments"]["E3"] = {k: {"sealed": spread_stats(v["sealed"]), "extension": spread_stats(v["extension"]),
                                    "pooled": spread_stats(v["sealed"] + v["extension"])} for k, v in block.items()}
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args(argv[1:])
    s = summary()
    for exp, block in s["experiments"].items():
        print(exp)
        for k, v in block.items():
            cells = []
            for part in ("sealed", "extension", "pooled"):
                x = v[part]
                if "mae" in x:
                    cells.append(f"{part} {x['returned']}/{x['data_sets']} MAE " + ("-" if x["mae"] is None else f"{x['mae']:.4f}"))
                else:
                    cells.append(f"{part} {x['returned']}/{x['data_sets']} mean " + ("-" if not x["returned"] else f"{x['mean']:+.3f}"))
            print(f"   {k:28s} " + " | ".join(cells))
    if a.write:
        out = EXT / "summary_v1.json"
        with open(out, "x") as handle:
            handle.write(json.dumps(s, indent=1) + "\n")
        print(f"written {out}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
