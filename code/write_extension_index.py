"""EXPERIMENT_INDEX_EXTENSION_v1.md from results/extension_v1/summary_v1.json
(memo/2026-09-23-replicate-extension-prd-v2.md, section 6).  The frozen EXPERIMENT_INDEX.md is not touched.

    python3 -B write_extension_index.py [--charges "smoke 1.00, recheck 0.26, main X"]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = ROOT / "EXPERIMENT_INDEX_EXTENSION_v1.md"
NAMES = {"probe": "**PROBE**", "PROBE, significance": "**PROBE** (reported rule)", "PROBE, v3": "PROBE, v3 rule",
         "probe_v3": "**PROBE**, v3 rule", "cevae": "CEVAE", "p2sls_given": "2SLS, roles given",
         "p2sls_swapped": "2SLS, roles swapped", "p2sls_given_roles": "2SLS, roles given",
         "p2sls_swapped_roles": "2SLS, roles swapped", "park_clean": "COCA, clean block",
         "park_contaminated": "COCA, contaminated block", "coca_ate": "COCA as an ATE", "raw": "adjust for X and all of W",
         "x": "adjust for X", "X": "adjust for X", "oracle": "oracle (adjusts for X and U)", "pixel": "pixel CNN",
         "kspc_clean": "KSPC with X, clean block†", "kspc_contaminated": "KSPC with X, contaminated block†",
         "naive": "difference in means", "adjust_X": "adjust for X", "adjust_X_and_all_W": "adjust for X and all ten proxies",
         "p2sls_cui_specification": "Proximal 2SLS, Cui et al.'s specification",
         "p2sls_W1_treats_W2_outcome": "2SLS, W1 treats", "p2sls_W2_treats_W1_outcome": "2SLS, W2 treats",
         "coca_W1_ate": "COCA, given W1", "coca_W2_ate": "COCA, given the image"}
TITLES = {"SCM-1": "E1, SCM-1", "SCM-2": "E1, SCM-2", "SCM-3": "E1, SCM-3", "SCM-4": "E2, SCM-4 (Shapes3D images)",
          "E5": "E5, Twins", "E6": "E6, IHDP", "E8": "E8, KSPC benchmark", "E3": "E3, observed RHC, bootstrap resamples"}


def cell(x: dict, digits: int) -> str:
    if x.get("mae") is None:
        return "–"
    se = f" ± {x['mc_se']:.{digits}f}" if x.get("mc_se") is not None else ""
    ret = "" if x["returned"] == x["data_sets"] else f" ({x['returned']}/{x['data_sets']} returned)"
    return f"{x['mae']:.{digits}f}{se}{ret}"


def spread(x: dict) -> str:
    if not x.get("returned"):
        return "–"
    return f"{x['mean']:+.2f} (sd {x['sd']:.2f}; 90% {x['q05']:+.2f} to {x['q95']:+.2f})"


def table(exp: str, block: dict) -> list[str]:
    digits = 4 if exp in ("E5", "E6") else 3
    if exp == "E3":
        lines = ["| effect on days survived | sealed resamples | added resamples | all |", "|---|---|---|---|"]
        for k, v in block.items():
            lines.append(f"| {NAMES.get(k, k)} | {spread(v['sealed'])} | {spread(v['extension'])} | {spread(v['pooled'])} |")
        return lines
    rows = sorted(block.items(), key=lambda kv: (kv[1]["pooled"]["mae"] is None, kv[1]["pooled"]["mae"] or 0))
    n = {part: max(v[part]["data_sets"] for v in block.values()) for part in ("sealed", "extension", "pooled")}
    lines = [f"| MAE ± Monte Carlo SE | sealed ({n['sealed']}) | added ({n['extension']}) | pooled ({n['pooled']}) |",
             "|---|---|---|---|"]
    for k, v in rows:
        lines.append(f"| {NAMES.get(k, k)} | {cell(v['sealed'], digits)} | {cell(v['extension'], digits)} | "
                     f"{cell(v['pooled'], digits)} |")
    return lines


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--charges", default="")
    ap.add_argument("--summary", type=Path, default=ROOT / "results" / "extension_v1" / "summary_v1.json")
    a = ap.parse_args(argv[1:])
    s = json.loads(a.summary.read_text())
    lines = ["# Experiment index, extension v1: added replicates",
             "",
             "Companion to the frozen `EXPERIMENT_INDEX.md`, which is not edited. Plan and decisions:",
             "`memo/2026-09-23-replicate-extension-prd-v2.md` (amendments 1 to 3). Protocol, frozen before any added",
             "replicate ran: `results/extension_v1/protocol_v1.json`. Summary: `results/extension_v1/summary_v1.json`.",
             "",
             "## How the added replicates were produced",
             "",
             "- Every added replicate ran through the frozen runner of its experiment, unchanged, with identifiers and",
             "  seeds fixed in the protocol. The sealed replicates are untouched; tables show them, the added ones and",
             "  both together.",
             "- The added replicates ran on NCSA DeltaAI (Arm Neoverse V2 cores; the SCM-4 reader and pixel CNN on an",
             "  H100), the sealed ones on an Apple M4. Rerunning sealed data sets on DeltaAI gave the same data and",
             "  PROBE outputs that differ from the Mac values by 0.0002 to 0.009 (E3: 0.028), each less than half the",
             "  spread of that experiment's sealed outputs (`results/extension_v1/gate_*.json`). The gate first fixed,",
             "  equality to $10^{-6}$, failed; the author replaced it after seeing that (amendment 3).",
             "- RHC's proximal 2SLS is reported only with Cui et al.'s specification (amendment 1); Cui et al.'s",
             "  standard DR is not drawn (amendment 2).",
             "- Seeds of the sealed E1 set: SCM-1 replicates 10 to 19 share seeds with SCM-2 replicates 0 to 9, and",
             "  SCM-2 10 to 19 with SCM-3 0 to 9, so those pairs share random draws; comparisons within a level are",
             "  unaffected. The added seeds have no such overlap.",
             "- IHDP has 88 data sets, not the planned 89. The file indexes its realizations 0 to 99; the protocol",
             "  named 22 to 100, and 100 does not exist (that task failed with an index error). Realization 0, never",
             "  used, was not in the protocol and was not run.",
             "- SCM-4's pixel CNN finished within its cap on all 80 added data sets. Two sealed data sets had passed",
             "  the cap on the Mac and were filled by the amended summary, as before.",
             f"- Charge on DeltaAI, GPU-hours: {a.charges}." if a.charges else "",
             "",
             "Figures: `figures/section5/extension-estimates.pdf`, `extension-realdata.pdf`,",
             "`extension-kspc-benchmark.pdf`, `extension-rhc-bootstrap.pdf` (source `code/make_extension_figures.py`).",
             "† KSPC with covariates, our implementation of its Appendix A, added after the protocol was frozen.",
             ""]
    for exp in ("SCM-1", "SCM-2", "SCM-3", "SCM-4", "E5", "E6", "E8", "E3"):
        lines += [f"## {TITLES[exp]}", ""] + table(exp, s["experiments"][exp]) + [""]
    OUT.write_text("\n".join(l for l in lines if l is not None) + "\n")
    print(f"written {OUT}")
    return 0


if __name__ == "__main__":
    import sys
    raise SystemExit(main(sys.argv))
