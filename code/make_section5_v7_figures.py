"""Figure 3 version 7 and the appendix detail figures version 5: PROBE with the blocks found from the proxy
(2026-10-20, memo section 43).

Same figures, data sets, comparators, layout and checks as make_section5_v6_figure (Figure 3) and
make_section5_v4_extension (the detailed synthetic and real-world figures of the appendix), with one change: in the
SCM-1 to SCM-3, Twins, IHDP and ACIC panels, and for RHC, PROBE's values are those of the merged discovered blocks
(discovered_probe_errors), in the same order of data sets and under the same rule.  Every comparator row and every
check against the stored indexes runs on the original data, so the checks still certify the comparators; PROBE's new
values are checked against their own files here (count, order, finiteness) and printed beside the old ones.  The
image panels (Shapes3D and the two chest X-ray scenarios) are unchanged: their blocks are given.

    python3 -B make_section5_v7_figures.py [--copy]     (--copy: write probe-experiments-v7.pdf,
                                                         probe-synthetic-v5-found.pdf, probe-realdata-v5-found.pdf
                                                         to manuscript/figures when every check passes)
"""
from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", SyntaxWarning)
    import discovered_probe_errors as dp  # noqa: E402
    import make_section5_v4_extension as ext  # noqa: E402
    import make_section5_v6_figure as f6  # noqa: E402

f5, v4 = f6.f5, ext.v4
MANUSCRIPT_FIGURES = CODE.parent / "manuscript" / "figures"   # the older scripts copy to ICLR/paper/figures
v4.PAPER_FIGURES = MANUSCRIPT_FIGURES
SYNTH, REAL = ("SCM-1", "SCM-2", "SCM-3"), ("twins", "ihdp", "acic")


def patched(data: dict, available: dict, panels) -> dict:
    new = {k: dict(v) for k, v in data.items()}
    for p in panels:
        e = dp.synth_errors(p) if p in SYNTH else dp.real_errors(p)
        if len(e) != len(data[p]["probe"]) or len(e) != available[p] or not np.all(np.isfinite(e)):
            raise ValueError(f"{p}: {len(e)} discovered-block errors ({np.isfinite(e).sum()} finite), figure holds {len(data[p]['probe'])}")
        print(f"  {p:6s} PROBE MAE, blocks given {np.mean(np.abs(data[p]['probe'])):.4f} -> blocks found {np.mean(np.abs(e)):.4f} ({len(e)} data sets)")
        new[p]["probe"] = e
    return new


def figure3(copy: bool) -> int:
    orig: dict = {}
    load, mae_table, probe_check = f5.load, f5.mae_table, f5.probe_mae_check

    def load_found():
        data, available = load()
        orig["data"] = data
        print("== Figure 3, version 7: PROBE with the blocks found from W")
        return patched(data, available, SYNTH + REAL), available

    f5.load = load_found
    f5.mae_table = lambda data: mae_table(orig["data"])            # the stored-index checks, on the original data
    f5.probe_mae_check = lambda data: probe_check(orig["data"])
    f6.STEM, f6.PAPER_COPY = "section5-v7-experiments", MANUSCRIPT_FIGURES / "probe-experiments-v7.pdf"
    return f6.main(["--copy"] if copy else [])


def appendix(copy: bool) -> int:
    orig: dict = {}
    load_s, load_r, load_rhc, check_mae = ext.load_synthetic, ext.load_real, v4.load_rhc, v4.check_mae

    def synth():
        data, available = load_s(); orig.update(data)
        print("== appendix detail figures, version 5: PROBE with the blocks found from W")
        return patched(data, available, SYNTH), available

    def real():
        data, available = load_r(); orig.update(data)
        return patched(data, available, REAL), available

    def rhc():
        out, paths = load_rhc()
        old = out["probe"]
        out["probe"] = dp.rhc()
        print(f"  RHC    PROBE, blocks given {old[0]:.3f} {tuple(round(x, 3) for x in old[1])} -> blocks found "
              f"{out['probe'][0]:.3f} {tuple(round(x, 3) for x in out['probe'][1])}")
        return out, paths

    ext.load_synthetic, ext.load_real, v4.load_rhc = synth, real, rhc
    index_rhc = v4.index_rhc

    def index_found():                             # the RHC check: PROBE against its own file, the rest against the index
        idx = index_rhc()
        out, (lo, hi) = dp.rhc()
        idx["probe"] = [out, lo, hi]
        return idx

    v4.index_rhc = index_found
    v4.check_mae = lambda errors, panels, index, decimals, groups: check_mae({p: orig[p] for p in panels}, panels, index, decimals, groups)
    ext.STEMS = {"synthetic": ("section5-v5-synthetic-found", "probe-synthetic-v5-found.pdf"),
                 "realdata": ("section5-v5-realdata-found", "probe-realdata-v5-found.pdf")}
    if not copy:                                   # make_section5_v4_extension copies unconditionally; keep it local
        ext.shutil = type("NoCopy", (), {"copyfile": staticmethod(lambda a, b: print(f"(not copied: {Path(a).name})"))})
    return ext.main()


if __name__ == "__main__":
    copy = "--copy" in sys.argv[1:]
    which = [a for a in sys.argv[1:] if a in ("figure3", "appendix")] or ["figure3", "appendix"]
    codes = [figure3(copy) if w == "figure3" else appendix(copy) for w in which]
    raise SystemExit(max(codes))
