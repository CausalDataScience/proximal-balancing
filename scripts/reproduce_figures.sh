#!/usr/bin/env bash
# Rebuild the paper's figures from the shipped results, in the order they depend on each other, then list where
# each output went, which file of paper/figures/ it corresponds to, and how the two compare.  About 2 minutes.
#
#   bash scripts/reproduce_figures.sh
#
# Works from any directory.  Needs the default datasets of scripts/download_data.py (the v4-ext and v5 figures read
# the Twins, IHDP and RHC files), pdflatex for the three TikZ figures (skipped without it), and pdftotext from
# poppler for one cross-check inside the v5 script (which skips that check without it).  The PROBE mechanism figure
# draws Shapes3D images and runs only when materials/benchmarks/shapes3d/3dshapes_images_uint8.npy exists.
# Thread counts default to 2 (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, VECLIB_MAXIMUM_THREADS,
# NUMEXPR_NUM_THREADS); set any of them before calling to change it.  PYTHON picks the interpreter (default python3).
#
# Outputs, all ignored by git; paper/figures/ is only read:
#   figures/section5/     every Python figure, as .pdf and .png
#   ICLR/paper/figures/   the copies that the v3, v4-ext, v5 and mechanism scripts make (a path fixed in those frozen
#                         scripts, the layout of the conference submission; they stop if the folder is missing)
#   figures/tikz/         the three TikZ figures, with their LaTeX logs
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
for var in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS; do
  export "$var=${!var:-2}"
done
export PYTHONDONTWRITEBYTECODE=1
export MPLBACKEND="${MPLBACKEND:-Agg}"

RWD=materials/real_world_data
missing=""
for f in "$RWD/twins/twin_pairs_X_3years_samesex.csv" "$RWD/twins/twin_pairs_T_3years_samesex.csv" \
         "$RWD/twins/twin_pairs_Y_3years_samesex.csv" "$RWD/ihdp/ihdp_npci_1-100.train.npz" \
         "$RWD/ihdp/ihdp_npci_1-100.test.npz" "$RWD/rhc/rhc.csv"; do
  [ -e "$ROOT/$f" ] || missing="$missing $f"
done
if [ -n "$missing" ]; then
  echo "These inputs are missing; fetch them first with: $PYTHON scripts/download_data.py" >&2
  for f in $missing; do echo "  $f" >&2; done
  exit 2
fi

mkdir -p "$ROOT/ICLR/paper/figures"
cd "$ROOT/code"
run() {
  echo
  echo "=== code/$1"
  "$PYTHON" -B "$@"
}
run make_section5_v2_figures.py     # appendix D: aggregation, coverage, learned summary
run make_section5_v3_figures.py     # appendix D: synthetic estimates
run make_kspc_figure.py             # appendix D: KSPC benchmark
run make_section5_v4_extension.py   # appendix D: all replicates, synthetic and real data
run make_section5_v5_figure.py      # Section 5; checks its numbers against the v4-ext PDFs, so it runs after them

TIKZ="2026-09-23-proximal-balancing-structures-v2 2026-08-27-proximal-balancing-causal-graphs
      2026-08-31-majority-valid-orientations"
echo
echo "=== TikZ figures -> figures/tikz/"
if command -v pdflatex > /dev/null 2>&1; then
  mkdir -p "$ROOT/figures/tikz"
  for src in $TIKZ; do
    if ! (cd "$ROOT/figures" &&
          pdflatex -interaction=nonstopmode -halt-on-error -output-directory="$ROOT/figures/tikz" "$src.tex" > /dev/null)
    then
      echo "pdflatex failed on figures/$src.tex; the end of figures/tikz/$src.log:" >&2
      tail -n 20 "$ROOT/figures/tikz/$src.log" >&2 || true
      exit 1
    fi
    echo "wrote figures/tikz/$src.pdf"
  done
  TIKZ_STATUS=built
else
  TIKZ_STATUS="skipped: pdflatex not found"
  echo "$TIKZ_STATUS"
fi

CACHE=materials/benchmarks/shapes3d/3dshapes_images_uint8.npy
if [ -f "$ROOT/$CACHE" ]; then
  run make_probe_mechanism_figure_v2.py   # Section 4
  MECH_STATUS=built
else
  MECH_STATUS="skipped: needs the Shapes3D image cache $CACHE (5.9 GB)"
  echo
  echo "=== code/make_probe_mechanism_figure_v2.py"
  echo "$MECH_STATUS.  To build it: $PYTHON scripts/download_data.py --only shapes3d (268 MB), then, from code/,"
  echo "  $PYTHON -B -c 'import family_v2_images as im; im.build_cache()'   (about 12 GB of free disk needed)"
fi

echo
echo "=== where each figure went (compared with the paper's copy in paper/figures/)"
TIKZ_STATUS="$TIKZ_STATUS" MECH_STATUS="$MECH_STATUS" "$PYTHON" - "$ROOT" << 'EOF'
import os
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
S5, TK, ICLR = "figures/section5", "figures/tikz", "ICLR/paper/figures"
# (paper figure, where it appears, the file that was built, its copy in ICLR/paper/figures or None, step status)
FIGURES = [
    ("causal-structures-3panel-v2.pdf", "Section 1", f"{TK}/2026-09-23-proximal-balancing-structures-v2.pdf", None, "tikz"),
    ("probe-mechanism-v2.pdf", "Section 4", f"{S5}/probe-mechanism-v2.pdf", "probe-mechanism-v2.pdf", "mech"),
    ("probe-experiments-v5.pdf", "Section 5", f"{S5}/section5-v5-experiments.pdf", "probe-experiments-v5.pdf", ""),
    ("causal-graphs.pdf", "appendix B", f"{TK}/2026-08-27-proximal-balancing-causal-graphs.pdf", None, "tikz"),
    ("majority-orientations.pdf", "appendix B", f"{TK}/2026-08-31-majority-valid-orientations.pdf", None, "tikz"),
    ("probe-estimates-v3.pdf", "appendix D", f"{S5}/section5-v3-estimates.pdf", "probe-estimates-v3.pdf", ""),
    ("probe-kspc.pdf", "appendix D", f"{S5}/kspc-benchmark.pdf", None, ""),
    ("probe-synthetic-v4-ext.pdf", "appendix D", f"{S5}/section5-v4-synthetic-ext.pdf", "probe-synthetic-v4-ext.pdf", ""),
    ("probe-realdata-v4-ext.pdf", "appendix D", f"{S5}/section5-v4-realdata-ext.pdf", "probe-realdata-v4-ext.pdf", ""),
    ("probe-aggregation.pdf", "appendix D", f"{S5}/section5-v2-aggregation.pdf", None, ""),
    ("probe-coverage.pdf", "appendix D", f"{S5}/section5-v2-coverage.pdf", None, ""),
    ("probe-learned.pdf", "appendix D", f"{S5}/section5-v2-learned.pdf", None, ""),
]
STATUS = {"tikz": os.environ["TIKZ_STATUS"], "mech": os.environ["MECH_STATUS"], "": "built"}


def masked(data: bytes) -> bytes:
    """The PDF with its creation and modification dates and its file ID blanked: the parts that change per build."""
    data = re.sub(rb"/(CreationDate|ModDate)\s*\(D:[^)]*\)", rb"/\1()", data)
    return re.sub(rb"/ID\s*\[\s*<[0-9A-Fa-f]*>\s*<[0-9A-Fa-f]*>\s*\]", b"/ID[]", data)


counts = {}
for paper, where, built, copy, step in FIGURES:
    print(f"paper/figures/{paper}  ({where})")
    if STATUS[step] != "built":
        verdict = STATUS[step]
        print(f"    not built ({verdict})")
    else:
        new, ref = (root / built).read_bytes(), (root / "paper" / "figures" / paper).read_bytes()
        verdict = ("byte-identical" if new == ref else
                   "identical apart from the embedded creation date and file ID" if masked(new) == masked(ref) else
                   "DIFFERS in content: compare the pages")
        print(f"    built: {built}" + (f", copied to {ICLR}/{copy}" if copy else ""))
        print(f"    {verdict}")
    counts[verdict.split(":")[0]] = counts.get(verdict.split(":")[0], 0) + 1
print("\nalso written, not in the paper: figures/section5/section5-v2-estimates.pdf and section5-v2-budget.pdf, and a "
      ".png beside every PDF in figures/section5/")
print("summary: " + "; ".join(f"{n} {v}" for v, n in counts.items()))
print("(byte comparisons assume the tools of the paper's build, matplotlib 3.11.0 and pdfTeX 1.40.29 of TeX Live 2026;\n"
      " with other versions the files differ in bytes and the pages are to be compared by eye or with pdftoppm)")
EOF
