#!/usr/bin/env bash
# Run the test suite: 15 test files, 222 tests (one skipped by design), about 4 minutes on two threads.
#
#   bash scripts/run_tests.sh                          every test file
#   bash scripts/run_tests.sh test_rhc test_kspc       only these files
#
# Works from any directory.  Needs the default datasets of scripts/download_data.py (checked first).
# Thread counts default to 2 (OMP_NUM_THREADS, MKL_NUM_THREADS, OPENBLAS_NUM_THREADS, VECLIB_MAXIMUM_THREADS,
# NUMEXPR_NUM_THREADS); set any of them before calling to change it.  PYTHON picks the interpreter (default python3).
# The unit-test files run as `python3 -B test_<name>.py` from code/; the three pytest-style files run under pytest.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
for var in OMP_NUM_THREADS MKL_NUM_THREADS OPENBLAS_NUM_THREADS VECLIB_MAXIMUM_THREADS NUMEXPR_NUM_THREADS; do
  export "$var=${!var:-2}"
done
export PYTHONDONTWRITEBYTECODE=1        # the tests must not leave __pycache__ folders behind
export MPLBACKEND="${MPLBACKEND:-Agg}"

UNITTEST="test_family_v2 test_scm4_images test_scm4_run test_probe_structured_scm test_positive_control
          test_noise_scaled test_v3_fresh test_rhc test_rhc_planted test_twins test_ihdp_acic test_extension_v1"
PYTEST="test_kspc test_kspc_comparator test_two_proxy"

RWD=materials/real_world_data
TWINS="$RWD/twins/twin_pairs_X_3years_samesex.csv $RWD/twins/twin_pairs_T_3years_samesex.csv
       $RWD/twins/twin_pairs_Y_3years_samesex.csv"
IHDP="$RWD/ihdp/ihdp_npci_1-100.train.npz $RWD/ihdp/ihdp_npci_1-100.test.npz"

inputs_of() {  # the downloaded files a test file reads
  case "$1" in
    test_twins | test_kspc_comparator) echo "$TWINS" ;;
    test_extension_v1) echo "$TWINS $IHDP" ;;
    test_ihdp_acic) echo "$IHDP $RWD/acic2016/x.csv $RWD/acic2016/zymu_2.csv $RWD/acic2016/zymu_3.csv" ;;
    test_rhc | test_rhc_planted) echo "$RWD/rhc/rhc.csv" ;;
    test_two_proxy) echo "$RWD/zika/Zika_Brazil_2013.csv $RWD/lalonde/NSW_AFDC_CS.dta $RWD/lalonde/nsw_dw.dta
                          $RWD/lalonde/cps_controls.dta $RWD/lalonde/cps_controls2.dta $RWD/lalonde/cps_controls3.dta
                          $RWD/lalonde/psid_controls.dta $RWD/lalonde/psid_controls2.dta
                          $RWD/lalonde/psid_controls3.dta" ;;
    test_kspc) echo "materials/benchmarks/mnist/mnist.npz materials/external_code/KernelSingleProxy/src
                     materials/external_code/KernelSingleProxy/configs" ;;
  esac
}

if [ "$#" -gt 0 ]; then
  selected=""
  for name in "$@"; do
    name="$(basename "${name%.py}")"
    case " $(echo $UNITTEST $PYTEST) " in
      *" $name "*) selected="$selected $name" ;;
      *) echo "unknown test file: $name (known: $(echo $UNITTEST $PYTEST))" >&2; exit 2 ;;
    esac
  done
else
  selected="$UNITTEST $PYTEST"
fi

missing=""
for t in $selected; do
  for f in $(inputs_of "$t"); do
    [ -e "$ROOT/$f" ] || case "$missing" in *" $f"*) ;; *) missing="$missing $f" ;; esac
  done
done
if [ -n "$missing" ]; then
  echo "These inputs are missing; fetch them first with: $PYTHON scripts/download_data.py" >&2
  for f in $missing; do echo "  $f" >&2; done
  exit 2
fi

cd "$ROOT/code"
echo "python: $("$PYTHON" -c 'import sys; print(sys.executable, sys.version.split()[0])'); threads: $OMP_NUM_THREADS"
summary=""
failures=0
for t in $selected; do
  echo
  echo "=== $t"
  start=$SECONDS
  case " $PYTEST " in
    *" $t "*) cmd=("$PYTHON" -B -m pytest -q -p no:cacheprovider "$t.py") ;;
    *) cmd=("$PYTHON" -B "$t.py") ;;
  esac
  if "${cmd[@]}"; then
    status="passed"
  else
    status="FAILED (exit $?)"
    failures=$((failures + 1))
  fi
  summary="$summary$(printf '  %-28s %-16s %5ss' "$t" "$status" "$((SECONDS - start))")
"
done

echo
echo "=== summary ($(echo $selected | wc -w | tr -d ' ') test files, $failures failed)"
printf '%s' "$summary"
[ "$failures" -eq 0 ]
