#!/bin/bash
#SBATCH --job-name=v7sup128
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=2:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/v7sup128_%j.out
#SBATCH --error=logs/v7sup128_%j.err
# 128-px supervised-encoder benchmark (memo/2026-10-10-pneumoniamnist-benchmark-v1.md, sections 32-33).
# One GH200 share for at most 2 hours, so the charge is at most 2.0 GPU-hours (the budget Yonghan approved on
# 2026-10-11).  A smoke replicate runs first; then four workers, one per scenario V0-V3, run ten replicates each
# on the same GPU and write one JSON line per finished replicate to results/single_image/pneumoniamnist_v1/.
# Each worker refuses to start a replicate that is not expected to finish before the deadline (cap minus 5 min),
# so a partial run leaves complete, resumable lines.  Results are written straight to /work; nothing is staged.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
START=$(date +%s); DEADLINE=$(( START + 7200 - 300 ))
module load python/3.11.9
source "${VENV:-/work/hdd/bgtp/yjung6/probe_ext_v1/.venv}/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6 MKL_NUM_THREADS=6
python3 - <<'PY'
import numpy, scipy, sklearn, torch
assert torch.cuda.is_available(), "no CUDA device visible"
print("device", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| numpy", numpy.__version__, "| scipy", scipy.__version__, "| sklearn", sklearn.__version__, flush=True)
PY
W=code/single_image/realimg/realimg_v7_sup128.py
echo "== smoke (800 images, 1 epoch, V3) =="
python3 -B -u "$W" --smoke --device cuda --threads 6
echo "== main: four workers, one per scenario, reps 10, tag r10gpu_V<c>, deadline $(date -d @$DEADLINE) =="
pids=()
for c in 0 1 2 3; do
  python3 -B -u "$W" --scenarios "$c" --reps 10 --tag "r10gpu_V$c" --device cuda --threads 6 --ignore-window --deadline "$DEADLINE" \
    > "logs/v7sup128_${SLURM_JOB_ID}_V$c.log" 2>&1 &
  pids+=($!)
done
trap 'echo "TERM received, waiting for workers to flush"; wait; exit 0' TERM
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "== lines per scenario file =="
for c in 0 1 2 3; do f="results/single_image/pneumoniamnist_v1/v7sup128_r10gpu_V$c.jsonl"; printf '%s %s\n' "$f" "$( [ -f "$f" ] && wc -l < "$f" || echo 0 )"; done
echo "== done after $(( $(date +%s) - START )) s, worker exit status $rc =="
exit $rc
