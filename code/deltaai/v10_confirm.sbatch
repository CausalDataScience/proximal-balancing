#!/bin/bash
#SBATCH --job-name=v10conf
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=100G
#SBATCH --time=3:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/v10conf_%j.out
#SBATCH --error=logs/v10conf_%j.err
# Confirmatory single-image run (memo/2026-10-10-pneumoniamnist-benchmark-v1.md, section 38): 50 replicates x 8 scenarios
# x 2 block layouts on one GH200 share, so the charge is at most 3 GPU-hours (Yonghan: "GPU 예산 3시간. 최대한 아껴").
# A short GPU smoke test runs first; then 16 workers with equal loads share the GPU, each writing its own file of one
# JSON line per finished replicate into
# results/single_image/pneumoniamnist_v1/.  No worker starts a replicate within 10 minutes of the cap, and every
# file is resumable, so a partial run loses nothing that finished.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
START=$(date +%s); DEADLINE=$(( START + 3 * 3600 - 600 ))
module load python/3.11.9
source /work/hdd/bgtp/yjung6/probe_ext_v1/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 - <<'PY'
import numpy, scipy, sklearn, torch
assert torch.cuda.is_available(), "no CUDA device visible"
print("device", torch.cuda.get_device_name(0), "| torch", torch.__version__, "| numpy", numpy.__version__, flush=True)
PY
W=code/single_image/realimg/realimg_v10_confirm.py
echo "== GPU smoke (800 images, one epoch, scenario 3 in each layout) =="
python3 -B -u "$W" --layout anat --smoke --device cuda --threads 4 --scenarios 3
python3 -B -u "$W" --layout grid4 --smoke --device cuda --threads 4 --scenarios 3
echo "== main: 16 workers with equal loads; worker w runs scenario w mod 8, replicates of parity w div 8 (25 each), anat first then grid4; deadline $(date -d @$DEADLINE) =="
pids=()
for w in $(seq 0 15); do
  c=$(( w % 8 )); par=$(( w / 8 ))
  ( for L in anat grid4; do
      python3 -B -u "$W" --layout "$L" --scenarios "$c" --reps 50 --rep-mod 2 --rep-res "$par" --tag "r50_w$w" --device cuda --threads 4 --deadline "$DEADLINE"
    done ) > "logs/v10_${SLURM_JOB_ID}_w$w.log" 2>&1 &
  pids+=($!)
done
trap 'echo "TERM received, waiting for workers to flush"; wait; exit 0' TERM
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "== lines per file =="
wc -l results/single_image/pneumoniamnist_v1/v10_*_r50_w*.jsonl || true
echo "== done after $(( $(date +%s) - START )) s, worker exit status $rc =="
exit $rc
