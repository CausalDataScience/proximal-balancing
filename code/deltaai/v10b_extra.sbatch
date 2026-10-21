#!/bin/bash
#SBATCH --job-name=v10extra
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=56
#SBATCH --mem=100G
#SBATCH --time=2:00:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/v10extra_%j.out
#SBATCH --error=logs/v10extra_%j.err
# Second confirmatory job (memo/2026-10-10-pneumoniamnist-benchmark-v1.md, section 40): hand-drawn blocks only, seven new
# scenarios (8: device + response in two blocks; 9-13: imprint sweep kappa = 0, .5, 1, 2, 3; 14: near-deterministic device),
# 50 replicates each, 14 workers with equal loads (one scenario, one replicate parity, 25 replicates) on one GH200 share.
# Cap two hours (Yonghan, 2026-10-13: "제출해. 예산은 2시간"), so the charge is at most 2 GPU-hours; expected 0.3-0.4.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs
START=$(date +%s); DEADLINE=$(( START + 7200 - 480 ))
module load python/3.11.9
source /work/hdd/bgtp/yjung6/probe_ext_v1/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4
python3 -c "import torch; assert torch.cuda.is_available(); print('device', torch.cuda.get_device_name(0), '| torch', torch.__version__, flush=True)"
W=code/single_image/realimg/realimg_v10_confirm.py
echo "== GPU smoke (800 images, one epoch, scenario 8) =="
python3 -B -u "$W" --layout anat --smoke --device cuda --threads 4 --scenarios 8
echo "== main: 14 workers; worker w runs scenario (8 + w mod 7), replicates of parity w div 7; deadline $(date -d @$DEADLINE) =="
pids=()
for w in $(seq 0 13); do
  c=$(( 8 + w % 7 )); par=$(( w / 7 ))
  python3 -B -u "$W" --layout anat --scenarios "$c" --reps 50 --rep-mod 2 --rep-res "$par" --tag "r50x_w$w" --device cuda --threads 4 --deadline "$DEADLINE" \
    > "logs/v10extra_${SLURM_JOB_ID}_w$w.log" 2>&1 &
  pids+=($!)
done
trap 'echo "TERM received, waiting for workers to flush"; wait; exit 0' TERM
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo "== lines per file =="
wc -l results/single_image/pneumoniamnist_v1/v10_anat_r50x_w*.jsonl || true
echo "== done after $(( $(date +%s) - START )) s, worker exit status $rc =="
exit $rc
