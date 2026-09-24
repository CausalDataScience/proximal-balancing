#!/bin/bash
#SBATCH --job-name=probe_ext_main
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=144
#SBATCH --mem=220G
#SBATCH --time=3:30:00
#SBATCH --signal=B:TERM@180
#SBATCH --output=logs/probe_ext_main_%j.out
#SBATCH --error=logs/probe_ext_main_%j.err
# Step C of memo/2026-09-23-replicate-extension-prd-v2.md, with amendment 3.  Two GPU shares (2 of 4 GPUs, 144 of
# 288 cores, 220 of about 440 GB), so the most this job can cost is 2 x 3.5 h = 7.0 GPU-hours.  The project tree,
# without the virtual environment, runs from the node's memory; results are copied back to /work every five
# minutes and at the end.  The driver stops launching a task that would not finish before the limit.
# EXTRA may carry --skip-cevae-e1 or --only, the fallbacks section 5 fixes in advance.
set -euo pipefail
SRC="$SLURM_SUBMIT_DIR"
STAGE="/dev/shm/probe_ext_v1_${SLURM_JOB_ID}"
mkdir -p "$SRC/logs" "$STAGE"
module load python/3.11.9
source "$SRC/.venv/bin/activate"
rsync -a --exclude .venv --exclude scratch --exclude logs "$SRC/" "$STAGE/"
back() { rsync -a "$STAGE/results/extension_v1/" "$SRC/results/extension_v1/" 2>/dev/null || true; }
trap 'back; rm -rf "$STAGE"' EXIT
trap 'back; exit 0' TERM
( while sleep 300; do back; done ) &
python3 -c "import numpy, scipy, sklearn, pandas, torch; assert torch.cuda.device_count() == 2; print('warm', torch.__version__, [torch.cuda.get_device_name(k) for k in range(2)], flush=True)"
# CPU-only tasks must not see the GPUs: in job 3198612 each opened a CUDA context while seeding torch, and
# about a hundred contexts exhausted GPU memory.  The driver gives each GPU task its own device, "0" or "1".
export CUDA_VISIBLE_DEVICES=""
cd "$STAGE/code"
python3 -B run_extension_v1.py check-sources
python3 -B -u run_extension_v1.py run --cores 144 --gpus 2 --per-gpu 4 --time-limit-s 12400 ${EXTRA:-} &
wait $! || echo "driver exit status $?"
echo "== done: main =="
