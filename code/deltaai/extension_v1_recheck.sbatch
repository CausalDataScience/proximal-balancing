#!/bin/bash
#SBATCH --job-name=probe_ext_recheck
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=0:30:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/probe_ext_recheck_%j.out
#SBATCH --error=logs/probe_ext_recheck_%j.err
# Amendment 3 of memo/2026-09-23-replicate-extension-prd-v2.md.  Reruns the sealed E2 (SCM-4 rep 6), E3, E6 and
# E8 items into scratch/.  The project tree, without the virtual environment, is copied to the node's memory
# and run there, so the SCM-4 images are read from memory.  One GPU share for at most 30 minutes: 0.5 GPU-hours.
set -euo pipefail
SRC="$SLURM_SUBMIT_DIR"
STAGE="/dev/shm/probe_ext_v1_${SLURM_JOB_ID}"
mkdir -p "$SRC/logs" "$SRC/scratch" "$STAGE"
module load python/3.11.9
source "$SRC/.venv/bin/activate"
rsync -a --exclude .venv --exclude scratch --exclude logs "$SRC/" "$STAGE/"
back() { rsync -a "$STAGE/scratch/" "$SRC/scratch/" 2>/dev/null || true; }
trap 'back; rm -rf "$STAGE"' EXIT
trap 'back; exit 0' TERM
( while sleep 300; do back; done ) &
python3 -c "import numpy, scipy, sklearn, pandas, torch; assert torch.cuda.is_available(); print('warm', torch.__version__, torch.cuda.get_device_name(0), flush=True)"
cd "$STAGE/code"
python3 -B -u run_extension_v1.py reproduce --items E2 E3 E6 E8 --root "../scratch/recheck_${SLURM_JOB_ID}" \
    --cores 16 --gpus 1 --per-gpu 2 --time-limit-s 1500 &
wait $! || echo "driver exit status $?"
python3 -B run_extension_v1.py compare --items E2 E3 E6 E8 --root "../scratch/recheck_${SLURM_JOB_ID}" || true
echo "== done: recheck =="
