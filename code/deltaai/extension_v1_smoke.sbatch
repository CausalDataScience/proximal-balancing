#!/bin/bash
#SBATCH --job-name=probe_ext_smoke
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --output=logs/probe_ext_smoke_%j.out
#SBATCH --error=logs/probe_ext_smoke_%j.err
# Step B of memo/2026-09-23-replicate-extension-prd-v2.md.  The GPU share dominates the charge (1 of 4 GPUs;
# 16 of 288 cores; 100 of about 440 GB), so the most this job can cost is 1.0 GPU-hour.  It reruns sealed items
# into scratch/ and writes nothing to results/extension_v1.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs scratch
module load python/3.11.9
source .venv/bin/activate
python3 - <<'PY'
import torch; assert torch.cuda.is_available(), "no CUDA device visible"
print("device", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
PY
lscpu | grep -E "Model name|^CPU\(s\)" || true
cd code
python3 -B run_extension_v1.py check-sources
python3 -B -u run_extension_v1.py reproduce --root "../scratch/smoke_${SLURM_JOB_ID}" --cores 16 --gpus 1 --per-gpu 2 --time-limit-s 3400
python3 -B run_extension_v1.py compare --root "../scratch/smoke_${SLURM_JOB_ID}"
echo "== done: smoke =="
