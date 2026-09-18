#!/bin/bash
#SBATCH --job-name=scm7v4
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=18
#SBATCH --mem=120G
#SBATCH --time=3:45:00
#SBATCH --output=logs/scm7v4_%j.out
#SBATCH --error=logs/scm7v4_%j.err
# One GH200, seeds run one after another so the charge is a single number: wall time x one GPU.
# --time is a cap, not a charge.  The CPU-bound r search runs on the node's 18 cores in parallel;
# it does not change any number, it only keeps the GPU from idling while scipy works.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
mkdir -p logs
module load python/3.11.9
source .venv/bin/activate
export PYTHONPATH=code
export OMP_NUM_THREADS=1
python3 - <<'PY'
import torch; assert torch.cuda.is_available(), "no CUDA device visible"
print("device", torch.cuda.get_device_name(0), "| torch", torch.__version__, flush=True)
PY
cd code
echo "== SCM-7-v4 confirmation: 15 fresh seeds, image-specific screen tolerance =="
python3 -u run_scm7_v4.py --workers 16
echo "== done =="
