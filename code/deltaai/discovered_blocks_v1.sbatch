#!/bin/bash
#SBATCH --job-name=dblk1
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100G
#SBATCH --time=0:40:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/dblk1_%j.out
#SBATCH --error=logs/dblk1_%j.err
# PROBE with blocks discovered from the proxy alone (memo/2026-10-10-pneumoniamnist-benchmark-v1.md, section 43, the fix
# after job 3414491).  CPU only: one GH200 share (72 cores) for at most 40 minutes = 0.67 GPU-hours; with the 0.16
# GPU-hours of job 3414491 the total stays within Yonghan's cap of 2 GPU-hours.  60 data sets (SCM-1 to SCM-3, 20 each)
# run in parallel, one per worker; every result file is written when its data set ends, and none starts within
# 8 minutes of the limit.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results_discovered
START=$(date +%s); DEADLINE=$(( START + 40 * 60 - 8 * 60 ))
module load python/3.11.9
source /work/hdd/bgtp/yjung6/probe_ext_v1/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -c "import numpy, scipy; print('numpy', numpy.__version__, '| scipy', scipy.__version__, flush=True)"
python3 -B -u code/run_discovered_blocks.py discovered_blocks_v1_tasks.json results_discovered --workers 60 --deadline "$DEADLINE"
echo "done $(date)"
