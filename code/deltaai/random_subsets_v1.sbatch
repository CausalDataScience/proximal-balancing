#!/bin/bash
#SBATCH --job-name=rsub1
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100G
#SBATCH --time=1:30:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/rsub1_%j.out
#SBATCH --error=logs/rsub1_%j.err
# PROBE with coordinate-level held-out sets drawn at random (memo/2026-10-10-pneumoniamnist-benchmark-v1.md, section 43,
# plan 1; Yonghan 2026-10-19: "1안으로 해. GPU 상한은 2시간").  CPU only: one GH200 share (72 cores) for at most
# 1.5 hours, so the charge is at most 1.5 GPU-hours.  100 data sets (SCM-1 uniform and single, SCM-2 uniform and
# single, SCM-3 single; 20 each, round robin by replicate) run one after another, each with its 4 x 30 units on 70
# workers; every result file is written when its data set ends, and no data set starts within 8 minutes of the limit.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results
START=$(date +%s); DEADLINE=$(( START + 90 * 60 - 8 * 60 ))
module load python/3.11.9
source /work/hdd/bgtp/yjung6/probe_ext_v1/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -c "import numpy, scipy; print('numpy', numpy.__version__, '| scipy', scipy.__version__, flush=True)"
python3 -B -u code/run_random_subsets_parallel.py random_subsets_v1_tasks.json results --workers 70 --deadline "$DEADLINE"
echo "done $(date)"
