#!/bin/bash
#SBATCH --job-name=cnf1
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100G
#SBATCH --time=0:30:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/cnf1_%j.out
#SBATCH --error=logs/cnf1_%j.err
# Confirmation of block discovery on new data (memo section 43): Twins replicates 111-210 and 20 new seeds per SCM level,
# each run with the designed blocks and with the merged discovered blocks (rule fixed in job 3419470), plus the Twins
# comparison methods X, raw, oracle, 2SLS and KSPC.  CPU only: one GH200 share (72 cores) for at most 30 minutes =
# 0.5 GPU-hours; with the 0.55 GPU-hours of jobs 3414491, 3416055, 3419235 and 3419470 the total stays within Yonghan's
# cap of 2 GPU-hours (at most 1.05).  Expected about 15 minutes.  Every result file is written when its task ends, and
# no task starts within 6 minutes of the limit.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results_confirm
START=$(date +%s); DEADLINE=$(( START + 30 * 60 - 6 * 60 ))
module load python/3.11.9
source /work/hdd/bgtp/yjung6/probe_ext_v1/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -c "import numpy, scipy, pandas; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| pandas', pandas.__version__, flush=True)"
python3 -B -u code/run_confirmation_v1.py confirm_v1_tasks.json results_confirm --workers 70 --deadline "$DEADLINE" > logs/cnf1_${SLURM_JOB_ID}.log 2>&1
echo "done $(date)"
