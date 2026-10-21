#!/bin/bash
#SBATCH --job-name=lrt1
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100G
#SBATCH --time=0:20:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/lrt1_%j.out
#SBATCH --error=logs/lrt1_%j.err
# Development comparison for the likelihood-ratio refinement (memo section 43): the 80 SCM-2 and SCM-3 data sets already
# seen (E1 seeds and confirmation seeds), merged discovered blocks with penalty 3.84, to compare with the stored runs
# without the penalty.  CPU only: one GH200 share (72 cores) for at most 20 minutes = 0.33 GPU-hours; with the 0.76
# GPU-hours used so far (jobs 3414491, 3416055, 3419235, 3419470, 3421791) the total stays within Yonghan's cap of 2
# GPU-hours (at most 1.10).  Expected under 10 minutes.  No data set starts within 6 minutes of the limit.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results_lrt_dev
START=$(date +%s); DEADLINE=$(( START + 20 * 60 - 6 * 60 ))
module load python/3.11.9
source /work/hdd/bgtp/yjung6/probe_ext_v1/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -c "import numpy, scipy, pandas; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| pandas', pandas.__version__, flush=True)"
python3 -B -u code/run_discovered_blocks.py lrt_dev_v1_tasks.json results_lrt_dev --workers 50 --merged --penalty 3.84 --deadline "$DEADLINE" > logs/lrt1_${SLURM_JOB_ID}.log 2>&1
echo "done $(date)"
