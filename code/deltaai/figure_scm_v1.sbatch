#!/bin/bash
#SBATCH --job-name=fscm1
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100G
#SBATCH --time=0:20:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/fscm1_%j.out
#SBATCH --error=logs/fscm1_%j.err
# The rest of Figure 3's synthetic data sets with the merged discovered blocks (memo section 43): the 20 sealed and the
# 60 remaining E1 data sets per level of SCM-1 to SCM-3 (240; the other 20 E1 per level ran in job 3422452), same data,
# fold seed and tolerance as the figure's runs.  CPU only: one GH200 share (72 cores) for at most 20 minutes = 0.33
# GPU-hours; with the 0.96 GPU-hours used since the cap was set (jobs 3414491 to 3422452) the total stays within
# Yonghan's cap of 2 GPU-hours (at most 1.29).  Expected about 8 minutes.  No data set starts within 6 minutes of the limit.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results_figure_scm
START=$(date +%s); DEADLINE=$(( START + 20 * 60 - 6 * 60 ))
module load python/3.11.9
source /work/hdd/bgtp/yjung6/probe_ext_v1/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -c "import numpy, scipy, pandas; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| pandas', pandas.__version__, flush=True)"
python3 -B -u code/run_discovered_blocks.py figure_scm_v1_tasks.json results_figure_scm --workers 70 --merged --deadline "$DEADLINE" > logs/fscm1_${SLURM_JOB_ID}.log 2>&1
echo "done $(date)"
