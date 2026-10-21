#!/bin/bash
#SBATCH --job-name=v3scm
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100G
#SBATCH --time=0:20:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/v3scm_%j.out
#SBATCH --error=logs/v3scm_%j.err
# The v3 rule on SCM (memo section 43): rerun the merged discovered blocks on the 60 E1 data sets, and the designed and
# merged blocks on the 60 confirmation data sets, now storing the screen statistics so that noise_scaled.decide can be
# applied exactly as on the benchmarks (the E1 files already hold them for the designed blocks).  CPU only: one GH200
# share (72 cores) for at most 20 minutes = 0.33 GPU-hours; with the 0.84 GPU-hours used so far (jobs 3414491,
# 3416055, 3419235, 3419470, 3421791, 3422366) the total stays within Yonghan's cap of 2 GPU-hours (at most 1.17).
# Expected about 10 minutes.  No data set starts within 6 minutes of the limit.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results_v3_e1 results_v3_confirm
START=$(date +%s); DEADLINE=$(( START + 20 * 60 - 6 * 60 ))
module load python/3.11.9
source /work/hdd/bgtp/yjung6/probe_ext_v1/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -c "import numpy, scipy, pandas; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| pandas', pandas.__version__, flush=True)"
python3 -B -u code/run_discovered_blocks.py discovered_blocks_v1_tasks.json results_v3_e1 --workers 25 --merged --deadline "$DEADLINE" > logs/v3scm_${SLURM_JOB_ID}_e1.log 2>&1 &
python3 -B -u code/run_confirmation_v1.py v3_confirm_tasks.json results_v3_confirm --workers 35 --deadline "$DEADLINE" > logs/v3scm_${SLURM_JOB_ID}_confirm.log 2>&1 &
wait
echo "done $(date)"
