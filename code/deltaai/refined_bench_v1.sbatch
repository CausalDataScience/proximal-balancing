#!/bin/bash
#SBATCH --job-name=rblk2
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100G
#SBATCH --time=0:30:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/rblk2_%j.out
#SBATCH --error=logs/rblk2_%j.err
# Blocks discovered from the proxy alone and refined by k-subspaces (memo section 43): SCM-1 to SCM-3 (60 data sets,
# 30 workers) and Twins, IHDP, ACIC (197 replicates, 40 workers), run side by side.  CPU only: one GH200 share (72 cores)
# for at most 30 minutes = 0.5 GPU-hours; with the 0.32 GPU-hours of jobs 3414491 and 3416055 the total stays within
# Yonghan's cap of 2 GPU-hours (at most 0.83).  Expected about 15 minutes (Twins PROBE took 190 s per replicate in E5).
# Every result file is written when its data set ends, and no data set starts within 6 minutes of the limit.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results_refined results_bench
START=$(date +%s); DEADLINE=$(( START + 30 * 60 - 6 * 60 ))
module load python/3.11.9
source /work/hdd/bgtp/yjung6/probe_ext_v1/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -c "import numpy, scipy, pandas; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| pandas', pandas.__version__, flush=True)"
python3 -B -u code/run_discovered_blocks.py discovered_blocks_v1_tasks.json results_refined --workers 30 --refined --deadline "$DEADLINE" > logs/rblk2_${SLURM_JOB_ID}_scm.log 2>&1 &
python3 -B -u code/run_benchmarks_discovered.py benchmarks_v1_tasks.json results_bench --workers 40 --deadline "$DEADLINE" > logs/rblk2_${SLURM_JOB_ID}_bench.log 2>&1 &
wait
echo "done $(date)"
