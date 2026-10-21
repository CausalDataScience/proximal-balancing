#!/bin/bash
#SBATCH --job-name=mblk1
#SBATCH --account=bgtp-dtai-gh
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=72
#SBATCH --mem=100G
#SBATCH --time=0:30:00
#SBATCH --signal=B:TERM@120
#SBATCH --output=logs/mblk1_%j.out
#SBATCH --error=logs/mblk1_%j.err
# Blocks discovered from the proxy alone, refined by k-subspaces, then coupled groups merged (memo section 43): SCM-1 to SCM-3 (60 data sets,
# 30 workers) and Twins, IHDP, ACIC (197 replicates, 40 workers), run side by side.  CPU only: one GH200 share (72 cores)
# for at most 30 minutes = 0.5 GPU-hours; with the 0.46 GPU-hours of jobs 3414491, 3416055 and 3419235 the total stays within
# Yonghan's cap of 2 GPU-hours (at most 0.96).  Expected under 10 minutes (job 3419235 took 8 minutes with 30 splits; J = 4 has 14).
# Every result file is written when its data set ends, and no data set starts within 6 minutes of the limit.
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"; mkdir -p logs results_merged results_bench_merged
START=$(date +%s); DEADLINE=$(( START + 30 * 60 - 6 * 60 ))
module load python/3.11.9
source /work/hdd/bgtp/yjung6/probe_ext_v1/.venv/bin/activate
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python3 -c "import numpy, scipy, pandas; print('numpy', numpy.__version__, '| scipy', scipy.__version__, '| pandas', pandas.__version__, flush=True)"
python3 -B -u code/run_discovered_blocks.py discovered_blocks_v1_tasks.json results_merged --workers 30 --merged --deadline "$DEADLINE" > logs/mblk1_${SLURM_JOB_ID}_scm.log 2>&1 &
python3 -B -u code/run_benchmarks_discovered.py benchmarks_v1_tasks.json results_bench_merged --workers 40 --merged --deadline "$DEADLINE" > logs/mblk1_${SLURM_JOB_ID}_bench.log 2>&1 &
wait
echo "done $(date)"
