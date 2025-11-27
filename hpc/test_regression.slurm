#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --time=0-01:00:00     # days, hours minute limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=1     # 1 CPU cores
#SBATCH --mem=40G             # system memory (not gpu mem)
#SBATCH --job-name="test_DTA"
#SBATCH --output=logs/test_reg_fulltoxcast_cpu.log    # Log file

echo "Starting job with ID: $SLURM_JOB_ID"

module load Python/3.12.3-GCCcore-13.3.0


cd MGraphDTA/regression

poetry run python test.py --dataset full_toxcast --model_path <PATH TO MODEL>
