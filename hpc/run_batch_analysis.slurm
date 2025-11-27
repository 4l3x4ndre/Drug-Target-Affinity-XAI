#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=0-50:00:00     # 0 days, hours minute limit
#SBATCH --nodes=1             #  compute nodes
#SBATCH --cpus-per-task=1     #  CPU cores
#SBATCH --mem=10G              # Gigabytes system memory (not gpu mem)
#SBATCH --gres=gpu:v100:1
#SBATCH --job-name="B_DTA"
#SBATCH --output=logs/tmp_placeholder.log  # Temporary placeholder

TIMESTAMP=$(date +"%Y%m%d_%H%M")
LOGFILE="logs/banalysis_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOGFILE" 2>&1 

echo "Starting job with ID: $SLURM_JOB_ID"
echo "Logging to: $LOGFILE"


module load Python/3.12.3-GCCcore-13.3.0

poetry run python -m  xai.run_batch_analysis --device "cuda:0" "$@" --output-dir "results/batch_analysis/"


echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"



