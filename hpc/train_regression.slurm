#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0-40:00:00     # days, hours minute limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=1     # 1 CPU cores
#SBATCH --mem=8G             # system memory (not gpu mem)
#SBATCH --job-name="TRAIN_DTA"
#SBATCH --output=logs/tmp_placeholder.log  # Temporary placeholder
TIMESTAMP=$(date +"%Y%m%d_%H")
LOGFILE="logs/train_reg_${TIMESTAMP}_${SLURM_JOB_ID}.log"
exec > "$LOGFILE" 2>&1 

echo "Starting job with ID: $SLURM_JOB_ID"
echo "Logging to: $LOGFILE"
echo "Running with arguments: $@"


module load Python/3.12.3-GCCcore-13.3.0

poetry run python -m MGraphDTA.regression.train --dataset kiba --save_model --device cuda:0 --dataset_path <PATH TO DATASET> --wandb_log

echo "Job ended on $(date)"
echo "Elapsed time: $(( SECONDS / 3600 ))h $(( (SECONDS % 3600) / 60 ))m $(( SECONDS % 60 ))s"


