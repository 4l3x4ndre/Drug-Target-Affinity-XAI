#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --gres=gpu:1
#SBATCH --time=0-20:00:00     # days, hours minute limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=1     # 1 CPU cores
#SBATCH --mem=24G             # system memory (not gpu mem)
#SBATCH --job-name="TRAIN_DTA_CV"
#SBATCH --output=logs/train_reg_cv_fulltoxcast_gpu.log    # Log file

echo "Starting job with ID: $SLURM_JOB_ID"

module load Python/3.12.3-GCCcore-13.3.0

poetry run python -m MGraphDTA.regression.train_cv --dataset full_toxcast --save_model_dir <SAVE MODEL DIR> --device cuda:0 --dataset_path <DATASET PATH> --wandb_log
