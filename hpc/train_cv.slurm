#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=0-40:00:00     # 0 days, 20 hours 00 minute limit
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --job-name="CV_TRAIN_DTA"
#SBATCH --output=logs/cv_train_davis_%A_%a.log    # Log file for each array job

# --- JOB ARRAY CONFIGURATION ---
# Total jobs = repeats * k_folds
# Default in train_cv.py is repeats=3, k_folds=5, so 15 jobs (0-14)
#SBATCH --array=0-14

module load Python/3.12.3-GCCcore-13.3.0

# --- FOLD AND REPEAT CALCULATION ---
# These should match the --repeats and --k_folds used for generation
K_FOLDS=5
REPEAT_NUM=$((SLURM_ARRAY_TASK_ID / K_FOLDS))
FOLD_NUM=$((SLURM_ARRAY_TASK_ID % K_FOLDS))

echo "Starting training for repeat ${REPEAT_NUM}, fold ${FOLD_NUM}"

poetry run python -m MGraphDTA.regression.train_cv \
    --dataset kiba \
    --dataset_path <PATH TO DATASET>
    --repeat ${REPEAT_NUM} \
    --fold ${FOLD_NUM} \
    --device cuda:0 \
    --wandb_log

