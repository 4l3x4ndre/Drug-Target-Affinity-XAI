#!/bin/sh
#SBATCH --partition=GPUQ
#SBATCH --time=0-20:00:00     # 0 days, 20 hours 00 minute limit
#SBATCH --nodes=1             # 1 compute nodes
#SBATCH --cpus-per-task=1     # 1 CPU cores
#SBATCH --mem=7G              # 7 gigabytes system memory (not gpu mem)
#SBATCH --gres=gpu:p100:1
#SBATCH --job-name="TRAIN_DTA"
#SBATCH --output=logs/train_reg_davis.log    # Log file

module load Python/3.12.3-GCCcore-13.3.0

cd MGraphDTA/regression

poetry run python preprocessing.py

poetry run python train.py --dataset kiba --save_model --device cuda:0
