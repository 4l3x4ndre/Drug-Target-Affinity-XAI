#!/bin/sh
#SBATCH --partition=CPUQ
#SBATCH --time=0-01:00:00     # 1 hour limit
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --job-name="PREP_GEN_FOLDS"
#SBATCH --output=logs/prep_generate_folds_davis.log    # Log file

module load Python/3.12.3-GCCcore-13.3.0


echo "Generating folds..."
poetry run python -m MGraphDTA.regression.train_cv --dataset kiba --dataset_path /cluster/home/alexaamr/drug-target-affinity-xai/MGraphDTA/regression/data --generate_folds 
