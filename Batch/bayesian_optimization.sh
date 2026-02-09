#!/bin/sh
#SBATCH --job-name=bayesOpt
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=100:00:00
#SBATCH --qos=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1000G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_bayesopt_%A_%a.out
#SBATCH --error=slurm_bayesopt_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail


export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"


srun apptainer exec \
  --nv \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python AE/training/bayesian_hyperparameter.py \
    --n_calls 150 \
    --n_initial_points 20 \
    --random_state 42 \
    --n_jobs 1 \
    --metric "combined"

