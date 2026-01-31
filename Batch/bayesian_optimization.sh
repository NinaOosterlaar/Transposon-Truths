#!/bin/sh
#SBATCH --job-name=bayesOpt_%a
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=72:00:00
#SBATCH --qos=long
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=500G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_bayesopt_%A_%a.out
#SBATCH --error=slurm_bayesopt_%A_%a.err
#SBATCH --array=0-1

set -euo pipefail

# Define metrics array
METRICS=("zinb_nll" "combined")
METRIC="${METRICS[$SLURM_ARRAY_TASK_ID]}"

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

echo "Running Bayesian optimization with metric: $METRIC"

srun apptainer exec \
  --nv \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python AE/training/bayesian_hyperparameter.py \
    --n_calls 300 \
    --n_initial_points 20 \
    --random_state 42 \
    --n_jobs 2 \
    --metric "$METRIC"

