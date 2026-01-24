#!/bin/sh
#SBATCH --job-name=BayesOpt
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=64G
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_bayesopt_%j.out
#SBATCH --error=slurm_bayesopt_%j.err

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

echo "Starting Bayesian Hyperparameter Optimization"
echo "CPUs available: $SLURM_CPUS_PER_TASK"
echo "Memory per CPU: 8G"
echo "Total memory: $((SLURM_CPUS_PER_TASK * 8))G"

srun apptainer exec \
  --nv \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python AE/training/bayesian_hyperparameter.py \
    --n_calls 150 \
    --n_initial_points 20 \
    --random_state 42 \
    --n_jobs 10

