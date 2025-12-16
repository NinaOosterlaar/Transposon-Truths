#!/bin/bash
#SBATCH --job-name=preprocessing
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=100G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

BINS=(5 10 20)

BIN_IDX=$((SLURM_ARRAY_TASK_ID / ${#MA_FLAGS[@]}))

BIN=${BINS[$BIN_IDX]}

echo "Task ${SLURM_ARRAY_TASK_ID}: --bin ${BIN}"

srun apptainer exec \
  --bind "$PROJECT_DIR":/workspace \
  --pwd /workspace \
  "$APPTAINER_IMAGE" \
  python AE/preprocessing.py \
    --bin "$BIN" \
    --zinb_mode \
    --no_moving_average \
    --train_val_test_split 0.7 0.0 0.3