#!/bin/bash
#SBATCH --job-name=sliding_ZINB_cpd
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=20G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err
#SBATCH --array=0

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

cd "$PROJECT_DIR"

# Array of dataset names
DATASETS=("SATAY_synthetic")

DATASET_NAME=${DATASETS[$SLURM_ARRAY_TASK_ID]}
INPUT_FILE="Signal_processing/sample_data/SATAY_synthetic.csv"

echo "Running sliding ZINB CPD on dataset: ${DATASET_NAME}"
echo "Input file: ${INPUT_FILE}"

srun apptainer exec \
    --bind "$PROJECT_DIR":/workspace \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    python Signal_processing/sliding_mean/sliding_ZINB_CPD.py \
    "$INPUT_FILE" \
    --dataset_name "${DATASET_NAME}" \
    --n_workers 4 \
    --output_folder "Signal_processing/results/sliding_mean/sliding_ZINB_CPD/"

echo "Finished processing ${DATASET_NAME}"

