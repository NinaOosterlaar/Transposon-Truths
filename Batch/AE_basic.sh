#!/bin/bash
#SBATCH --job-name=AE_basic
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64G
#SBATCH --array=0-7
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_%A_%a.out
#SBATCH --error=slurm_%A_%a.err

set -euo pipefail

export APPTAINER_IMAGE="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/my-container.sif"
export PROJECT_DIR="/tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis"

module use /opt/insy/modulefiles
module load cuda/12.4

cd "$PROJECT_DIR"

# Map array index -> model and filename
# Jobs 0-3: AE with BinSize 5, 10, 50, 100
# Jobs 4-7: VAE with BinSize 5, 10, 50, 100
MODELS=("AE" "AE" "AE" "AE" "VAE" "VAE" "VAE" "VAE")
FILENAMES=("Features['Pos', 'Chrom', 'Nucl', 'Centr']_SplitOnDataset_BinSize5_DataPointLen2000_StepSize500_NormalizeTrue_MovingAvgTrue_"
           "Features['Pos', 'Chrom', 'Nucl', 'Centr']_SplitOnDataset_BinSize10_DataPointLen2000_StepSize500_NormalizeTrue_MovingAvgTrue_"
           "Features['Pos', 'Chrom', 'Nucl', 'Centr']_SplitOnDataset_BinSize50_DataPointLen2000_StepSize500_NormalizeTrue_MovingAvgTrue_"
           "Features['Pos', 'Chrom', 'Nucl', 'Centr']_SplitOnDataset_BinSize100_DataPointLen2000_StepSize500_NormalizeTrue_MovingAvgTrue_"
           "Features['Pos', 'Chrom', 'Nucl', 'Centr']_SplitOnDataset_BinSize5_DataPointLen2000_StepSize500_NormalizeTrue_MovingAvgTrue_"
           "Features['Pos', 'Chrom', 'Nucl', 'Centr']_SplitOnDataset_BinSize10_DataPointLen2000_StepSize500_NormalizeTrue_MovingAvgTrue_"
           "Features['Pos', 'Chrom', 'Nucl', 'Centr']_SplitOnDataset_BinSize50_DataPointLen2000_StepSize500_NormalizeTrue_MovingAvgTrue_"
           "Features['Pos', 'Chrom', 'Nucl', 'Centr']_SplitOnDataset_BinSize100_DataPointLen2000_StepSize500_NormalizeTrue_MovingAvgTrue_")

MODEL=${MODELS[$SLURM_ARRAY_TASK_ID]}
FILENAME=${FILENAMES[$SLURM_ARRAY_TASK_ID]}

echo "Running model: $MODEL with file: $FILENAME (SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID})"

srun apptainer exec \
    --nv \
    --bind "$PROJECT_DIR":/workspace \
    --pwd /workspace \
    "$APPTAINER_IMAGE" \
    python AE/training.py --model "$MODEL" --filename "$FILENAME"