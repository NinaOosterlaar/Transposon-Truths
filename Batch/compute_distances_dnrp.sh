#!/bin/bash
#SBATCH --job-name=distances_dnrp
#SBATCH --account=ewi-insy-prb
#SBATCH --partition=general,insy
#SBATCH --qos=medium 
#SBATCH --time=16:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --mail-type=ALL
#SBATCH --mail-user=n.i.m.oosterlaar@student.tudelft.nl
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

module use /opt/insy/modulefiles  # If not already
module load miniconda

source ~/.bashrc
conda activate env

cd /tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis

srun python Data_exploration/reader.py --input_dir Data/wiggle_format/strain_dnrp --output_dir Data_exploration/results/distances_with_zeros --with_zeros