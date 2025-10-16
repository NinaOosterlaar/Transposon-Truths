#!/bin/bash
#SBATCH --job-name=piece3
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --qos=medium
#SBATCH --time=30:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --mem-per-cpu=20G
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

srun python Data_exploration/polynomial_regression.py --degree 3 --piecewise