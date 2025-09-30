#!/bin/bash
#SBATCH --job-name=nina-plots
#SBATCH --partition=general
#SBATCH --qos=short
#SBATCH --time=00:05:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=n.i.m.oosterlaar@tudelft.nl
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

set -euo pipefail

module use /opt/insy/modulefiles  # If not already
module load miniconda

source ~/.bashrc
conda activate env

cd /tudelft.net/staff-umbrella/SATAYanalysis/Nina/Thesis/SGD_API

srun python yeast_genes.py