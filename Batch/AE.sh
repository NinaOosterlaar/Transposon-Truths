#!/bin/bash
#SBATCH --job-name=AE
#SBATCH --partition=general,insy
#SBATCH --account=ewi-insy-prb
#SBATCH --qos=long
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=128G
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

srun python AE/Autoencoder.py --model AE