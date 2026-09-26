#!/bin/bash
#SBATCH -J Observations_EN4
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=976G
#SBATCH -t 10:00:00
#SBATCH --output=LOG_HIST/slurm-%j.out
#SBATCH --account=lrgroup

set -euo pipefail

# Ensure conda is available in non-interactive shells
source /scratch/gpfs/GEOCLIM/LRGROUP/db9274/Anaconda/etc/profile.d/conda.sh
conda activate /scratch/gpfs/GEOCLIM/LRGROUP/db9274/Anaconda/envs/xmip

cd /scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2024_ML_algorithm/ML4O2_Dhruv_potden/automated_codes

python3 Preprocessing_obs_EN4.py RF
