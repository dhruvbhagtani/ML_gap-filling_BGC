#!/bin/bash
#SBATCH -J Observations_parallel
#SBATCH --output=logs/Observations.out
#SBATCH --error=logs/Observations.err
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --mem=976G
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00

set -euo pipefail

# Ensure conda is available in non-interactive shells
source /scratch/gpfs/GEOCLIM/LRGROUP/db9274/Anaconda/etc/profile.d/conda.sh
conda activate /scratch/gpfs/GEOCLIM/LRGROUP/db9274/Anaconda/envs/xmip

ALG=${ALG:-NN}
BASIN_K=${SLURM_ARRAY_TASK_ID}

cd /scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2024_ML_algorithm/ML4O2_Dhruv_potden/automated_codes

echo "Starting basin ${BASIN_K}"
echo "Step 1: train"
python3 Training_obs.py ${ALG} ${BASIN_K}

echo "Step 2: evaluate"
python3 Evaluating_obs.py ${ALG} ${BASIN_K}

echo "Step 3: map oxygen product"
python3 Projecting_obs.py ${ALG} ${BASIN_K}

echo "Finished basin ${BASIN_K}"
