#!/bin/bash
#SBATCH -J NorESM2_LM_parallel
#SBATCH --output=logs/NorESM2_LM.out
#SBATCH --error=logs/NorESM2_LM.err
#SBATCH --array=0-4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00

set -euo pipefail

# Ensure conda is available in non-interactive shells
source /scratch/gpfs/GEOCLIM/LRGROUP/db9274/Anaconda/etc/profile.d/conda.sh
conda activate /scratch/gpfs/GEOCLIM/LRGROUP/db9274/Anaconda/envs/xmip

ALG=${ALG:-NN}
MODEL_I=${MODEL_I:-12}
ENS_J=${ENS_J:-0}
BASIN_K=${SLURM_ARRAY_TASK_ID}

cd /scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2024_ML_algorithm/ML4O2_Dhruv_potden/automated_codes

echo "Starting basin ${BASIN_K}"
echo "Step 1: train"
python3 Training.py ${ALG} ${MODEL_I} ${ENS_J} ${BASIN_K}

echo "Step 2: evaluate"
python3 Evaluating.py ${ALG} ${MODEL_I} ${ENS_J} ${BASIN_K}

echo "Step 3: map oxygen product"
python3 Projecting.py ${ALG} ${MODEL_I} ${ENS_J} ${BASIN_K}

echo "Finished basin ${BASIN_K}"