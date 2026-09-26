#!/bin/bash 
#SBATCH -J MPI_ESM1_2_HR    # job name
#SBATCH -N 1                    # node number 
#SBATCH --ntasks-per-node=1                  # how many tasks per node
#SBATCH --mem=976G
#SBATCH -t 1:00:00               # requested time duration for job
#SBATCH --output=LOG_HIST/slurm-%j.out  # path to save slurm log file
#SBATCH --account=lrgroup

# Ensure conda is available in non-interactive shells
source /scratch/gpfs/GEOCLIM/LRGROUP/db9274/Anaconda/etc/profile.d/conda.sh
conda activate /scratch/gpfs/GEOCLIM/LRGROUP/db9274/Anaconda/envs/xmip

python3 Preprocessing.py 'RF' '10' '0'    ## Form vectors and training data