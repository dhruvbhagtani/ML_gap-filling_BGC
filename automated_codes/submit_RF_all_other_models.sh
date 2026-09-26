#!/bin/bash
set -euo pipefail

# Submit RF preprocessing and basin-array mapping jobs for all CMIP6 models
# except GFDL_ESM4, which has already been run in this potden workflow.

preprocessing_scripts=(
  GFDL_CM4_preprocessing.sh
  CanESM5_preprocessing.sh
  CanESM5_CanOE_preprocessing.sh
  ACCESS_ESM1_5_preprocessing.sh
  MIROC_ES2L_preprocessing.sh
  UKESM1.0LL_preprocessing.sh
  IPSL_CM6_preprocessing.sh
  MRI_ESM2_0_preprocessing.sh
  MPI_ESM1_2_LR_preprocessing.sh
  MPI_ESM1_2_HR_preprocessing.sh
  CNRM_ESM2_1_preprocessing.sh
  NorESM2_LM_preprocessing.sh
  NorESM2_MM_preprocessing.sh
)

parallel_scripts=(
  GFDL_CM4_parallel.sh
  CanESM5_parallel.sh
  CanESM5_CanOE_parallel.sh
  ACCESS_ESM1_5_parallel.sh
  MIROC_ES2L_parallel.sh
  UKESM1.0LL_parallel.sh
  IPSL_CM6A_LR_parallel.sh
  MRI_ESM2_0_parallel.sh
  MPI_ESM1_2_LR_parallel.sh
  MPI_ESM1_2_HR_parallel.sh
  CNRM_ESM2_1_parallel.sh
  NorESM2_LM_parallel.sh
  NorESM2_MM_parallel.sh
)

for idx in "${!preprocessing_scripts[@]}"; do
  pre_job=$(sbatch --parsable "${preprocessing_scripts[$idx]}")
  sbatch --dependency=afterok:"${pre_job}" --export=ALL,ALG=RF "${parallel_scripts[$idx]}"
done
