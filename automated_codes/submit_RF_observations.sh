#!/bin/bash
set -euo pipefail

pre_job=$(sbatch --parsable Observations_preprocessing.sh)
sbatch --dependency=afterok:"${pre_job}" --export=ALL,ALG=RF Observations_parallel.sh
