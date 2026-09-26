# ML4O2 with potential density predictors

Scripts for reconstructing ocean oxygen fields using random forests (`RF`) and neural networks (`NN`). The workflows cover CMIP6 model experiments and observational reconstructions, including an EN4 variant.

This gap-filling framework is inspired by Ito et al. (2024), [*Mapping Dissolved Oxygen Concentrations by Combining Shipboard and Argo Observations Using Machine Learning Algorithms*](https://doi.org/10.1029/2024JH000272). This implementation includes enhancements and adaptations to the training procedure, reconstruction time periods, and historical data inputs.

The gap-filling framework is not limited to oxygen and can be extended to other biogeochemical tracers by adapting the target observations, predictor variables, preprocessing, and validation to the tracer of interest. The scripts currently included in this repository implement oxygen reconstruction.

## Workflow

1. **Preprocessing** forms annual means, computes potential density anomaly (`sigma0`), selects samples with finite oxygen and predictors, and saves vectors and five contiguous year blocks for cross-validation.
2. **Training** searches hyperparameters with blocked cross-validation. Predictors are salinity, temperature, potential density anomaly, longitude, latitude, depth, and year. Standardization uses training-fold statistics only.
3. **Evaluating** selects hyperparameters from cross-validation metrics, fits the selected model to all available training samples, and saves the fitted model and scaling parameters. Scores from this final fit are in-sample scores.
4. **Projecting** applies the fitted model and saved scaling parameters to gridded predictors and writes basin products.
5. **Projection_merging.ipynb** combines basin products and inspects predictions and parameters. **CMIP6_models_variability_compute.ipynb** contains further model analysis.

## Files

| Files in `automated_codes/` | Purpose |
| --- | --- |
| `Preprocessing.py`, `Training.py`, `Evaluating.py`, `Projecting.py` | CMIP6 workflow |
| Files ending in `_obs.py` | Observational workflow |
| Files ending in `_obs_EN4.py` | EN4 observational variant |
| `*_preprocessing.sh` | Slurm preprocessing jobs |
| `*_parallel.sh` and other submission scripts | Slurm training, fitting, and projection jobs |

`NPZ/` stores training vectors, fold indices, metrics, and model artifacts. `FINAL/` and `NETCDF/` contain gridded products or supporting data. These directories, cluster logs, and caches are excluded from Git because they are large generated or external files.

## Requirements and local setup

The scripts use NumPy, xarray, scikit-learn, GSW, joblib, and (in the CMIP6 projection script) xmip.

These scripts currently use absolute paths on Princeton's Tiger3 cluster. Before running elsewhere:

- Update input, output, basin-mask, and helper-module paths in the Python scripts and notebooks.
- Supply the input oxygen, temperature, salinity, and basin-mask datasets referenced by the selected workflow.
- Provide the external `regridding_operations`, `fourier_transform`, and `basin_masks` modules imported from `Analysis/Python_functions/` by preprocessing scripts.
- Update the Conda activation paths, working directories, account, resources, and log paths in Slurm scripts. Create their log directories before submission.

## Example commands

Run these from `automated_codes/` after completing the setup above. They illustrate the command-line interfaces; the full workflows require substantial data and compute resources.

```bash
# GFDL-ESM4 (model index 1), first ensemble (index 0), Atlantic (basin 0)
python Preprocessing.py RF 1 0
python Training.py RF 1 0 0
python Evaluating.py RF 1 0 0
python Projecting.py RF 1 0 0

# Observations, Atlantic
python Preprocessing_obs.py RF
python Training_obs.py RF 0
python Evaluating_obs.py RF 0
python Projecting_obs.py RF 0

# EN4 variant, Atlantic
python Preprocessing_obs_EN4.py RF
python Training_obs_EN4.py RF 0
python Evaluating_obs_EN4.py RF 0
python Projecting_obs_EN4.py RF 0
```

Use `NN` instead of `RF` for neural networks. Preprocessing accepts the algorithm argument but does not use it to construct the vectors. Model and ensemble indices follow the lists in the CMIP6 scripts. Basin indices are `0=Atlantic`, `1=Pacific`, `2=Indian`, `3=Southern`, and `4=Arctic`.

The CMIP6 preprocessing blocks are 1965–1976, 1977–1987, 1988–1998, 1999–2009, and 2010–2021. Consult each observational preprocessing script for its dates and fold definitions.
