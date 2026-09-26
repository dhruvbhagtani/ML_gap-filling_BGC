import os
import gc
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# =========================================================
# Inputs
# =========================================================
# Example:
# python Evaluating.py RF 1 0 2
#
# argv[1] = algorithm: "RF" or "NN"
# argv[2] = model index i
# argv[3] = ensemble index j
# argv[4] = basin index k

alg = sys.argv[1]
i   = int(sys.argv[2])
j   = int(sys.argv[3])
k   = int(sys.argv[4])


# =========================================================
# Paths
# =========================================================
out_np_dir = '/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2024_ML_algorithm/ML4O2_Dhruv_potden/NPZ/'
dirfin     = '/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2024_ML_algorithm/ML4O2_Dhruv_potden/FINAL/'


# =========================================================
# Metadata
# =========================================================
expt_name = [
    'GFDL_CM4', 'GFDL_ESM4', 'CanESM5', 'CanESM5_CanOE', 'ACCESS_ESM1.5',
    'MIROC_ES2L', 'UKESM1.0LL', 'IPSL_CM6A_LR', 'MRI_ESM2_0',
    'MPI_ESM1_2_LR', 'MPI_ESM1_2_HR', 'CNRM_ESM2_1', 'NorESM2_LM', 'NorESM2_MM'
]

ensembles = [
    ['r1i1p1f1'],
    ['r1i1p1f1'],
    ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1', 'r4i1p1f1', 'r5i1p1f1'],
    ['r1i1p2f1', 'r2i1p2f1', 'r3i1p2f1'],
    ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1'],
    ['r1i1p1f2', 'r2i1p1f2', 'r3i1p1f2', 'r4i1p1f2', 'r5i1p1f2'],
    ['r1i1p1f2', 'r2i1p1f2', 'r3i1p1f2'],
    ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1', 'r4i1p1f1', 'r5i1p1f1'],
    ['r1i2p1f1'],
    ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1', 'r4i1p1f1', 'r5i1p1f1'],
    ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1', 'r4i1p1f1', 'r5i1p1f1'],
    ['r1i1p1f2', 'r2i1p1f2', 'r3i1p1f2', 'r4i1p1f2', 'r5i1p1f2'],
    ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1'],
    ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1']
]

basin_name = ['Atlantic', 'Pacific', 'Indian', 'Southern', 'Arctic']


# =========================================================
# Hyperparameters
# =========================================================
RF_parameters = {"n_estimators": [50, 100, 200, 400, 600], "min_samples_split": [2, 5, 10]}

NN_parameters = {
    "hidden_layer_sizes": [
        [10, 10, 10, 10],
        [20, 20, 20, 20],
        [40, 40, 40, 40],
        [60, 60, 60, 60],
        [60, 40, 20, 10],
        [20, 20, 20, 20, 20, 20, 10, 5]
    ],
    "alpha": [0.001, 0.01, 0.1]
}


# =========================================================
# Helpers
# =========================================================
def fast_r2_corr(est, test):
    e = est - est.mean()
    t = test - test.mean()
    denom = np.sqrt((e * e).sum() * (t * t).sum())
    if denom == 0:
        return np.nan
    cc = (e * t).sum() / denom
    return float(cc * cc)


def standardize_all_data(X_raw, y_raw):
    """
    X_raw shape: (n_features, n_samples)
    y_raw shape: (n_samples,)
    """
    Xm = np.mean(X_raw, axis=1, dtype=np.float64).astype("float32")
    Xstd = np.std(X_raw, axis=1, dtype=np.float64).astype("float32")
    Xstd = np.where(Xstd == 0, 1.0, Xstd).astype("float32")

    ym = np.float32(np.mean(y_raw, dtype=np.float64))
    ystd = np.float32(np.std(y_raw, dtype=np.float64))
    if ystd == 0:
        ystd = np.float32(1.0)

    X = ((X_raw.T - Xm) / Xstd).astype("float32")
    y = ((y_raw - ym) / ystd).astype("float32")

    return X, y, Xm, Xstd, ym, ystd


def build_raw_features(S, T, sigma0, lon, lat, dep, year):
    """
    Annual-mean predictors. No month predictor.
    Shape returned: (n_features, n_samples)
    """
    X_raw = np.vstack([
        S,
        T,
        sigma0,
        lon,
        lat,
        dep,
        year
    ]).astype("float32")
    return X_raw


def fit_model(alg, parm1, parm2, X, y):
    if alg == "RF":
        regr = RandomForestRegressor(
            n_jobs=-1,
            n_estimators=RF_parameters["n_estimators"][parm1],
            min_samples_split=RF_parameters["min_samples_split"][parm2],
            min_samples_leaf=5,
            max_features="sqrt",
            random_state=0,
            verbose=False
        )
    elif alg == "NN":
        regr = MLPRegressor(
            max_iter=300,
            solver='adam',
            early_stopping=True,
            n_iter_no_change=5,
            validation_fraction=0.1,
            hidden_layer_sizes=tuple(NN_parameters["hidden_layer_sizes"][parm1]),
            alpha=NN_parameters["alpha"][parm2],
            random_state=0,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown algorithm: {alg}")

    regr.fit(X, y)
    return regr


def choose_best_hyperparameters(metrics, criterion="rmse"):
    """
    metrics contains:
      R2   shape (5, nparm2, nparm1)
      RMSE shape (5, nparm2, nparm1)

    Returns best parm1, parm2.
    """
    R2 = metrics["R2"]
    RMSE = metrics["RMSE"]

    mean_R2 = np.nanmean(R2, axis=0)      # shape (nparm2, nparm1)
    mean_RMSE = np.nanmean(RMSE, axis=0)  # shape (nparm2, nparm1)

    if criterion.lower() == "rmse":
        flat_idx = np.nanargmin(mean_RMSE)
        parm2, parm1 = np.unravel_index(flat_idx, mean_RMSE.shape)
    elif criterion.lower() == "r2":
        flat_idx = np.nanargmax(mean_R2)
        parm2, parm1 = np.unravel_index(flat_idx, mean_R2.shape)
    else:
        raise ValueError("criterion must be 'rmse' or 'r2'")

    return parm1, parm2, mean_R2, mean_RMSE


# =========================================================
# Load files
# =========================================================
model = expt_name[i]
ens   = ensembles[i][j]
basin = basin_name[k]

mdir = os.path.join(out_np_dir, model)
fdir = os.path.join(dirfin, model)
os.makedirs(fdir, exist_ok=True)

vecfile = os.path.join(mdir, f"training_vectors_{basin}_{ens}.npz")
metfile = os.path.join(mdir, f"CV_metrics_{alg}_{basin}_{ens}.npz")

if not os.path.exists(vecfile):
    raise FileNotFoundError(f"Missing vector file: {vecfile}")
if not os.path.exists(metfile):
    raise FileNotFoundError(f"Missing metrics file: {metfile}")

vec = np.load(vecfile)
metrics = np.load(metfile)

y    = vec["y"].astype("float32")
T    = vec["T"].astype("float32")
S    = vec["S"].astype("float32")
sigma0 = vec["sigma0"].astype("float32")
lon  = vec["lon"].astype("float32")
lat  = vec["lat"].astype("float32")
dep  = vec["dep"].astype("float32")
year = vec["year"].astype("float32")

print(f"Loaded model={model}, ensemble={ens}, basin={basin}")
print(f"Samples: {y.size}")


# =========================================================
# Select best hyperparameters from CV
# =========================================================
parm1_best, parm2_best, mean_R2, mean_RMSE = choose_best_hyperparameters(
    metrics,
    criterion="rmse"
)

print(f"Best hyperparameters for {alg}: parm1={parm1_best}, parm2={parm2_best}")

if alg == "RF":
    best_desc = {
        "n_estimators": RF_parameters["n_estimators"][parm1_best],
        "min_samples_split": RF_parameters["min_samples_split"][parm2_best]
    }
else:
    best_desc = {
        "hidden_layer_sizes": NN_parameters["hidden_layer_sizes"][parm1_best],
        "alpha": NN_parameters["alpha"][parm2_best]
    }

print("Best parameter values:", best_desc)


# =========================================================
# Train final model on all samples
# =========================================================
X_raw = build_raw_features(S, T, sigma0, lon, lat, dep, year)
X_all, y_all, Xm, Xstd, ym, ystd = standardize_all_data(X_raw, y)

regr = fit_model(alg, parm1_best, parm2_best, X_all, y_all)

y_est_std = regr.predict(X_all).astype("float32")

# In physical units
o2_true = y_all * ystd + ym
o2_est  = y_est_std * ystd + ym

r2_all = fast_r2_corr(y_est_std, y_all)
rmse_all = float(np.sqrt(np.mean((o2_true - o2_est) ** 2)))

print(f"Final all-data fitted R2 = {r2_all:.4f}")
print(f"Final all-data fitted RMSE = {rmse_all:.4f}")


# =========================================================
# Save fitted predictions
# =========================================================
outpred = os.path.join(
    fdir,
    f"o2fit_pred_{alg}_{basin}_{model}_{ens}.npz"
)

np.savez(
    outpred,
    year=year.astype("int32"),
    lon=lon,
    lat=lat,
    dep=dep,
    T=T,
    S=S,
    sigma0=sigma0,
    truth=o2_true.astype("float32"),
    est=o2_est.astype("float32"),
    parm1_best=np.array([parm1_best], dtype="int32"),
    parm2_best=np.array([parm2_best], dtype="int32"),
    r2_all=np.array([r2_all], dtype="float32"),
    rmse_all=np.array([rmse_all], dtype="float32")
)

print(f"Saved fitted predictions: {outpred}")


# =========================================================
# Save final model
# =========================================================
outmodel = os.path.join(
    fdir,
    f"algorithm_{alg}_{basin}_{model}_{ens}.joblib"
)
joblib.dump(regr, outmodel)
print(f"Saved final model: {outmodel}")


# =========================================================
# Save final scaling parameters
# =========================================================
outparams = os.path.join(
    fdir,
    f"ML_params_{alg}_{basin}_{model}_{ens}.npz"
)

np.savez(
    outparams,
    Xm=Xm,
    Xstd=Xstd,
    ym=np.array([ym], dtype="float32"),
    ystd=np.array([ystd], dtype="float32"),
    parm1_best=np.array([parm1_best], dtype="int32"),
    parm2_best=np.array([parm2_best], dtype="int32")
)

print(f"Saved final scaling parameters: {outparams}")


# =========================================================
# Save evaluation summary
# =========================================================
outsummary = os.path.join(
    fdir,
    f"evaluation_summary_{alg}_{basin}_{model}_{ens}.npz"
)

np.savez(
    outsummary,
    R2_folds=metrics["R2"],
    RMSE_folds=metrics["RMSE"],
    mean_R2=mean_R2,
    mean_RMSE=mean_RMSE,
    parm1_best=np.array([parm1_best], dtype="int32"),
    parm2_best=np.array([parm2_best], dtype="int32"),
    r2_all=np.array([r2_all], dtype="float32"),
    rmse_all=np.array([rmse_all], dtype="float32")
)

print(f"Saved evaluation summary: {outsummary}")


# =========================================================
# Cleanup
# =========================================================
del vec, metrics, X_raw, X_all, y_all, y_est_std, o2_true, o2_est
del Xm, Xstd, ym, ystd, regr
gc.collect()
