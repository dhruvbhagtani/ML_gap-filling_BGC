import os
import gc
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np

from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor


# =========================================================
# Inputs
# =========================================================
# Example usage:
# python Training.py RF 1 0 2
#
# argv[1] = algorithm: "RF" or "NN"
# argv[2] = model index i
# argv[3] = ensemble index j
# argv[4] = basin index k  (0=Atlantic, 1=Pacific, 2=Indian, 3=Southern, 4=Arctic)

alg = sys.argv[1]                 # "RF" or "NN"
i   = int(sys.argv[2])            # model index
j   = int(sys.argv[3])            # ensemble index
k   = int(sys.argv[4])            # basin index

# =========================================================
# Paths
# =========================================================
out_np_dir = '/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2024_ML_algorithm/ML4O2_Dhruv_potden/NPZ/'

# =========================================================
# Metadata
# =========================================================
expt_name = ['GFDL_CM4', 'GFDL_ESM4', 'CanESM5', 'CanESM5_CanOE', 'ACCESS_ESM1.5', 'MIROC_ES2L', 'UKESM1.0LL', 
             'IPSL_CM6A_LR', 'MRI_ESM2_0', 'MPI_ESM1_2_LR', 'MPI_ESM1_2_HR', 'CNRM_ESM2_1', 'NorESM2_LM', 'NorESM2_MM']

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
    ['r1i1p1f1', 'r2i1p1f1', 'r3i1p1f1']]

basin_name = ['Atlantic', 'Pacific', 'Indian', 'Southern', 'Arctic']

# =========================================================
# Hyperparameters
# =========================================================
#RF_parameters = {"n_estimators": [50, 100, 200, 500, 800, 1000], "min_samples_split": [2, 5, 10]}
RF_parameters = {"n_estimators": [50, 100, 200, 400, 600], "min_samples_split": [2, 5, 10]}

NN_parameters = {"hidden_layer_sizes": [[10, 10, 10, 10], [20, 20, 20, 20],
        [40, 40, 40, 40], [60, 60, 60, 60], [60, 40, 20, 10], [20, 20, 20, 20, 20, 20, 10, 5]], "alpha": [0.001, 0.01, 0.1]}

# =========================================================
# Helpers
# =========================================================
def fast_r2_corr(est, test):
    """
    R^2 based on squared correlation, matching your earlier setup.
    """
    e = est - est.mean()
    t = test - test.mean()
    denom = np.sqrt((e * e).sum() * (t * t).sum())
    if denom == 0:
        return np.nan
    cc = (e * t).sum() / denom
    return float(cc * cc)

def standardize_fold(X_train_raw, X_test_raw, y_train_raw, y_test_raw):
    """
    Standardize using training-fold statistics only.

    Inputs
    ------
    X_train_raw : (n_features, n_train)
    X_test_raw  : (n_features, n_test)
    y_train_raw : (n_train,)
    y_test_raw  : (n_test,)

    Returns
    -------
    X_train : (n_train, n_features)
    X_test  : (n_test, n_features)
    y_train : (n_train,)
    y_test  : (n_test,)
    Xm, Xstd, ym, ystd
    """
    Xm = np.mean(X_train_raw, axis=1, dtype=np.float64).astype("float32")
    Xstd = np.std(X_train_raw, axis=1, dtype=np.float64).astype("float32")
    Xstd = np.where(Xstd == 0, 1.0, Xstd).astype("float32")

    ym = np.float32(np.mean(y_train_raw, dtype=np.float64))
    ystd = np.float32(np.std(y_train_raw, dtype=np.float64))
    if ystd == 0:
        ystd = np.float32(1.0)

    X_train = ((X_train_raw.T - Xm) / Xstd).astype("float32")
    X_test  = ((X_test_raw.T  - Xm) / Xstd).astype("float32")

    y_train = ((y_train_raw - ym) / ystd).astype("float32")
    y_test  = ((y_test_raw  - ym) / ystd).astype("float32")

    return X_train, X_test, y_train, y_test, Xm, Xstd, ym, ystd

def build_raw_features(S, T, sigma0, lon, lat, dep, year, idx):
    """
    Build feature array with shape (n_features, n_samples)
    using annual-mean predictors.
    """
    X_raw = np.vstack([S[idx], T[idx], sigma0[idx], lon[idx], lat[idx], dep[idx], year[idx]]).astype("float32")
    return X_raw

def fit_model(alg, parm1, parm2, X_train, y_train):
    """
    Fit RF or NN model for one hyperparameter combination.
    """
    if alg == "RF":
        regr = RandomForestRegressor(n_jobs=-1, n_estimators=RF_parameters["n_estimators"][parm1],
            min_samples_split=RF_parameters["min_samples_split"][parm2],
            min_samples_leaf=5, max_features="sqrt",
            random_state=0, verbose=False)

    elif alg == "NN":
        regr = MLPRegressor(max_iter=300, solver='adam', early_stopping=True,
            n_iter_no_change=5, validation_fraction=0.1,
            hidden_layer_sizes=tuple(NN_parameters["hidden_layer_sizes"][parm1]),
            alpha=NN_parameters["alpha"][parm2], random_state=0, verbose=True)
    else:
        raise ValueError(f"Unknown algorithm: {alg}")

    regr.fit(X_train, y_train)
    return regr

def train_one_fold(fold, parm1, parm2, y, T, S, sigma0, lon, lat, dep, year, fld, alg):
    """
    Train/evaluate one saved fold.
    """
    train_idx = fld[f"train_idx_fold{fold}"]
    test_idx  = fld[f"test_idx_fold{fold}"]

    if train_idx.size == 0 or test_idx.size == 0:
        return np.nan, np.nan

    X_train_raw = build_raw_features(S, T, sigma0, lon, lat, dep, year, train_idx)
    X_test_raw  = build_raw_features(S, T, sigma0, lon, lat, dep, year, test_idx)

    y_train_raw = y[train_idx].astype("float32")
    y_test_raw  = y[test_idx].astype("float32")

    X_train, X_test, y_train, y_test, Xm, Xstd, ym, ystd = standardize_fold(
        X_train_raw, X_test_raw, y_train_raw, y_test_raw)

    regr = fit_model(alg, parm1, parm2, X_train, y_train)

    y_est = regr.predict(X_test).astype("float32")

    r2 = fast_r2_corr(y_est, y_test)
    rmse = float(np.sqrt(np.mean((y_est - y_test) ** 2)))

    # cleanup
    del X_train_raw, X_test_raw, y_train_raw, y_test_raw
    del X_train, X_test, y_train, y_test
    del Xm, Xstd, ym, ystd, regr, y_est
    gc.collect()

    return r2, rmse

# =========================================================
# Load data
# =========================================================
model = expt_name[i]
ens   = ensembles[i][j]
basin = basin_name[k]

mdir = os.path.join(out_np_dir, model)
os.makedirs(mdir, exist_ok=True)

vecfile = os.path.join(mdir, f'training_vectors_{basin}_{ens}.npz')
foldfile = os.path.join(mdir, f'cv_folds_{basin}_{ens}.npz')

if not os.path.exists(vecfile):
    raise FileNotFoundError(f"Missing vector file: {vecfile}")
if not os.path.exists(foldfile):
    raise FileNotFoundError(f"Missing fold file: {foldfile}")

vec = np.load(vecfile)
fld = np.load(foldfile)

y    = vec["y"].astype("float32")
T    = vec["T"].astype("float32")
S    = vec["S"].astype("float32")
sigma0 = vec["sigma0"].astype("float32")
lon  = vec["lon"].astype("float32")
lat  = vec["lat"].astype("float32")
dep  = vec["dep"].astype("float32")
year = vec["year"].astype("float32")

print(f"Loaded model={model}, ensemble={ens}, basin={basin}")
print(f"Number of samples = {y.size}")

# =========================================================
# Grid search over 5 CV folds
# =========================================================
if alg == "RF":
    nparm1 = len(RF_parameters["n_estimators"])
    nparm2 = len(RF_parameters["min_samples_split"])
elif alg == "NN":
    nparm1 = len(NN_parameters["hidden_layer_sizes"])
    nparm2 = len(NN_parameters["alpha"])
else:
    raise ValueError(f"Unknown algorithm: {alg}")

R2   = np.full((5, nparm2, nparm1), np.nan, dtype="float64")
RMSE = np.full((5, nparm2, nparm1), np.nan, dtype="float64")

for parm1 in range(nparm1):
    for parm2 in range(nparm2):
        print(f"\nHyperparameters: parm1={parm1}, parm2={parm2}")
        for fold in range(5):
            print(f"  Fold {fold}")
            r2, rmse = train_one_fold(fold=fold, parm1=parm1, parm2=parm2, y=y, T=T,
                S=S, sigma0=sigma0, lon=lon, lat=lat, dep=dep, year=year, fld=fld, alg=alg)
            R2[fold, parm2, parm1] = r2
            RMSE[fold, parm2, parm1] = rmse
            print(f"    R2={r2:.4f}, RMSE={rmse:.4f}")

# =========================================================
# Save CV results
# =========================================================
outfile = os.path.join(mdir, f'CV_metrics_{alg}_{basin}_{ens}.npz')
np.savez(outfile, R2=R2, RMSE=RMSE)

print(f"\nSaved CV metrics to: {outfile}")

# Optional summary
mean_R2 = np.nanmean(R2, axis=0)
mean_RMSE = np.nanmean(RMSE, axis=0)

print("\nMean CV R2 over folds:")
print(mean_R2)

print("\nMean CV RMSE over folds:")
print(mean_RMSE)
