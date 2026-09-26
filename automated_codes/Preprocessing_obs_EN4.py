import os
import gc
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import gsw
import sys

sys.path.append('/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Python_functions/')

import regridding_operations
import fourier_transform
import basin_masks

training_model = sys.argv[1]

start_time, end_time = '1965-01-01', '2021-12-30'

# ----------------------------
# Paths
# ----------------------------
save_dir = '/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2024_ML_algorithm/ML4O2_Dhruv_potden/NETCDF/OSD_CTD_Argo/'
subsampled_dir = '/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2022_OI_algorithm/Preprocessing_RB23/Monthly_subsampled_fields_1965_2021/o2/'
out_nc_dir = '/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2024_ML_algorithm/ML4O2_Dhruv_potden/NPZ/'
en4_dir = '/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Observational_data/EN4/'
model_name = 'Observations_EN4'

# ----------------------------
# Standard grid
# ----------------------------
Nlev = 102
zstd = np.array([
    0.00e+00, 5.00e+00, 1.00e+01, 1.50e+01, 2.00e+01, 2.50e+01,
    3.00e+01, 3.50e+01, 4.00e+01, 4.50e+01, 5.00e+01, 5.50e+01,
    6.00e+01, 6.50e+01, 7.00e+01, 7.50e+01, 8.00e+01, 8.50e+01,
    9.00e+01, 9.50e+01, 1.00e+02, 1.25e+02, 1.50e+02, 1.75e+02,
    2.00e+02, 2.25e+02, 2.50e+02, 2.75e+02, 3.00e+02, 3.25e+02,
    3.50e+02, 3.75e+02, 4.00e+02, 4.25e+02, 4.50e+02, 4.75e+02,
    5.00e+02, 5.50e+02, 6.00e+02, 6.50e+02, 7.00e+02, 7.50e+02,
    8.00e+02, 8.50e+02, 9.00e+02, 9.50e+02, 1.00e+03, 1.05e+03,
    1.10e+03, 1.15e+03, 1.20e+03, 1.25e+03, 1.30e+03, 1.35e+03,
    1.40e+03, 1.45e+03, 1.50e+03, 1.55e+03, 1.60e+03, 1.65e+03,
    1.70e+03, 1.75e+03, 1.80e+03, 1.85e+03, 1.90e+03, 1.95e+03,
    2.00e+03, 2.10e+03, 2.20e+03, 2.30e+03, 2.40e+03, 2.50e+03,
    2.60e+03, 2.70e+03, 2.80e+03, 2.90e+03, 3.00e+03, 3.10e+03,
    3.20e+03, 3.30e+03, 3.40e+03, 3.50e+03, 3.60e+03, 3.70e+03,
    3.80e+03, 3.90e+03, 4.00e+03, 4.10e+03, 4.20e+03, 4.30e+03,
    4.40e+03, 4.50e+03, 4.60e+03, 4.70e+03, 4.80e+03, 4.90e+03,
    5.00e+03, 5.10e+03, 5.20e+03, 5.30e+03, 5.40e+03, 5.50e+03
], dtype="float32")
zc = zstd[:Nlev].astype("float32")

xc = (np.arange(0, 360, 1) + 0.5).astype("float32")
yc = (np.arange(-90, 90, 1) + 0.5).astype("float32")

# ----------------------------
# Metadata
# ----------------------------

basin_name = ['Atlantic', 'Pacific', 'Indian', 'Southern', 'Arctic']
basin_nums = [1, 2, 3, 10, 11]
basin_idx = [0, 1, 2, 3, 4]

os.makedirs(os.path.join(out_nc_dir, "vectors"), exist_ok=True)
os.makedirs(os.path.join(out_nc_dir, "params"), exist_ok=True)
os.makedirs(os.path.join(out_nc_dir, "folds"), exist_ok=True)
os.makedirs(os.path.join(out_nc_dir, model_name), exist_ok=True)

out_np_dir = '/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2024_ML_algorithm/ML4O2_Dhruv_potden/NPZ/'
model_dir = os.path.join(out_np_dir, model_name)

# --------------------------------------------------
# Helpers
# --------------------------------------------------
def fix_lon_0_360(da: xr.DataArray, lon_name: str = "x") -> xr.DataArray:
    lon = da[lon_name]
    lon_fixed = (lon % 360)
    return da.assign_coords({lon_name: lon_fixed}).sortby(lon_name)

def get_cv_year_blocks():
    return [(1965, 1975), (1976, 1986), (1987, 1997), (1998, 2008), (2009, 2021)]

def potential_density_from_pt(salt, theta):
    """Potential density anomaly sigma0 from practical salinity and potential temperature."""
    lev2, lat2 = xr.broadcast(theta["lev"], theta["y"])
    pressure = xr.apply_ufunc(gsw.p_from_z, -lev2, lat2)
    pressure, lon, lat = xr.broadcast(pressure, theta["x"], theta["y"])
    absolute_salinity = xr.apply_ufunc(
        gsw.SA_from_SP,
        salt,
        pressure,
        lon,
        lat,
        dask="allowed",
    )
    conservative_temp = xr.apply_ufunc(
        gsw.CT_from_pt,
        absolute_salinity,
        theta,
        dask="allowed",
    )
    sigma0 = xr.apply_ufunc(
        gsw.density.sigma0,
        absolute_salinity,
        conservative_temp,
        dask="allowed",
    )
    return sigma0.rename("sigma0")

def vectors_from_masked_fast(o2_sub, t_sub, s_sub, sigma0_sub):
    needed_dims = ("time", "lev", "y", "x")

    # one combined validity mask
    valid4 = np.isfinite(o2_sub) & np.isfinite(t_sub) & np.isfinite(s_sub) & np.isfinite(sigma0_sub)

    # stack only the mask first, then keep valid samples
    valid1 = valid4.stack(sample=needed_dims)
    valid1 = valid1.where(valid1, drop=True).compute()

    n = valid1.sizes.get("sample", 0)
    if n == 0:
        raise ValueError("No valid samples found.")

    sel = {"sample": valid1.sample}

    o2_1d = o2_sub.stack(sample=needed_dims).sel(sel)
    t_1d  = t_sub.stack(sample=needed_dims).sel(sel)
    s_1d  = s_sub.stack(sample=needed_dims).sel(sel)
    sigma0_1d = sigma0_sub.stack(sample=needed_dims).sel(sel)

    time_1d = o2_1d["time"]
    year  = time_1d.dt.year.astype("int32")
    month = time_1d.dt.month.astype("int32")
    tmon  = ((year - 1965) * 12 + (month - 1)).astype("int32")

    y_o2 = np.asarray(o2_1d.astype("float32").values)
    v_t  = np.asarray(t_1d.astype("float32").values)
    v_s  = np.asarray(s_1d.astype("float32").values)
    v_sigma0 = np.asarray(sigma0_1d.astype("float32").values)
    lon  = np.asarray(o2_1d["x"].astype("float32").values)
    lat  = np.asarray(o2_1d["y"].astype("float32").values)
    dep  = np.asarray(o2_1d["lev"].astype("float32").values)
    tmon = np.asarray(tmon.values, dtype="int32")

    return y_o2, v_t, v_s, v_sigma0, lon, lat, dep, tmon

# ----------------------------
# Read data
# ----------------------------
basin_mask = xr.open_dataset('/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2022_OI_algorithm/optint_wod_o2/basin_mask_01.nc')['basin_mask']
basin_mask = basin_mask.rename({'lon': 'x', 'lat': 'y', 'depth': 'lev'}).isel(lev = 0)
basin_mask = fix_lon_0_360(basin_mask)

oxygen_file_path = os.path.join(subsampled_dir, 'Oxygen_obs_OSD_CTD_Argo.nc')
oxygen_s = xr.open_dataset(oxygen_file_path)['o2'].sel(time = slice(start_time, end_time))
oxygen_s = fix_lon_0_360(oxygen_s)
oxygen_s = oxygen_s.where(oxygen_s >= 0)
oxygen_s = oxygen_s.resample(time = '1YS').mean('time').load();

temperature_file_path = os.path.join(en4_dir, 'thetao/thetao_1x1bin_EN4_g10_1965-2021_annual.nc')
thetao_s = xr.open_dataset(temperature_file_path)['thetao'].rename({'depth': 'lev', 'lat': 'y', 'lon': 'x'}).sel(time = slice(start_time, end_time))
thetao_s = fix_lon_0_360(thetao_s)
thetao_s['time'] = oxygen_s['time']
thetao_s = thetao_s.interp_like(oxygen_s)
thetao_s = thetao_s.where(oxygen_s >=0)

salinity_file_path = os.path.join(en4_dir, 'so/so_1x1bin_EN4_g10_1965-2021_annual.nc')
so_s = xr.open_dataset(salinity_file_path)['so'].rename({'depth': 'lev', 'lat': 'y', 'lon': 'x'}).sel(time = slice(start_time, end_time))
so_s = fix_lon_0_360(so_s)
so_s['time'] = oxygen_s['time']
so_s = so_s.interp_like(oxygen_s)
so_s = so_s.where(oxygen_s >=0)

thetao_s = thetao_s.load();
so_s = so_s.load();
sigma0_s = potential_density_from_pt(so_s, thetao_s).load()

print('Oxygen, temperature, salinity, and potential density loaded!')

for l, k in enumerate(basin_nums):
    basin = basin_name[l]
    print(f"Processing basin: {basin}")

    sector = (basin_mask == (k))
    
    oxygen_sub = oxygen_s.where(sector)
    thetao_sub = thetao_s.where(sector)
    so_sub     = so_s.where(sector)
    sigma0_sub = sigma0_s.where(sector)

    y, T, S, sigma0, lon, lat, dep, tmon = vectors_from_masked_fast(oxygen_sub, thetao_sub, so_sub, sigma0_sub)
    year = (tmon // 12 + 1965).astype("int32")
    
    # save raw vectors once
    vecfile = os.path.join(out_nc_dir, model_name, f'training_vectors_{basin}.npz')
    np.savez_compressed(vecfile, y = y, T = T, S = S, sigma0 = sigma0, lon = lon, lat = lat, dep = dep, tmon = tmon, year = year)

    # save fold indices
    fold_data = {}
    year_blocks = get_cv_year_blocks()
    
    for fold, (y0, y1) in enumerate(year_blocks):
        test_mask = (year >= y0) & (year <= y1)
        train_idx = np.where(~test_mask)[0].astype("int32")
        test_idx  = np.where(test_mask)[0].astype("int32")

        fold_data[f'train_idx_fold{fold}'] = train_idx
        fold_data[f'test_idx_fold{fold}']  = test_idx
        fold_data[f'test_year_start_fold{fold}'] = np.array([y0], dtype="int32")
        fold_data[f'test_year_end_fold{fold}']   = np.array([y1], dtype="int32")

        print(f'  Fold {fold}: test years {y0}-{y1}, 'f'ntrain={train_idx.size}, ntest={test_idx.size}')

    np.savez(os.path.join(model_dir, f'cv_folds_{basin}.npz'), **fold_data)

    del oxygen_sub, thetao_sub, so_sub, sigma0_sub
    del y, T, S, sigma0, lon, lat, dep, tmon, year, fold_data
    gc.collect()
