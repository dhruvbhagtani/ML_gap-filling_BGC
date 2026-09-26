import os
import gc
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import xarray as xr
import joblib
import gsw

from xmip.preprocessing import (
    rename_cmip6,
    promote_empty_dims,
    broadcast_lonlat,
    replace_x_y_nominal_lat_lon,
    correct_lon,
    correct_coordinates,
    parse_lon_lat_bounds,
)


# =========================================================
# Inputs
# =========================================================
# Example:
# python Projecting.py RF 1 0 2
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
base_dir = '/scratch/gpfs/GEOCLIM/synda/data/CMIP6/CMIP/'
out_np_dir = '/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2024_ML_algorithm/ML4O2_Dhruv_potden/NPZ/'
dirfin = '/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2024_ML_algorithm/ML4O2_Dhruv_potden/FINAL/'


# =========================================================
# Metadata
# =========================================================
subdir_salt = ['Omon/so/gr/', 'Omon/so/gr/', 'Omon/so/gn/', 'Omon/so/gn/', 'Omon/so/gn/', 'Omon/so/gn/', 'Omon/so/gn/', 'Omon/so/gn/', 'Omon/so/gn/', 'Omon/so/gn/', 'Omon/so/gn/', 'Omon/so/gn/', 'Omon/so/gr/', 'Omon/so/gr/']
subdir_temp = ['Omon/thetao/gr/', 'Omon/thetao/gr/', 'Omon/thetao/gn/', 'Omon/thetao/gn/', 'Omon/thetao/gn/', 'Omon/thetao/gn/', 'Omon/thetao/gn/', 'Omon/thetao/gn/', 'Omon/thetao/gn/', 'Omon/thetao/gn/', 'Omon/thetao/gn/', 'Omon/thetao/gn/', 'Omon/thetao/gr/', 'Omon/thetao/gr/']

expt = ['NOAA-GFDL/GFDL-CM4/', 'NOAA-GFDL/GFDL-ESM4/', 'CCCma/CanESM5/', 'CCCma/CanESM5-CanOE/', 'CSIRO/ACCESS-ESM1-5/', 'MIROC/MIROC-ES2L/', 'MOHC/UKESM1-0-LL/', 'IPSL/IPSL-CM6A-LR/', 'MRI/MRI-ESM2-0/', 'MPI-M/MPI-ESM1-2-LR/', 'MPI-M/MPI-ESM1-2-HR/', 'CNRM-CERFACS/CNRM-ESM2-1/', 'NCC/NorESM2-LM/', 'NCC/NorESM2-MM/']
expt_name = ['GFDL_CM4', 'GFDL_ESM4', 'CanESM5', 'CanESM5_CanOE', 'ACCESS_ESM1.5', 'MIROC_ES2L', 'UKESM1.0LL', 'IPSL_CM6A_LR', 'MRI_ESM2_0', 'MPI_ESM1_2_LR', 'MPI_ESM1_2_HR', 'CNRM_ESM2_1', 'NorESM2_LM', 'NorESM2_MM']
scenario_name = 'historical'

ensembles = [['r1i1p1f1'],
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

version = [['v20180701'],
           ['v20190726'],
           ['v20190429', 'v20190429', 'v20190429', 'v20190429', 'v20190429'],
           ['v20190429', 'v20190429', 'v20190429'],
           ['v20191115', 'v20191128', 'v20191203'],
           ['v20190823', 'v20190823', 'v20190823', 'v20200318', 'v20200318'],
           ['v20190627', 'v20190708', 'v20190708'],
           ['v20180803', 'v20190305', 'v20180803', 'v20180803', 'v20180803'],
           ['v20210311'],
           ['v20190710', 'v20190710', 'v20190710', 'v20190710', 'v20190710'],
           ['v20190710', 'v20190710', 'v20190710', 'v20190710', 'v20190710'],
           ['v20181206', 'v20190125', 'v20190125', 'v20190125', 'v20190125'],
           ['v20190815', 'v20190920', 'v20190920'],
           ['v20191108', 'v20200218', 'v20200702']]

version_ssp245 = [['v20180701'],
                  ['v20180701'],
                  ['v20190429', 'v20190429', 'v20190429', 'v20190429', 'v20190429'],
                  ['v20190429', 'v20190429', 'v20190429'],
                  ['v20191115', 'v20191128', 'v20191203'],
                  ['v20190823', 'v20190823', 'v20190823', 'v20200318', 'v20200318'],
                  ['v20190507', 'v20190718', 'v20190801'],
                  ['v20190119', 'v20190305', 'v20180803', 'v20180803', 'v20180803'],
                  ['v20210311'],
                  ['v20190710', 'v20190710', 'v20190710', 'v20190710'],
                  ['v20190710', 'v20190710', 'v20190710', 'v20190710', 'v20190710'],
                  ['v20190328', 'v20190125', 'v20190125', 'v20190125', 'v20190125'],
                  ['v20191108', 'v20190920', 'v20190920'],
                  ['v20191108', 'v20200218', 'v20200702']]

basin_name = ['Atlantic', 'Pacific', 'Indian', 'Southern', 'Arctic']
basin_nums = [1, 2, 3, 10, 11]

start_time, end_time = '1965-01-01', '2021-12-30'

CMIP_full_fields_dir = '/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2022_OI_algorithm/Preprocessing_RB23/Monthly_full_fields_1965_2021/'

# =========================================================
# Helpers
# =========================================================
#def xmip_wrapper(ds):
#    ds = ds.copy()
#    ds = rename_cmip6(ds)
#    ds = promote_empty_dims(ds)
#    ds = broadcast_lonlat(ds)
#    ds = replace_x_y_nominal_lat_lon(ds)
#    ds = correct_lon(ds)
#    ds = correct_coordinates(ds)
#    ds = parse_lon_lat_bounds(ds)
#    return ds

def fix_lon_0_360(da, lon_name="x"):
    lon = da[lon_name]
    lon_fixed = (lon % 360)
    return da.assign_coords({lon_name: lon_fixed}).sortby(lon_name)

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

def load_full_temperature_and_salinity(i, j):
    #temperature_historical = xr.open_mfdataset(
    #    base_dir + expt[i] + scenario_name + '/' + ensembles[i][j] + '/' +
    #    subdir_temp[i] + version[i][j] + '/thetao*.nc',
    #    combine='by_coords',
    #    decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
    #).sel(time=slice(start_time, end_time))

    #if expt_name[i] != 'MRI_ESM2_0' and expt_name[i] != 'MPI_ESM1_2_HR':
    #    temperature_ssp245 = xr.open_mfdataset(
    #        base_dir + '../ScenarioMIP/' + expt[i] + 'ssp245/' + ensembles[i][j] + '/' +
    #        subdir_temp[i] + version_ssp245[i][j] + '/thetao*.nc',
    #        combine='by_coords',
    #        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
    #    ).sel(time=slice(start_time, end_time))

    #if expt_name[i] == 'IPSL_CM6A_LR':
    #    temperature_historical = temperature_historical.rename({'olevel': 'lev'})
    #    if expt_name[i] != 'MRI_ESM2_0' and expt_name[i] != 'MPI_ESM1_2_HR':
    #        temperature_ssp245 = temperature_ssp245.rename({'olevel': 'lev'})

    #temperature_historical = xmip_wrapper(temperature_historical).thetao

    #if expt_name[i] != 'MRI_ESM2_0' and expt_name[i] != 'MPI_ESM1_2_HR':
    #    temperature_ssp245 = xmip_wrapper(temperature_ssp245).thetao
    #    temperature = xr.concat([temperature_historical, temperature_ssp245], dim='time')
    #else:
    #    temperature = temperature_historical

    #temperature = temperature.sel(time=slice(start_time, end_time))
    #for vv in ['lat', 'lon']:
    #    if vv in temperature.coords or vv in temperature.variables:
    #        temperature = temperature.drop_vars(vv)
    
    temperature = xr.open_dataset(CMIP_full_fields_dir + f'thetao/thetao_1x1bin_{expt_name[i]}_{ensembles[i][j]}.nc')['thetao']
    temperature = fix_lon_0_360(temperature, 'x')
    temperature = temperature.resample(time='1YS').mean('time')

    if(expt_name[i] != 'MRI_ESM2_0' and expt_name[i]!= 'MPI_ESM1_2_HR'):
        temperature['time'] = oxygen_obs_mask.time
    else:
        temperature['time'] = oxygen_obs_mask.time.sel(time = slice('1965-01-01', '2014-12-30'))
    temperature = temperature.interp(x = oxygen_obs_mask.x, y = oxygen_obs_mask.y, lev = oxygen_obs_mask.lev, kwargs = {'fill_value': 'extrapolate'})

    #salt_historical = xr.open_mfdataset(
    #    base_dir + expt[i] + scenario_name + '/' + ensembles[i][j] + '/' +
    #    subdir_salt[i] + version[i][j] + '/so*.nc',
    #    combine='by_coords',
    #    decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
    #).sel(time=slice(start_time, end_time))

    #if expt_name[i] != 'MRI_ESM2_0' and expt_name[i] != 'MPI_ESM1_2_HR':
    #    salt_ssp245 = xr.open_mfdataset(
    #        base_dir + '../ScenarioMIP/' + expt[i] + 'ssp245/' + ensembles[i][j] + '/' +
    #        subdir_salt[i] + version_ssp245[i][j] + '/so*.nc',
    #        combine='by_coords',
    #        decode_times=xr.coders.CFDatetimeCoder(use_cftime=True)
    #    ).sel(time=slice(start_time, end_time))

    #if expt_name[i] == 'IPSL_CM6A_LR':
    #    salt_historical = salt_historical.rename({'olevel': 'lev'})
    #    if expt_name[i] != 'MRI_ESM2_0' and expt_name[i] != 'MPI_ESM1_2_HR':
    #        salt_ssp245 = salt_ssp245.rename({'olevel': 'lev'})

    #salt_historical = xmip_wrapper(salt_historical).so

    #if expt_name[i] != 'MRI_ESM2_0' and expt_name[i] != 'MPI_ESM1_2_HR':
    #    salt_ssp245 = xmip_wrapper(salt_ssp245).so
    #    salt = xr.concat([salt_historical, salt_ssp245], dim='time')
    #else:
    #    salt = salt_historical

    #salt = salt.sel(time=slice(start_time, end_time))
    #for vv in ['lat', 'lon']:
    #    if vv in salt.coords or vv in salt.variables:
    #        salt = salt.drop_vars(vv)
    salt = xr.open_dataset(CMIP_full_fields_dir + f'so/so_1x1bin_{expt_name[i]}_{ensembles[i][j]}.nc')['so']
    salt = fix_lon_0_360(salt, 'x')
    salt = salt.resample(time='1YS').mean('time')

    if(expt_name[i] != 'MRI_ESM2_0' and expt_name[i]!= 'MPI_ESM1_2_HR'):
        salt['time'] = oxygen_obs_mask.time
    else:
        salt['time'] = oxygen_obs_mask.time.sel(time = slice('1965-01-01', '2014-12-30'))
    salt = salt.interp(x = oxygen_obs_mask.x, y = oxygen_obs_mask.y, lev = oxygen_obs_mask.lev, kwargs = {'fill_value': 'extrapolate'})

    sigma0 = potential_density_from_pt(salt, temperature).load()

    return temperature, salt, sigma0

def build_feature_stack(temperature_basin, salt_basin, sigma0_basin):
    needed_dims = ("time", "lev", "y", "x")

    valid4 = np.isfinite(temperature_basin) & np.isfinite(salt_basin) & np.isfinite(sigma0_basin)
    valid1 = valid4.stack(sample=needed_dims).compute()

    keep = np.flatnonzero(valid1.values)
    if keep.size == 0:
        raise ValueError("No valid predictor samples found.")

    t_1d = temperature_basin.stack(sample=needed_dims).isel(sample=keep)
    s_1d = salt_basin.stack(sample=needed_dims).isel(sample=keep)
    sigma0_1d = sigma0_basin.stack(sample=needed_dims).isel(sample=keep)

    year = t_1d["time"].dt.year.astype("float32")

    T = np.asarray(t_1d.astype("float32").values)
    S = np.asarray(s_1d.astype("float32").values)
    sigma0 = np.asarray(sigma0_1d.astype("float32").values)
    lon = np.asarray(t_1d["x"].astype("float32").values)
    lat = np.asarray(t_1d["y"].astype("float32").values)
    dep = np.asarray(t_1d["lev"].astype("float32").values)
    year = np.asarray(year.values, dtype="float32")

    X_raw = np.vstack([S, T, sigma0, lon, lat, dep, year]).astype("float32")

    coords = {
        "sample": t_1d["sample"]   # keep the real MultiIndex coordinate
    }
    return X_raw, coords

def standardize_predictors(X_raw, Xm, Xstd):
    return ((X_raw.T - Xm) / Xstd).astype("float32")

def reconstruct_to_grid(pred_1d, coords, template):
    pred_flat = xr.DataArray(
        pred_1d.astype("float32"),
        dims=("sample",),
        coords={"sample": coords["sample"]}
    )
    pred_unstack = pred_flat.unstack("sample").transpose("time", "lev", "y", "x")
    return pred_unstack.reindex_like(template)

def project_model_to_grid(i, j, k, alg, model_path, params_path):
    model = expt_name[i]
    ens   = ensembles[i][j]
    basin = basin_name[k]

    regr = joblib.load(model_path)
    params = np.load(params_path)

    Xm = params["Xm"].astype("float32")
    Xstd = params["Xstd"].astype("float32")
    ym = float(np.ravel(params["ym"])[0])
    ystd = float(np.ravel(params["ystd"])[0])

    temperature, salt, sigma0 = load_full_temperature_and_salinity(i, j)

    basin_mask = xr.open_dataset(
        '/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2022_OI_algorithm/optint_wod_o2/basin_mask_01.nc'
    )["basin_mask"]
    basin_mask = basin_mask.rename({'lon': 'x', 'lat': 'y', 'depth': 'lev'}).isel(lev = 0)
    basin_mask = fix_lon_0_360(basin_mask)

    sector = (basin_mask == (basin_nums[k]))
    temperature_basin = temperature.where(sector)
    salt_basin = salt.where(sector)
    sigma0_basin = sigma0.where(sector)

    X_raw, coords = build_feature_stack(temperature_basin, salt_basin, sigma0_basin)
    X = standardize_predictors(X_raw, Xm, Xstd)

    y_pred_std = regr.predict(X).astype("float32")
    y_pred = (y_pred_std * ystd + ym).astype("float32")

    template = xr.full_like(temperature_basin, np.nan, dtype=np.float32)
    o2_pred = reconstruct_to_grid(y_pred, coords, template)
    o2_pred.name = "o2"

    o2_pred.attrs["long_name"] = "Projected dissolved oxygen"
    o2_pred.attrs["algorithm"] = alg
    o2_pred.attrs["model_name"] = model
    o2_pred.attrs["ensemble"] = ens
    o2_pred.attrs["basin"] = basin

    fdir = os.path.join(dirfin, model)
    os.makedirs(fdir, exist_ok=True)

    out_file = os.path.join(
        fdir,
        f"projected_o2_{alg}_{basin}_{model}_{ens}.nc"
    )
    o2_pred.to_netcdf(out_file)

    del temperature, salt, sigma0, temperature_basin, salt_basin, sigma0_basin
    del X_raw, X, y_pred_std, y_pred, o2_pred, regr, params
    gc.collect()

    return out_file


# =========================================================
# Main
# =========================================================
if __name__ == "__main__":

    oxygen_obs_mask = xr.open_dataset('/scratch/gpfs/GEOCLIM/LRGROUP/db9274/Analysis/Variability_quantification/Ito_2022_OI_algorithm/' +
                               'Preprocessing_RB23/Monthly_subsampled_fields_1965_2021/o2/Oxygen_obs_OSD_CTD_Argo.nc')['o2'].sel(time = slice(start_time, end_time)).resample(time='1YS').mean('time')
    oxygen_obs_mask = fix_lon_0_360(oxygen_obs_mask, 'x').load();
    
    model = expt_name[i]
    ens   = ensembles[i][j]
    basin = basin_name[k]

    fdir = os.path.join(dirfin, model)

    model_path = os.path.join(
        fdir,
        f"algorithm_{alg}_{basin}_{model}_{ens}.joblib"
    )

    params_path = os.path.join(
        fdir,
        f"ML_params_{alg}_{basin}_{model}_{ens}.npz"
    )

    print("Projecting:", model, ens, basin)
    print("Using model:", model_path)
    print("Using params:", params_path)

    out_file = project_model_to_grid(i, j, k, alg, model_path, params_path)

    print("Projection complete.")
    print("Saved to:", out_file)
