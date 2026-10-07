"""smogsense.preprocessing.gridded - GRIB/NetCDF to station time series.

Reads CAMS/ERA5 files via xarray+cfgrib, converts units (kg m-3 to ug m-3, K to degC), and
extracts station series by bilinear interpolation recording grid distance.

Public contract (implemented in Phase 1):
- extract_stations(grib_path, stations) -> DataFrame

Specification: docs/data-engineering.md -> 'Copernicus ADS: CAMS global forecasts'
"""
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
import yaml
from pyproj import Geod
from scipy.interpolate import RegularGridInterpolator

PM25_ALIASES = {"pm2p5", "particulate_matter_2.5um", "pm25"}
TEMP_ALIASES = {"t2m", "2m_temperature"}
DEWPOINT_ALIASES = {"d2m", "2m_dewpoint_temperature"}

def get_centroid_from_config(domain: str = "lahore") -> dict[str, Any]:
    project_root = Path(__file__).resolve().parents[3]
    config_path = project_root / "configs" / "domains.yaml"
    if not config_path.exists():
        return {}
    with open(config_path, encoding="utf-8") as f:
        conf = yaml.safe_load(f)
        if conf is None:
            conf = {}
    res = conf.get("domains", {}).get(domain, {}).get("centroid", {})
    return dict(res) if res else {}

def calc_rh(t_c: np.ndarray, td_c: np.ndarray) -> np.ndarray:
    """Calculate Relative Humidity using Magnus formula."""
    e_t = 6.1094 * np.exp(17.625 * t_c / (t_c + 243.04))
    e_td = 6.1094 * np.exp(17.625 * td_c / (td_c + 243.04))
    return 100.0 * e_td / e_t

def _normalize_longitudes(pts_lon: np.ndarray, grid_lon: np.ndarray) -> np.ndarray:
    if len(pts_lon) == 0:
        return pts_lon

    grid_min, grid_max = grid_lon.min(), grid_lon.max()
    pts_out = pts_lon.copy()

    # if grid is [0, 360]
    if grid_max > 180 and pts_out.min() < 0:
        pts_out = np.where(pts_out < 0, pts_out + 360.0, pts_out)

    # if grid is [-180, 180]
    if grid_min < 0 and pts_out.max() > 180:
        pts_out = np.where(pts_out > 180, pts_out - 360.0, pts_out)

    return pts_out

def extract_stations(grib_path: Path, stations: pd.DataFrame, domain: str = "lahore") -> pd.DataFrame:
    """Extract station and centroid series from GRIB via bilinear interpolation."""
    required_cols = {"location_id", "lat", "lon"}
    if not required_cols.issubset(stations.columns):
        raise ValueError(f"Station DataFrame missing required columns: {required_cols - set(stations.columns)}")

    if stations.empty:
        return pd.DataFrame()

    df_pts = stations.copy()

    # 1. Add centroid if missing
    centroid = get_centroid_from_config(domain)
    if not centroid:
        raise ValueError(f"Centroid configuration missing for domain: {domain}")

    if "centroid" not in df_pts["location_id"].values:
        cent_row = pd.DataFrame([{
            "location_id": "centroid",
            "lat": centroid["lat"],
            "lon": centroid["lon"]
        }])
        df_pts = pd.concat([df_pts, cent_row], ignore_index=True)

    pts_lat = np.asarray(df_pts["lat"], dtype=float)
    pts_lon = np.asarray(df_pts["lon"], dtype=float)

    # 2. Open GRIB
    with xr.open_dataset(grib_path, engine="cfgrib") as ds:
        # Find spatial dimensions
        lat_dim = next((d for d in ds.dims if d in ("latitude", "lat")), None)
        lon_dim = next((d for d in ds.dims if d in ("longitude", "lon")), None)

        if not lat_dim or not lon_dim:
            raise ValueError("Could not identify spatial dimensions (latitude/longitude) in GRIB.")

        grid_lats = ds[lat_dim].values
        grid_lons = ds[lon_dim].values

        pts_lon = _normalize_longitudes(pts_lon, grid_lons)

        # 3. Find distance to nearest node
        # Document: grid_distance_km represents the distance to the nearest grid node, not interpolation distance.
        geod = Geod(ellps="WGS84")
        dist_km = []
        for i in range(len(pts_lat)):
            if np.isnan(pts_lat[i]) or np.isnan(pts_lon[i]):
                dist_km.append(np.nan)
            else:
                lat_idx = np.abs(grid_lats - pts_lat[i]).argmin()
                lon_idx = np.abs(grid_lons - pts_lon[i]).argmin()
                _, _, d = geod.inv(pts_lon[i], pts_lat[i], grid_lons[lon_idx], grid_lats[lat_idx])
                dist_km.append(d / 1000.0)

        df_pts["grid_distance_km"] = dist_km

        # 4. Interpolate variables
        lats_asc = grid_lats
        lons_asc = grid_lons

        flip_lat = False
        if len(grid_lats) > 1 and grid_lats[0] > grid_lats[-1]:
            flip_lat = True
            lats_asc = grid_lats[::-1]

        flip_lon = False
        if len(grid_lons) > 1 and grid_lons[0] > grid_lons[-1]:
            flip_lon = True
            lons_asc = grid_lons[::-1]

        valid_mask = ~(np.isnan(pts_lat) | np.isnan(pts_lon))
        interp_pts = np.column_stack((pts_lat[valid_mask], pts_lon[valid_mask]))

        non_spatial_dims = [d for d in ds.dims if d not in (lat_dim, lon_dim)]

        if non_spatial_dims:
            ds_stacked = ds.stack(sample=non_spatial_dims)
        else:
            ds_stacked = ds.expand_dims("sample")

        ds_stacked = ds_stacked.transpose("sample", lat_dim, lon_dim)
        num_samples = ds_stacked.sizes["sample"]

        if "valid_time" in ds_stacked.coords:
            valid_times = np.atleast_1d(ds_stacked["valid_time"].values)
        elif "time" in ds_stacked.coords and "step" in ds_stacked.coords:
            valid_times = np.atleast_1d(ds_stacked["time"].values + ds_stacked["step"].values)
        else:
            valid_times = np.array([pd.NaT] * num_samples)

        if len(valid_times) == 1 and num_samples > 1:
            valid_times = np.repeat(valid_times, num_samples)

        records = []

        for i_sample in range(num_samples):
            t_val = valid_times[i_sample]
            ts_obj = pd.Timestamp(t_val) if pd.notnull(t_val) else pd.NaT
            if pd.isnull(ts_obj):
                ts_utc = pd.NaT  # type: ignore
            elif ts_obj.tzinfo is None:
                ts_utc = ts_obj.tz_localize("UTC")  # type: ignore
            else:
                ts_utc = ts_obj.tz_convert("UTC")  # type: ignore

            extracted_vars: dict[str, np.ndarray] = {}
            t_c = None
            td_c = None

            for v_obj in ds.data_vars:
                v = str(v_obj)
                da = ds_stacked[v]

                if "sample" in da.dims:
                    da_slice = da.isel(sample=i_sample)
                else:
                    da_slice = da

                if lat_dim not in da_slice.dims or lon_dim not in da_slice.dims:
                    continue

                da_slice = da_slice.transpose(lat_dim, lon_dim)
                data = da_slice.values

                if flip_lat:
                    data = data[::-1, :]
                if flip_lon:
                    data = data[:, ::-1]

                interp = RegularGridInterpolator(
                    (lats_asc, lons_asc),
                    data,
                    method='linear',
                    bounds_error=False,
                    fill_value=np.nan
                )
                vals = interp(interp_pts)

                full_vals = np.full(len(pts_lat), np.nan)
                full_vals[valid_mask] = vals

                units = da.attrs.get("units", "").strip()

                if v in PM25_ALIASES:
                    if units in ("kg m**-3", "kg m-3"):
                        extracted_vars["pm25_ugm3"] = full_vals * 1e9
                    elif units in ("ug m**-3", "ug m-3", "µg m-3", "µg/m3"):
                        extracted_vars["pm25_ugm3"] = full_vals
                    else:
                        raise ValueError(f"Unknown PM2.5 units: {units} for variable {v}")
                elif v in TEMP_ALIASES:
                    if units == "K":
                        t_c = full_vals - 273.15
                    elif units in ("C", "degC", "°C"):
                        t_c = full_vals
                    else:
                        raise ValueError(f"Unknown temperature units: {units} for variable {v}")
                elif v in DEWPOINT_ALIASES:
                    if units == "K":
                        td_c = full_vals - 273.15
                    elif units in ("C", "degC", "°C"):
                        td_c = full_vals
                    else:
                        raise ValueError(f"Unknown dewpoint units: {units} for variable {v}")
                else:
                    extracted_vars[v] = full_vals

            if t_c is not None:
                extracted_vars["temperature_c"] = t_c
                if td_c is not None:
                    rh = calc_rh(t_c, td_c)
                    rh = np.clip(rh, 0.0, 100.0)
                    extracted_vars["rh_pct"] = rh

            # Build records for this timestamp
            for i_loc, loc_id in enumerate(df_pts["location_id"]):
                rec = {
                    "location_id": loc_id,
                    "ts_utc": ts_utc,
                    "grid_distance_km": df_pts["grid_distance_km"].iloc[i_loc],
                }
                # Quality indicator for out-of-domain stations
                if not valid_mask[i_loc]:
                    rec["outside_grid"] = True
                else:
                    # check if distance is nan (happens if out of bounds)
                    # Actually, if bounding box is tight, values might be NaN
                    rec["outside_grid"] = False

                for k, v_arr in extracted_vars.items():
                    rec[k] = v_arr[i_loc]
                    if np.isnan(v_arr[i_loc]):
                        rec["outside_grid"] = True

                records.append(rec)

    out_df = pd.DataFrame(records)
    return out_df.reset_index(drop=True)
