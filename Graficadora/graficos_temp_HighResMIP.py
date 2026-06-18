# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 21:28:58 2026

@author: catar
"""

# -*- coding: utf-8 -*-
"""
HIGHRES TEMPERATURE - PERCENT CHANGE ONLY

Inputs:
  - tasmax historical
  - tasmax future
  - tasmin historical
  - tasmin future

Calculates:
  - Annual mean temperature
  - Historical period mean
  - Future period mean
  - Percent change = (Future - Historical) / abs(Historical) * 100
  - tasmax, tasmin and tmean = (tasmax + tasmin) / 2

Important:
  - Correctly handles monthly files, yearly files and multi-year files.
  - It accumulates all time steps belonging to the same year before calculating
    the annual mean.

Maps:
  - Only percent change
  - Scale fixed from -30% to 30%
  - No title
  - Longitude / Latitude in English
  - Chile data only inside Chile mask
  - Neighboring countries in gray using Natural Earth
  - Ocean in white
  - Hatching where at least 70% of models agree in sign
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

try:
    import geopandas as gpd
except Exception:
    gpd = None

try:
    import shapely
except Exception:
    shapely = None

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)


# ============================================================
# 1. CONFIGURATION
# ============================================================

TASMAX_FUT_DIR = Path(
    r"E:\Cata\HighRes\tasmax_futuro\RESAMPLE_FILES\Regrid_005_conservative"
)

TASMAX_HIST_DIR = Path(
    r"E:\Cata\HighRes\tasmax_hist\RESAMPLE_FILES\Regrid_005_conservative"
)

TASMIN_FUT_DIR = Path(
    r"E:\Cata\HighRes\tasmin_futuro\RESAMPLE_FILES\Regrid_005_conservative"
)

TASMIN_HIST_DIR = Path(
    r"E:\Cata\HighRes\tasmin_hist\RESAMPLE_FILES\Regrid_005_conservative"
)

SHAPEFILE = Path(
    r"C:/Users/catar/OneDrive - Universidad Técnica Federico Santa María/Universidad/2026/Seminario de Tesis I/Provincias/Provincias.shp"
)

OUT_DIR = Path(
    r"E:\Cata\HighRes\salida_temperatura"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Periods
HIST_START_YEAR = 1980
HIST_END_YEAR   = 2010

FUT_START_YEAR  = 2020
FUT_END_YEAR    = 2050

HIST_LABEL = f"{HIST_START_YEAR}-{HIST_END_YEAR}"
FUT_LABEL  = f"{FUT_START_YEAR}-{FUT_END_YEAR}"

# Candidate variable names inside NetCDF files
TASMAX_VAR_CANDIDATES = ["tasmax", "TASMAX", "Tasmax", "tmax", "TMAX"]
TASMIN_VAR_CANDIDATES = ["tasmin", "TASMIN", "Tasmin", "tmin", "TMIN"]

MAKE_MAPS = True

# Keep current coordinates / extent
REGION_EXTENT = (-78.5, -64.5, -56.5, -15.5)

# Percent scale
PCT_LIM = 30

# Agreement hatching
AGREEMENT_PCT = 0.70

# Map style
MAP_FIGSIZE = (4.2, 10.5)
OTHER_COUNTRIES_FACE = "#D9D9D9"
OTHER_COUNTRIES_EDGE = "#9E9E9E"
OCEAN_FACE = "white"
CHILE_BORDER_COLOR = "black"
CHILE_BORDER_WIDTH = 0.6
OTHER_COUNTRIES_WIDTH = 0.5


# ============================================================
# 2. GENERAL FUNCTIONS
# ============================================================

def round_half_up(x):
    """
    Standard rounding:
      4.2 -> 4
      4.5 -> 5
      4.6 -> 5
    """
    return int(np.floor(float(x) + 0.5))


def agreement_threshold(n_models, pct=0.70):
    return max(1, round_half_up(n_models * pct))


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text))


def extract_model_name_from_file(file_path, variable):
    """
    Extract model name from filenames such as:
      tasmax_day_MODEL_...
      tasmin_day_MODEL_...
      tas_day_MODEL_...

    It is flexible because folders are already separated by variable.
    """
    name = Path(file_path).name
    name_no_ext = re.sub(r"\.nc4?$", "", name, flags=re.IGNORECASE)
    parts = name_no_ext.split("_")

    if len(parts) >= 3:
        if parts[0].lower() in [
            variable.lower(),
            "tasmax",
            "tasmin",
            "tmax",
            "tmin",
            "tas"
        ]:
            if parts[1].lower() in ["day", "amon", "mon", "3hr", "6hr"]:
                return parts[2]

    pattern = r"(?:tasmax|tasmin|tmax|tmin|tas|pr)_(?:day|Amon|mon|3hr|6hr)_([^_]+)_"
    match = re.search(pattern, name_no_ext, flags=re.IGNORECASE)

    if match:
        return match.group(1)

    pattern2 = r"_day_([^_]+)_"
    match2 = re.search(pattern2, name_no_ext, flags=re.IGNORECASE)

    if match2:
        return match2.group(1)

    return Path(file_path).parent.name


def find_files_by_model(folder, variable):
    """
    Search .nc and .nc4 files in a direct folder and group them by model.

    Important:
    The folder already corresponds to tasmax or tasmin,
    so this does not require the variable name to appear in the filename.
    """
    folder = Path(folder)

    if not folder.exists():
        print(f"Warning: folder does not exist: {folder}")
        return {}

    files = sorted(
        list(folder.rglob("*.nc")) +
        list(folder.rglob("*.nc4"))
    )

    print(f"\nScanning folder for {variable}:")
    print(f"  {folder}")
    print(f"  NetCDF files found: {len(files)}")

    by_model = {}

    for f in files:
        model = extract_model_name_from_file(f, variable)
        by_model.setdefault(model, []).append(f)

    print(f"  Models detected for {variable}: {len(by_model)}")

    for model, model_files in sorted(by_model.items()):
        print(f"   - {model}: {len(model_files)} files")

    return by_model


def open_dataset_safe(path):
    try:
        return xr.open_dataset(path, decode_times=True, use_cftime=True)
    except TypeError:
        return xr.open_dataset(path, decode_times=True)
    except Exception:
        return xr.open_dataset(path, decode_times=True)


def standardize_coords(da):
    rename = {}

    possible_names = {
        "latitude": "lat",
        "longitude": "lon",
        "nav_lat": "lat",
        "nav_lon": "lon",
        "LAT": "lat",
        "LON": "lon",
        "Latitude": "lat",
        "Longitude": "lon",
        "xlat": "lat",
        "xlon": "lon",
        "XLAT": "lat",
        "XLONG": "lon",
    }

    for old, new in possible_names.items():
        if old in da.coords or old in da.dims:
            rename[old] = new

    if rename:
        da = da.rename(rename)

    # Convert 0-360 longitude to -180/180 if lon is 1D
    if "lon" in da.coords and da["lon"].ndim == 1:
        try:
            lon_max = float(da["lon"].max())
            if lon_max > 180:
                da = da.assign_coords(
                    lon=(((da["lon"] + 180) % 360) - 180)
                )
                da = da.sortby("lon")
        except Exception:
            pass

    return da


def clean_for_concat(da):
    """
    Remove auxiliary coordinates that can break xarray concat.
    """
    da = standardize_coords(da)

    for dim in list(da.dims):
        if dim not in ["time", "lat", "lon", "x", "y", "year", "model"]:
            if da.sizes.get(dim, 0) == 1:
                da = da.isel({dim: 0}, drop=True)

    da = da.squeeze(drop=True)

    keep_coords = {
        "lat", "lon",
        "time", "year", "model",
        "x", "y"
    }

    coords_to_drop = []

    for coord in list(da.coords):
        if coord not in da.dims and coord not in keep_coords:
            coords_to_drop.append(coord)

    if coords_to_drop:
        da = da.drop_vars(coords_to_drop, errors="ignore")

    return da


def select_variable(ds, candidates):
    for cand in candidates:
        if cand in ds.data_vars:
            return cand

    lower_map = {v.lower(): v for v in ds.data_vars}

    for cand in candidates:
        if cand.lower() in lower_map:
            return lower_map[cand.lower()]

    for v in ds.data_vars:
        vlow = v.lower()

        for cand in candidates:
            if cand.lower() in vlow:
                return v

    raise ValueError(
        f"Variable not found among {candidates}. "
        f"Available variables: {list(ds.data_vars)}"
    )


def is_kelvin_temperature(da):
    units = str(da.attrs.get("units", "")).lower().strip()

    kelvin_units = [
        "k",
        "kelvin",
        "degrees_k",
        "degree_k",
        "degree kelvin",
        "degrees kelvin",
    ]

    return units in kelvin_units


def convert_annual_mean_to_celsius(annual_mean, original_da):
    """
    Convert only the annual mean map to Celsius.
    This avoids loading the full NetCDF in memory.
    """
    if is_kelvin_temperature(original_da):
        annual_mean = annual_mean - 273.15
    else:
        try:
            sample = float(annual_mean.mean(skipna=True).values)

            if np.isfinite(sample) and sample > 100:
                annual_mean = annual_mean - 273.15

        except Exception:
            pass

    annual_mean.attrs["units"] = "degC"

    return annual_mean


def spatial_mean(da):
    """
    Spatial mean weighted by cos(lat), ignoring NaN.
    Since the Chile mask leaves ocean as NaN, ocean is not used.
    """
    da = da.squeeze()
    dims = list(da.dims)

    if not dims:
        return float(da.values)

    if "lat" in da.coords:
        try:
            lat = da["lat"]
            weights = np.cos(np.deg2rad(lat))

            if set(weights.dims) != set(da.dims):
                weights = weights.broadcast_like(da)

            weights = weights.where(np.isfinite(da), 0)

            return float(da.weighted(weights).mean(dim=dims).values)

        except Exception:
            pass

    return float(da.mean(dim=dims, skipna=True).values)


# ============================================================
# 3. ROBUST CHILE MASK
# ============================================================

_CHILE_GEOM = None
_MASK_CACHE = {}


def fix_geometry_safe(geom):
    if geom is None:
        return None

    try:
        if geom.is_empty:
            return None
    except Exception:
        return None

    try:
        if shapely is not None and hasattr(shapely, "make_valid"):
            geom = shapely.make_valid(geom)
    except Exception:
        pass

    try:
        if not geom.is_valid:
            geom = geom.buffer(0)
    except Exception:
        try:
            geom = geom.buffer(0)
        except Exception:
            return None

    try:
        if geom.is_empty:
            return None
    except Exception:
        return None

    return geom


def keep_polygonal_parts(geom):
    from shapely.geometry import Polygon, MultiPolygon, GeometryCollection

    if geom is None:
        return []

    if isinstance(geom, Polygon):
        return [geom]

    if isinstance(geom, MultiPolygon):
        return list(geom.geoms)

    if isinstance(geom, GeometryCollection):
        parts = []

        for g in geom.geoms:
            parts.extend(keep_polygonal_parts(g))

        return parts

    return []


def load_chile_geometry(shapefile):
    """
    Load Chile shapefile for masking.
    """
    global _CHILE_GEOM

    if shapefile is None:
        print("No Chile shapefile provided. Chile mask will not be used.")
        return None

    if gpd is None:
        print("Warning: geopandas is not available. Chile mask will not be used.")
        return None

    shapefile = Path(shapefile)

    if not shapefile.exists():
        print(f"Warning: Chile shapefile does not exist: {shapefile}")
        return None

    print("Loading Chile shapefile...")

    try:
        gdf = gpd.read_file(shapefile)
    except Exception as e:
        print("Warning: could not read Chile shapefile.")
        print(f"Reason: {e}")
        return None

    try:
        if gdf.crs is not None:
            gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        pass

    gdf = gdf[gdf.geometry.notnull()].copy()

    if len(gdf) == 0:
        print("Warning: Chile shapefile has no valid geometries.")
        return None

    cleaned_polygons = []

    print("Repairing Chile geometries...")

    for geom in gdf.geometry:
        fixed = fix_geometry_safe(geom)
        parts = keep_polygonal_parts(fixed)
        cleaned_polygons.extend(parts)

    if len(cleaned_polygons) == 0:
        print("Warning: no valid Chile polygons after repair.")
        return None

    try:
        from shapely.ops import unary_union

        geom_union = unary_union(cleaned_polygons)
        geom_union = fix_geometry_safe(geom_union)

        if geom_union is not None:
            _CHILE_GEOM = geom_union
            print("Chile mask loaded using unified geometry.")
            return geom_union

    except Exception as e:
        print("Warning: could not union Chile geometries.")
        print(f"Reason: {e}")
        print("Using separate Chile polygons.")

    _CHILE_GEOM = cleaned_polygons
    print("Chile mask loaded using separate polygons.")

    return cleaned_polygons


def get_xy_for_dataarray(da):
    da = da.squeeze()

    if "lon" not in da.coords or "lat" not in da.coords:
        raise ValueError("lat/lon coordinates were not found.")

    lon = da["lon"]
    lat = da["lat"]

    if lon.ndim == 1 and lat.ndim == 1:
        x, y = np.meshgrid(lon.values, lat.values)
    else:
        x, y = lon.values, lat.values

    return x, y


def contains_xy_safe(geom, x, y):
    if geom is None:
        return np.ones(x.shape, dtype=bool)

    if isinstance(geom, list):
        mask = np.zeros(x.shape, dtype=bool)

        for g in geom:
            try:
                if shapely is not None and hasattr(shapely, "contains_xy"):
                    mask |= shapely.contains_xy(g, x, y)
                else:
                    from shapely.geometry import Point

                    temp = np.zeros(x.shape, dtype=bool)

                    for i in range(x.shape[0]):
                        for j in range(x.shape[1]):
                            temp[i, j] = g.contains(
                                Point(float(x[i, j]), float(y[i, j]))
                            )

                    mask |= temp

            except Exception:
                continue

        return mask

    try:
        if shapely is not None and hasattr(shapely, "contains_xy"):
            return shapely.contains_xy(geom, x, y)
    except Exception:
        pass

    from shapely.geometry import Point

    mask = np.zeros(x.shape, dtype=bool)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            try:
                mask[i, j] = geom.contains(
                    Point(float(x[i, j]), float(y[i, j]))
                )
            except Exception:
                mask[i, j] = False

    return mask


def build_chile_mask(da, chile_geom):
    da = standardize_coords(da)
    da = da.squeeze()

    if chile_geom is None:
        return xr.ones_like(da, dtype=bool)

    if "lat" in da.dims and "lon" in da.dims:
        da_ref = da.transpose("lat", "lon")
    else:
        da_ref = da

    x, y = get_xy_for_dataarray(da_ref)

    key = (
        da_ref.shape,
        float(np.nanmin(x)),
        float(np.nanmax(x)),
        float(np.nanmin(y)),
        float(np.nanmax(y)),
    )

    if key in _MASK_CACHE:
        mask_np = _MASK_CACHE[key]
    else:
        print("Building Chile mask for this grid...")
        mask_np = contains_xy_safe(chile_geom, x, y)
        _MASK_CACHE[key] = mask_np

    mask = xr.DataArray(
        mask_np,
        coords=da_ref.coords,
        dims=da_ref.dims,
        name="mask_chile"
    )

    if set(mask.dims) == set(da.dims):
        mask = mask.transpose(*da.dims)

    return mask


def apply_chile_mask(da, chile_geom):
    if chile_geom is None:
        return da

    da = standardize_coords(da)
    da = clean_for_concat(da)

    if "lat" not in da.coords or "lon" not in da.coords:
        return da

    mask = build_chile_mask(da, chile_geom)

    try:
        return da.where(mask)
    except Exception:
        da_aligned, mask_aligned = xr.align(da, mask, join="inner")
        return da_aligned.where(mask_aligned)


# ============================================================
# 4. SHAPEFILES / NATURAL EARTH FOR PLOTTING
# ============================================================

def load_shape_for_plot(shapefile):
    """
    Load Chile shapefile for plotting borders.
    """
    if shapefile is None or gpd is None:
        return None

    shapefile = Path(shapefile)

    if not shapefile.exists():
        return None

    try:
        gdf = gpd.read_file(shapefile)

        if gdf.crs is not None:
            gdf = gdf.to_crs("EPSG:4326")

        return gdf

    except Exception:
        return None


def load_regional_layer():
    """
    Load neighboring countries from Natural Earth using Cartopy.
    This avoids needing a local LatAm shapefile.
    """
    if gpd is None:
        print("Warning: geopandas is not available. Regional layer will not be shown.")
        return None

    world = None

    try:
        import cartopy.io.shapereader as shpreader

        print("Loading countries from Natural Earth through Cartopy...")

        shpfilename = shpreader.natural_earth(
            resolution="50m",
            category="cultural",
            name="admin_0_countries"
        )

        world = gpd.read_file(shpfilename)

    except Exception as e:
        print("Warning: could not load Natural Earth through Cartopy.")
        print("Install cartopy with:")
        print("conda install -c conda-forge cartopy")
        print(f"Reason: {e}")
        world = None

    if world is None:
        try:
            print("Trying GeoPandas built-in Natural Earth dataset...")
            world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
        except Exception as e:
            print("Warning: GeoPandas built-in Natural Earth also failed.")
            print(f"Reason: {e}")
            world = None

    if world is None:
        print("Warning: no regional countries layer available.")
        print("Map will be generated without gray neighboring countries.")
        return None

    try:
        if world.crs is not None:
            world = world.to_crs("EPSG:4326")
    except Exception:
        pass

    possible_name_cols = ["ADMIN", "NAME", "NAME_EN", "SOVEREIGNT", "name"]

    name_col = None

    for col in possible_name_cols:
        if col in world.columns:
            name_col = col
            break

    if name_col is not None:
        countries_to_keep = [
            "Chile",
            "Argentina",
            "Bolivia",
            "Peru",
            "Brazil",
            "Paraguay",
            "Uruguay",
            "Ecuador",
            "Colombia"
        ]

        try:
            world = world[world[name_col].isin(countries_to_keep)].copy()
        except Exception:
            pass

    lon_min, lon_max, lat_min, lat_max = REGION_EXTENT

    try:
        world = world.cx[lon_min:lon_max, lat_min:lat_max]
    except Exception:
        pass

    print(f"Natural Earth regional layer loaded: {len(world)} geometries")

    try:
        print("Regional geometry types:")
        print(world.geom_type.value_counts())
    except Exception:
        pass

    return world


# ============================================================
# 5. ANNUAL TEMPERATURE CALCULATION
# ============================================================

def compute_annual_and_period_mean(
    files,
    var_candidates,
    model,
    variable_label,
    period_label,
    start_year,
    end_year,
    chile_geom=None
):
    """
    Compute annual mean temperature and period mean.

    This version correctly handles:
      - monthly files
      - yearly files
      - multi-year files

    It accumulates all time steps belonging to the same year before
    calculating the annual mean.
    """

    yearly_accumulator = {}

    for f in files:
        print(f"      Reading: {f.name}")

        try:
            ds = open_dataset_safe(f)
        except Exception as e:
            print(f"      Warning: could not open {f.name}. Skipping.")
            print(f"      Reason: {e}")
            continue

        with ds:
            try:
                var_name = select_variable(ds, var_candidates)

                da = ds[var_name]
                da = standardize_coords(da)
                da = da.squeeze(drop=True)
                da = clean_for_concat(da)

                if "time" not in da.dims:
                    print(f"      Warning: {f.name} has no time dimension. Skipping.")
                    continue

                years = da["time"].dt.year.values.astype(int)
                unique_years = sorted(np.unique(years))

                for yy in unique_years:

                    if yy < start_year or yy > end_year:
                        continue

                    idx = np.where(years == yy)[0]

                    if len(idx) == 0:
                        continue

                    print(f"         Adding data to year {yy}")

                    sub = da.isel(time=idx)

                    sub_sum = sub.sum(dim="time", skipna=True).load()
                    sub_count = sub.count(dim="time").load()

                    sub_sum = clean_for_concat(sub_sum)
                    sub_count = clean_for_concat(sub_count)

                    if yy not in yearly_accumulator:
                        yearly_accumulator[yy] = {
                            "sum": sub_sum,
                            "count": sub_count,
                            "ref_da": da,
                        }
                    else:
                        old_sum = yearly_accumulator[yy]["sum"]
                        old_count = yearly_accumulator[yy]["count"]

                        old_sum, sub_sum = xr.align(
                            old_sum,
                            sub_sum,
                            join="inner"
                        )

                        old_count, sub_count = xr.align(
                            old_count,
                            sub_count,
                            join="inner"
                        )

                        yearly_accumulator[yy]["sum"] = old_sum + sub_sum
                        yearly_accumulator[yy]["count"] = old_count + sub_count

            except Exception as e:
                print(f"      Warning: failed on file {f.name}. Skipping.")
                print(f"      Reason: {e}")
                continue

    if not yearly_accumulator:
        print(
            f"      Warning: no valid years found for "
            f"{model} - {variable_label} - {period_label} "
            f"in {start_year}-{end_year}."
        )
        return None, []

    annual_maps = []
    annual_years = []
    annual_rows = []

    for yy in sorted(yearly_accumulator.keys()):

        print(f"      Calculating annual mean for {yy}")

        total_sum = yearly_accumulator[yy]["sum"]
        total_count = yearly_accumulator[yy]["count"]
        ref_da = yearly_accumulator[yy]["ref_da"]

        annual_mean = total_sum / total_count.where(total_count > 0)

        annual_mean = convert_annual_mean_to_celsius(
            annual_mean,
            ref_da
        )

        annual_mean = clean_for_concat(annual_mean)
        annual_mean = apply_chile_mask(annual_mean, chile_geom)

        annual_mean.name = f"{variable_label}_{period_label}_{model}_{yy}"
        annual_mean.attrs["units"] = "degC"

        annual_maps.append(annual_mean)
        annual_years.append(int(yy))

        annual_rows.append({
            "variable": variable_label,
            "model": model,
            "period": period_label,
            "year": int(yy),
            "spatial_mean_temperature_c": spatial_mean(annual_mean),
        })

    annual_stack = xr.concat(
        annual_maps,
        dim=xr.DataArray(
            annual_years,
            dims="year",
            name="year"
        ),
        join="inner",
        coords="minimal",
        compat="override"
    )

    period_mean = annual_stack.mean(dim="year", skipna=True)
    period_mean = clean_for_concat(period_mean)
    period_mean = apply_chile_mask(period_mean, chile_geom)

    period_mean.name = f"{variable_label}_{period_label}_{model}_mean"
    period_mean.attrs["units"] = "degC"

    return period_mean, annual_rows


def build_tmean_annual_rows(tasmax_rows, tasmin_rows, model, period_label):
    df_max = pd.DataFrame(tasmax_rows)
    df_min = pd.DataFrame(tasmin_rows)

    if df_max.empty or df_min.empty:
        return []

    df_max = df_max[["year", "spatial_mean_temperature_c"]].rename(
        columns={"spatial_mean_temperature_c": "tasmax_c"}
    )

    df_min = df_min[["year", "spatial_mean_temperature_c"]].rename(
        columns={"spatial_mean_temperature_c": "tasmin_c"}
    )

    df = pd.merge(df_max, df_min, on="year", how="inner")

    df["spatial_mean_temperature_c"] = (
        df["tasmax_c"] + df["tasmin_c"]
    ) / 2

    rows = []

    for _, r in df.iterrows():
        rows.append({
            "variable": "tmean",
            "model": model,
            "period": period_label,
            "year": int(r["year"]),
            "spatial_mean_temperature_c": float(r["spatial_mean_temperature_c"]),
        })

    return rows


def add_summary_row(
    summary_rows,
    variable,
    model,
    hist_map,
    fut_map,
    pct_map,
    n_hist,
    n_fut
):
    summary_rows.append({
        "variable": variable,
        "model": model,
        "n_years_historical": int(n_hist),
        "n_years_future": int(n_fut),
        "historical_spatial_mean_c": spatial_mean(hist_map),
        "future_spatial_mean_c": spatial_mean(fut_map),
        "percent_change_spatial_mean": spatial_mean(pct_map),
    })


# ============================================================
# 6. PERCENT CHANGE AND MAPS
# ============================================================

def compute_percent_change(fut_map, hist_map):
    """
    Percent change for Celsius:
      (Future - Historical) / abs(Historical) * 100

    abs(Historical) avoids sign inversion when historical Celsius values are negative.
    """
    denom = xr.where(np.abs(hist_map) > 1e-6, np.abs(hist_map), np.nan)
    pct = ((fut_map - hist_map) / denom) * 100.0
    pct = clean_for_concat(pct)
    pct.attrs["units"] = "%"

    return pct


def make_percent_cmap():
    colors = [
        (0.00, "#08306B"),
        (0.22, "#4292C6"),
        (0.40, "#C6DBEF"),
        (0.50, "#FFFFFF"),
        (0.60, "#FFF7BC"),
        (0.70, "#FEC44F"),
        (0.80, "#F16913"),
        (0.88, "#D7301F"),
        (0.95, "#8C510A"),
        (1.00, "#000000"),
    ]

    return LinearSegmentedColormap.from_list(
        "percent_change_cmap",
        colors,
        N=256
    )


def get_xy_for_plot(da):
    da = da.squeeze()

    if "lat" in da.dims and "lon" in da.dims:
        da = da.transpose("lat", "lon")

    if "lon" not in da.coords or "lat" not in da.coords:
        raise ValueError("lat/lon coordinates were not found for plotting.")

    lon = da["lon"]
    lat = da["lat"]

    if lon.ndim == 1 and lat.ndim == 1:
        x, y = np.meshgrid(lon.values, lat.values)
    else:
        x, y = lon.values, lat.values

    z = da.values

    return x, y, z


def plot_percent_map(
    da,
    output_path,
    gdf_chile=None,
    gdf_region=None,
    agreement_mask=None
):
    cmap = make_percent_cmap()
    norm = TwoSlopeNorm(vmin=-PCT_LIM, vcenter=0, vmax=PCT_LIM)

    x, y, z = get_xy_for_plot(da)

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE)
    ax.set_facecolor(OCEAN_FACE)

    # Neighboring countries in gray
    if gdf_region is not None:
        try:
            gdf_region.plot(
                ax=ax,
                facecolor=OTHER_COUNTRIES_FACE,
                edgecolor=OTHER_COUNTRIES_EDGE,
                linewidth=OTHER_COUNTRIES_WIDTH,
                zorder=1
            )
        except Exception as e:
            print("Warning: could not plot regional layer.")
            print(f"Reason: {e}")

    # Chile data
    im = ax.pcolormesh(
        x,
        y,
        z,
        cmap=cmap,
        norm=norm,
        shading="auto",
        zorder=3
    )

    # Hatching
    if agreement_mask is not None:
        agreement_mask = agreement_mask.squeeze()

        if "lat" in agreement_mask.dims and "lon" in agreement_mask.dims:
            agreement_mask = agreement_mask.transpose("lat", "lon")

        _, _, z_mask = get_xy_for_plot(agreement_mask)

        ax.contourf(
            x,
            y,
            z_mask.astype(float),
            levels=[0.5, 1.5],
            colors="none",
            hatches=["///"],
            alpha=0,
            zorder=4
        )

    # Chile borders over everything
    if gdf_chile is not None:
        try:
            gdf_chile.boundary.plot(
                ax=ax,
                linewidth=CHILE_BORDER_WIDTH,
                color=CHILE_BORDER_COLOR,
                zorder=5
            )
        except Exception as e:
            print("Warning: could not plot Chile border.")
            print(f"Reason: {e}")

    lon_min, lon_max, lat_min, lat_max = REGION_EXTENT
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_title("")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)

    cbar = fig.colorbar(
        im,
        ax=ax,
        shrink=0.82,
        pad=0.06,
        extend="both"
    )

    cbar.set_label(
        "Percent change (%)",
        fontsize=11,
        rotation=90,
        labelpad=15
    )

    cbar.set_ticks([-30, -20, -10, 0, 10, 20, 30])
    cbar.ax.tick_params(labelsize=10)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)


# ============================================================
# 7. MAIN PROCESS
# ============================================================

def main():
    print("====================================================")
    print("HIGHRES TEMPERATURE - PERCENT CHANGE ONLY")
    print("Annual mean -> period mean -> percent change")
    print("Chile mask enabled")
    print("Natural Earth regional background enabled")
    print("====================================================")

    print("\nFolders:")
    print(f"TASMAX historical: {TASMAX_HIST_DIR}")
    print(f"TASMAX future:     {TASMAX_FUT_DIR}")
    print(f"TASMIN historical: {TASMIN_HIST_DIR}")
    print(f"TASMIN future:     {TASMIN_FUT_DIR}")
    print(f"Output:            {OUT_DIR}")

    print("\nPeriods:")
    print(f"Historical: {HIST_LABEL}")
    print(f"Future:     {FUT_LABEL}")

    chile_geom = load_chile_geometry(SHAPEFILE)
    gdf_chile = load_shape_for_plot(SHAPEFILE)
    gdf_region = load_regional_layer()

    print("\nDetecting files by model...")

    tasmax_hist_models = find_files_by_model(TASMAX_HIST_DIR, "tasmax")
    tasmax_fut_models  = find_files_by_model(TASMAX_FUT_DIR, "tasmax")

    tasmin_hist_models = find_files_by_model(TASMIN_HIST_DIR, "tasmin")
    tasmin_fut_models  = find_files_by_model(TASMIN_FUT_DIR, "tasmin")

    print("\n====================================================")
    print("VARIABLE AVAILABILITY DIAGNOSTIC")
    print("====================================================")

    all_detected_models = sorted(
        set(tasmax_hist_models.keys()) |
        set(tasmax_fut_models.keys()) |
        set(tasmin_hist_models.keys()) |
        set(tasmin_fut_models.keys())
    )

    for model in all_detected_models:
        print(
            f"{model}: "
            f"tasmax_hist={model in tasmax_hist_models}, "
            f"tasmax_fut={model in tasmax_fut_models}, "
            f"tasmin_hist={model in tasmin_hist_models}, "
            f"tasmin_fut={model in tasmin_fut_models}"
        )

    print("====================================================\n")

    all_models = all_detected_models

    if not all_models:
        raise ValueError("No models were found in the input folders.")

    print(f"\nDetected models: {len(all_models)}")

    for m in all_models:
        print(f" - {m}")

    summary_rows = []
    annual_rows_all = []

    pct_by_var = {
        "tasmax": {},
        "tasmin": {},
        "tmean": {},
    }

    # --------------------------------------------------------
    # ABSOLUTE CHANGE BLOCK COMMENTED OUT
    # --------------------------------------------------------
    #
    # If later you want absolute temperature change in °C:
    #
    # delta_abs = fut_map - hist_map
    # delta_abs.attrs["units"] = "degC"
    #
    # --------------------------------------------------------

    for model in all_models:
        print("\n----------------------------------------------------")
        print(f"Processing model: {model}")
        print("----------------------------------------------------")

        has_tasmax = (
            model in tasmax_hist_models and
            model in tasmax_fut_models
        )

        has_tasmin = (
            model in tasmin_hist_models and
            model in tasmin_fut_models
        )

        if not has_tasmax and not has_tasmin:
            print(f"Warning: {model} does not have tasmax or tasmin in both periods. Skipping.")
            continue

        available_vars = []

        if has_tasmax:
            available_vars.append("tasmax")

        if has_tasmin:
            available_vars.append("tasmin")

        print(f"Available variables for {model}: {available_vars}")

        tasmax_hist_map = None
        tasmax_fut_map = None
        tasmax_hist_rows = []
        tasmax_fut_rows = []

        tasmin_hist_map = None
        tasmin_fut_map = None
        tasmin_hist_rows = []
        tasmin_fut_rows = []

        # ====================================================
        # TASMAX
        # ====================================================
        if has_tasmax:
            print("   TASMAX historical")

            tasmax_hist_map, tasmax_hist_rows = compute_annual_and_period_mean(
                files=tasmax_hist_models[model],
                var_candidates=TASMAX_VAR_CANDIDATES,
                model=model,
                variable_label="tasmax",
                period_label="historical",
                start_year=HIST_START_YEAR,
                end_year=HIST_END_YEAR,
                chile_geom=chile_geom
            )

            print("   TASMAX future")

            tasmax_fut_map, tasmax_fut_rows = compute_annual_and_period_mean(
                files=tasmax_fut_models[model],
                var_candidates=TASMAX_VAR_CANDIDATES,
                model=model,
                variable_label="tasmax",
                period_label="future",
                start_year=FUT_START_YEAR,
                end_year=FUT_END_YEAR,
                chile_geom=chile_geom
            )

            if tasmax_hist_map is not None and tasmax_fut_map is not None:
                try:
                    tasmax_hist_map, tasmax_fut_map = xr.align(
                        tasmax_hist_map,
                        tasmax_fut_map,
                        join="inner"
                    )

                    tasmax_pct = compute_percent_change(
                        tasmax_fut_map,
                        tasmax_hist_map
                    )

                    tasmax_pct = apply_chile_mask(tasmax_pct, chile_geom)
                    tasmax_pct.name = f"tasmax_pct_{model}"
                    tasmax_pct.attrs["units"] = "%"

                    annual_rows_all.extend(tasmax_hist_rows)
                    annual_rows_all.extend(tasmax_fut_rows)

                    add_summary_row(
                        summary_rows,
                        "tasmax",
                        model,
                        tasmax_hist_map,
                        tasmax_fut_map,
                        tasmax_pct,
                        len({r["year"] for r in tasmax_hist_rows}),
                        len({r["year"] for r in tasmax_fut_rows}),
                    )

                    pct_by_var["tasmax"][model] = tasmax_pct

                except Exception as e:
                    print(f"   Warning: TASMAX failed for {model}. Skipping.")
                    print(f"   Reason: {e}")

                    tasmax_hist_map = None
                    tasmax_fut_map = None
                    tasmax_hist_rows = []
                    tasmax_fut_rows = []

        else:
            print(f"   {model}: TASMAX not processed because it is missing in one period.")

        # ====================================================
        # TASMIN
        # ====================================================
        if has_tasmin:
            print("   TASMIN historical")

            tasmin_hist_map, tasmin_hist_rows = compute_annual_and_period_mean(
                files=tasmin_hist_models[model],
                var_candidates=TASMIN_VAR_CANDIDATES,
                model=model,
                variable_label="tasmin",
                period_label="historical",
                start_year=HIST_START_YEAR,
                end_year=HIST_END_YEAR,
                chile_geom=chile_geom
            )

            print("   TASMIN future")

            tasmin_fut_map, tasmin_fut_rows = compute_annual_and_period_mean(
                files=tasmin_fut_models[model],
                var_candidates=TASMIN_VAR_CANDIDATES,
                model=model,
                variable_label="tasmin",
                period_label="future",
                start_year=FUT_START_YEAR,
                end_year=FUT_END_YEAR,
                chile_geom=chile_geom
            )

            if tasmin_hist_map is not None and tasmin_fut_map is not None:
                try:
                    tasmin_hist_map, tasmin_fut_map = xr.align(
                        tasmin_hist_map,
                        tasmin_fut_map,
                        join="inner"
                    )

                    tasmin_pct = compute_percent_change(
                        tasmin_fut_map,
                        tasmin_hist_map
                    )

                    tasmin_pct = apply_chile_mask(tasmin_pct, chile_geom)
                    tasmin_pct.name = f"tasmin_pct_{model}"
                    tasmin_pct.attrs["units"] = "%"

                    annual_rows_all.extend(tasmin_hist_rows)
                    annual_rows_all.extend(tasmin_fut_rows)

                    add_summary_row(
                        summary_rows,
                        "tasmin",
                        model,
                        tasmin_hist_map,
                        tasmin_fut_map,
                        tasmin_pct,
                        len({r["year"] for r in tasmin_hist_rows}),
                        len({r["year"] for r in tasmin_fut_rows}),
                    )

                    pct_by_var["tasmin"][model] = tasmin_pct

                except Exception as e:
                    print(f"   Warning: TASMIN failed for {model}. Skipping.")
                    print(f"   Reason: {e}")

                    tasmin_hist_map = None
                    tasmin_fut_map = None
                    tasmin_hist_rows = []
                    tasmin_fut_rows = []

        else:
            print(f"   {model}: TASMIN not processed because it is missing in one period.")

        # ====================================================
        # TMEAN
        # ====================================================
        if (
            tasmax_hist_map is not None and
            tasmax_fut_map is not None and
            tasmin_hist_map is not None and
            tasmin_fut_map is not None
        ):
            print("   Calculating TMEAN = (TASMAX + TASMIN) / 2")

            try:
                tasmax_hist_map, tasmin_hist_map = xr.align(
                    tasmax_hist_map,
                    tasmin_hist_map,
                    join="inner"
                )

                tasmax_fut_map, tasmin_fut_map = xr.align(
                    tasmax_fut_map,
                    tasmin_fut_map,
                    join="inner"
                )

                tmean_hist_map = (tasmax_hist_map + tasmin_hist_map) / 2
                tmean_fut_map = (tasmax_fut_map + tasmin_fut_map) / 2

                tmean_pct = compute_percent_change(
                    tmean_fut_map,
                    tmean_hist_map
                )

                tmean_pct = apply_chile_mask(tmean_pct, chile_geom)
                tmean_pct.name = f"tmean_pct_{model}"
                tmean_pct.attrs["units"] = "%"

                annual_rows_all.extend(
                    build_tmean_annual_rows(
                        tasmax_hist_rows,
                        tasmin_hist_rows,
                        model,
                        "historical"
                    )
                )

                annual_rows_all.extend(
                    build_tmean_annual_rows(
                        tasmax_fut_rows,
                        tasmin_fut_rows,
                        model,
                        "future"
                    )
                )

                add_summary_row(
                    summary_rows,
                    "tmean",
                    model,
                    tmean_hist_map,
                    tmean_fut_map,
                    tmean_pct,
                    min(
                        len({r["year"] for r in tasmax_hist_rows}),
                        len({r["year"] for r in tasmin_hist_rows}),
                    ),
                    min(
                        len({r["year"] for r in tasmax_fut_rows}),
                        len({r["year"] for r in tasmin_fut_rows}),
                    ),
                )

                pct_by_var["tmean"][model] = tmean_pct

            except Exception as e:
                print(f"   Warning: TMEAN failed for {model}. Skipping.")
                print(f"   Reason: {e}")

        else:
            print(f"   {model}: TMEAN not calculated because TASMAX or TASMIN is missing.")

    # ========================================================
    # SAVE CSV
    # ========================================================

    summary_df = pd.DataFrame(summary_rows)
    annual_df = pd.DataFrame(annual_rows_all)

    summary_csv = OUT_DIR / "temperature_summary_models.csv"
    annual_csv = OUT_DIR / "temperature_annual_models.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    annual_df.to_csv(annual_csv, index=False, encoding="utf-8-sig")

    print("\nCSV files generated:")
    print(f" - {summary_csv}")
    print(f" - {annual_csv}")

    # ========================================================
    # MULTI-MODEL, HATCHING AND MAPS
    # ========================================================

    threshold_rows = []

    for variable in ["tasmax", "tasmin", "tmean"]:

        if len(pct_by_var[variable]) == 0:
            print(f"\nNo models available for {variable}. Multi-model skipped.")
            continue

        print("\n====================================================")
        print(f"Processing multi-model: {variable}")
        print("====================================================")

        models = sorted(pct_by_var[variable].keys())

        pct_list = [
            apply_chile_mask(
                clean_for_concat(pct_by_var[variable][m]),
                chile_geom
            )
            for m in models
        ]

        aligned_pct = xr.align(*pct_list, join="inner")

        aligned_pct = [
            apply_chile_mask(clean_for_concat(a), chile_geom)
            for a in aligned_pct
        ]

        stack_pct = xr.concat(
            aligned_pct,
            dim=xr.DataArray(
                models,
                dims="model",
                name="model"
            ),
            coords="minimal",
            compat="override",
            join="inner"
        )

        stack_pct.name = f"{variable}_percent_by_model"
        stack_pct.attrs["units"] = "%"

        multimodel_pct = stack_pct.mean(dim="model", skipna=True)
        multimodel_pct = clean_for_concat(multimodel_pct)
        multimodel_pct = apply_chile_mask(multimodel_pct, chile_geom)
        multimodel_pct.name = f"{variable}_percent_multimodel"
        multimodel_pct.attrs["units"] = "%"

        n_models = len(models)
        threshold = agreement_threshold(n_models, AGREEMENT_PCT)

        n_positive = (stack_pct > 0).sum(dim="model")
        n_negative = (stack_pct < 0).sum(dim="model")

        agreement_mask = (n_positive >= threshold) | (n_negative >= threshold)
        agreement_mask = clean_for_concat(agreement_mask)
        agreement_mask = apply_chile_mask(agreement_mask, chile_geom)
        agreement_mask.name = f"{variable}_agreement_hatching"

        actual_pct = int(np.ceil((threshold / n_models) * 100))

        print(f"Models used: {n_models}")
        print(f"70% of models: {n_models * AGREEMENT_PCT}")
        print(f"Hatching threshold used: {threshold}")
        print(f"Actual equivalent percent: ≥{actual_pct}%")

        threshold_rows.append({
            "variable": variable,
            "n_models_used": n_models,
            "target_70_pct_value": n_models * AGREEMENT_PCT,
            "hatching_threshold_used": threshold,
            "actual_equivalent_percent": actual_pct,
            "rule": "standard rounding: 4.2->4, 4.6->5, 4.5->5",
        })

        stack_pct.to_netcdf(OUT_DIR / f"{variable}_percent_by_model.nc")
        multimodel_pct.to_netcdf(OUT_DIR / f"{variable}_percent_multimodel.nc")

        agreement_mask.fillna(0).astype("int8").to_netcdf(
            OUT_DIR / f"{variable}_agreement_hatching.nc"
        )

        print(f"NetCDF files generated for {variable}.")

        if MAKE_MAPS:

            var_out = OUT_DIR / f"maps_{variable}"
            var_out.mkdir(parents=True, exist_ok=True)

            # Per-model maps
            for model, da_pct in zip(models, aligned_pct):

                output_png = var_out / f"{variable}_percent_{safe_name(model)}.png"

                plot_percent_map(
                    da=da_pct,
                    output_path=output_png,
                    gdf_chile=gdf_chile,
                    gdf_region=gdf_region,
                    agreement_mask=None
                )

            # Multi-model map with hatching
            output_png = var_out / f"{variable}_percent_multimodel_hatched.png"

            plot_percent_map(
                da=multimodel_pct,
                output_path=output_png,
                gdf_chile=gdf_chile,
                gdf_region=gdf_region,
                agreement_mask=agreement_mask
            )

            print(f"Maps generated in: {var_out}")

    # ========================================================
    # SAVE HATCHING THRESHOLD CSV
    # ========================================================

    if len(threshold_rows) > 0:
        threshold_df = pd.DataFrame(threshold_rows)

        threshold_csv = OUT_DIR / "temperature_hatching_threshold.csv"

        threshold_df.to_csv(
            threshold_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nHatching threshold CSV generated:")
        print(f" - {threshold_csv}")

    print("\n====================================================")
    print("PROCESS FINISHED")
    print("====================================================")
    print(f"Output folder: {OUT_DIR}")


if __name__ == "__main__":
    main()