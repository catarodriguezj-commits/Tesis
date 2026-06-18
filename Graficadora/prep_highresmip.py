# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:20:51 2026

@author: catar
"""

# -*- coding: utf-8 -*-
"""
HIGHRES PRECIPITATION CHANGE MAPS - ANNUAL + SEASONAL

Calcula:
  - precipitación acumulada anual y estacional
  - delta_mm  = futuro - histórico
  - delta_pct = ((futuro - histórico) / histórico) * 100

Mapas:
  - sin escala lateral
  - escala PNG aparte
  - rojo = más seco
  - azul = más húmedo
  - escala absoluta: ±500 mm
  - escala porcentual: ±30%
  - achurado donde >=70% de modelos coinciden en signo

Al final imprime:
  - modelos procesados
  - modelos omitidos
  - modelos con error
  - modelos incluidos en cada multimodelo
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

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
# 1. CONFIGURACIÓN
# ============================================================

HIGHRES_HIST_DIR = Path(
    r"E:\Cata\HighRes\pr_hist\RESAMPLE_FILES\Regrid_005_conservative"
)

HIGHRES_FUT_DIR = Path(
    r"E:\Cata\HighRes\pr_futuro\RESAMPLE_FILES\Regrid_005_conservative"
)

SHAPEFILE = Path(
    r"C:\Users\catar\OneDrive - Universidad Técnica Federico Santa María\Universidad\2026\Seminario de Tesis I\Provincias\Provincias.shp"
)

OUT_DIR = Path(
    r"E:\Cata\HighRes\salida_precipitacion"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Periodos
HIST_START_YEAR = 1980
HIST_END_YEAR   = 2010

FUT_START_YEAR  = 2020
FUT_END_YEAR    = 2050

HIST_LABEL = f"{HIST_START_YEAR}-{HIST_END_YEAR}"
FUT_LABEL  = f"{FUT_START_YEAR}-{FUT_END_YEAR}"

# Variable precipitación
PR_VAR_CANDIDATES = ["pr", "PR", "precip", "precipitation", "Precipitation"]

# Extensión mapa
REGION_EXTENT = (-78.5, -64.5, -56.5, -15.5)

# Escalas
PR_MM_MAX = 500
PR_PCT_MAX = 30

# Achurado
AGREEMENT_PCT = 0.70

# Mapas
MAKE_MODEL_MAPS = True
MAKE_MULTIMODEL_MAPS = True

MAP_FIGSIZE = (4.2, 10.5)

OTHER_COUNTRIES_FACE = "#D9D9D9"
OTHER_COUNTRIES_EDGE = "#9E9E9E"
OCEAN_FACE = "white"

CHILE_BORDER_COLOR = "black"
CHILE_BORDER_WIDTH = 0.6
OTHER_COUNTRIES_WIDTH = 0.5

# Si hay archivos originales y regrillados, prefiere regrillados
PREFER_REGRID_FILES = True

# Estaciones
SEASONS = {
    "verano_DJF": {
        "label": "Summer (DJF)",
        "months": [12, 1, 2],
    },
    "otono_MAM": {
        "label": "Autumn (MAM)",
        "months": [3, 4, 5],
    },
    "invierno_JJA": {
        "label": "Winter (JJA)",
        "months": [6, 7, 8],
    },
    "primavera_SON": {
        "label": "Spring (SON)",
        "months": [9, 10, 11],
    },
}


# ============================================================
# 2. FUNCIONES GENERALES
# ============================================================

def round_half_up(x):
    return int(np.floor(float(x) + 0.5))


def agreement_threshold(n_models, pct=0.70):
    return max(1, round_half_up(n_models * pct))


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text))


def register_failure(failed_models, model, step, reason):
    failed_models.append({
        "model": model,
        "step": step,
        "reason": str(reason),
    })


def open_dataset_safe(path):
    """
    Abre NetCDF con chunks para evitar cargar archivos gigantes completos.
    """
    try:
        return xr.open_dataset(
            path,
            decode_times=True,
            use_cftime=True,
            chunks={"time": 365}
        )
    except Exception:
        try:
            return xr.open_dataset(
                path,
                decode_times=True,
                chunks={"time": 365}
            )
        except Exception:
            try:
                return xr.open_dataset(
                    path,
                    decode_times=True,
                    use_cftime=True
                )
            except TypeError:
                return xr.open_dataset(
                    path,
                    decode_times=True
                )


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
        f"No se encontró variable entre {candidates}. "
        f"Variables disponibles: {list(ds.data_vars)}"
    )


def spatial_mean(da):
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
# 3. CONVERSIÓN DE PRECIPITACIÓN
# ============================================================

def convert_precip_to_mm_per_timestep(da):
    """
    Convierte precipitación a mm por paso temporal.

    HighResMIP normalmente:
      pr = kg m-2 s-1

    Si son datos diarios:
      kg m-2 s-1 * 86400 = mm/day

    Luego se suma en el tiempo.
    """
    units = str(da.attrs.get("units", "")).lower().strip()

    if (
        "kg m-2 s-1" in units or
        "kg m^-2 s^-1" in units or
        "kg/m2/s" in units or
        "mm s-1" in units or
        "mm/s" in units or
        "s-1" in units
    ):
        out = da * 86400.0
        out.attrs["units"] = "mm_per_day"
        return out

    if (
        "mm/day" in units or
        "mm d-1" in units or
        "kg m-2 day-1" in units or
        "kg/m2/day" in units or
        "day" in units or
        "d-1" in units
    ):
        out = da.copy()
        out.attrs["units"] = "mm_per_day"
        return out

    if units in ["mm", "kg m-2", "kg/m2"]:
        out = da.copy()
        out.attrs["units"] = "mm_per_timestep"
        return out

    try:
        sample = float(da.isel(time=0).mean(skipna=True).values)
        if np.isfinite(sample) and sample < 0.05:
            out = da * 86400.0
            out.attrs["units"] = "mm_per_day"
            return out
    except Exception:
        pass

    out = da.copy()
    out.attrs["units"] = "mm_per_timestep"
    return out


# ============================================================
# 4. BUSCAR ARCHIVOS HIGHRES
# ============================================================

def is_precip_file(path):
    name = path.name.lower()
    full = str(path).lower()

    return (
        name.startswith("pr_") or
        "_pr_" in name or
        "precip" in name or
        "precipitation" in name or
        "/pr/" in full or
        "\\pr\\" in full
    )


def is_output_or_auxiliary_file(path):
    text = str(path).lower()

    bad_keys = [
        "salida_precipitacion",
        "maps_",
        "netcdf",
        "hatching",
        "summary",
        "scale",
        ".png",
        ".csv"
    ]

    return any(k in text for k in bad_keys)


def extract_highres_model_name(path, base_dir):
    """
    Intenta extraer el modelo HighRes.
    Sirve para dos estructuras:

    1) base/MODELO/*.nc
    2) base/*.nc con nombre tipo:
       pr_day_HadGEM3-GC31-LL_highres-future_...
    """
    path = Path(path)
    base_dir = Path(base_dir)

    try:
        rel = path.relative_to(base_dir)
        parts = rel.parts

        if len(parts) > 1:
            first = parts[0]

            generic = {
                "pr", "Pr", "PR",
                "RESAMPLE_FILES",
                "resample_files",
                "Regrid_005_conservative",
                "regrid_a_005_conservativo",
                "regrid_005_conservative"
            }

            if first not in generic:
                return first

    except Exception:
        pass

    name = path.name

    patterns = [
        r"pr_day_(.*?)_(historical|hist-1950|highres-future|highresSST-present|highresSST-future|future|ssp|rcp)",
        r"pr_(.*?)_(historical|hist-1950|highres-future|highresSST-present|highresSST-future|future|ssp|rcp)",
    ]

    for pat in patterns:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            return m.group(1)

    return path.parent.name


def prefer_regrid_if_available(files):
    if not PREFER_REGRID_FILES:
        return files

    regrid_keys = [
        "regrid",
        "resample",
        "005",
        "conservative",
        "conservativo"
    ]

    regrid_files = [
        f for f in files
        if any(k in str(f).lower() for k in regrid_keys)
    ]

    if len(regrid_files) > 0:
        return sorted(regrid_files)

    return sorted(files)


def find_highres_pr_files_by_model(base_dir, period_label):
    """
    Busca archivos HighRes de precipitación en una carpeta histórica o futura.

    Puede estar como:
      base/MODELO/*.nc
      base/MODELO/.../*.nc
      base/*.nc
    """
    base_dir = Path(base_dir)

    if not base_dir.exists():
        raise FileNotFoundError(f"No existe la carpeta: {base_dir}")

    all_nc = sorted(
        list(base_dir.rglob("*.nc")) +
        list(base_dir.rglob("*.nc4"))
    )

    all_nc = [
        f for f in all_nc
        if not is_output_or_auxiliary_file(f)
    ]

    pr_files = [
        f for f in all_nc
        if is_precip_file(f)
    ]

    # Si no detecta por nombre, usa todos los NetCDF como respaldo.
    if len(pr_files) == 0:
        pr_files = all_nc

    by_model = {}

    for f in pr_files:
        model = extract_highres_model_name(f, base_dir)

        if model not in by_model:
            by_model[model] = []

        by_model[model].append(f)

    for model in list(by_model.keys()):
        by_model[model] = prefer_regrid_if_available(by_model[model])

    print("\n====================================================")
    print(f"ARCHIVOS HIGHRES {period_label.upper()}")
    print("====================================================")
    print(f"Ruta: {base_dir}")
    print(f"Modelos detectados: {len(by_model)}")

    for model, files in sorted(by_model.items()):
        print(f" - {model}: {len(files)} archivos")
        if len(files) > 0:
            print(f"   Ejemplo: {files[0].name}")

    print("====================================================\n")

    return by_model


# ============================================================
# 5. MÁSCARA CHILE
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
    global _CHILE_GEOM

    if shapefile is None:
        print("No se entregó shapefile. No se aplicará máscara.")
        return None

    if gpd is None:
        print("Advertencia: geopandas no está instalado. No se aplicará máscara.")
        return None

    shapefile = Path(shapefile)

    if not shapefile.exists():
        print(f"Advertencia: no existe shapefile: {shapefile}")
        return None

    print("Cargando shapefile de Chile...")

    try:
        gdf = gpd.read_file(shapefile)
    except Exception as e:
        print("Advertencia: no se pudo leer el shapefile.")
        print(f"Motivo: {e}")
        return None

    try:
        if gdf.crs is not None:
            gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        pass

    gdf = gdf[gdf.geometry.notnull()].copy()

    cleaned_polygons = []

    print("Reparando geometrías de Chile...")

    for geom in gdf.geometry:
        fixed = fix_geometry_safe(geom)
        parts = keep_polygonal_parts(fixed)
        cleaned_polygons.extend(parts)

    if len(cleaned_polygons) == 0:
        print("Advertencia: no quedaron polígonos válidos.")
        return None

    try:
        from shapely.ops import unary_union

        geom_union = unary_union(cleaned_polygons)
        geom_union = fix_geometry_safe(geom_union)

        if geom_union is not None:
            _CHILE_GEOM = geom_union
            print("Máscara de Chile cargada con geometría unida.")
            return geom_union

    except Exception as e:
        print("Advertencia: no se pudieron unir geometrías.")
        print(f"Motivo: {e}")
        print("Se usará lista de polígonos separados.")

    _CHILE_GEOM = cleaned_polygons
    print("Máscara de Chile cargada con polígonos separados.")

    return cleaned_polygons


def get_xy_for_dataarray(da):
    da = da.squeeze()

    if "lon" not in da.coords or "lat" not in da.coords:
        raise ValueError("No se encontraron coordenadas lat/lon.")

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
        print("Construyendo máscara de Chile para esta grilla...")
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
# 6. CAPAS MAPA
# ============================================================

def load_shape_for_plot(shapefile):
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
    if gpd is None:
        print("Advertencia: geopandas no está disponible. No se dibujará fondo regional.")
        return None

    world = None

    try:
        import cartopy.io.shapereader as shpreader

        print("Cargando países desde Natural Earth con Cartopy...")

        shpfilename = shpreader.natural_earth(
            resolution="50m",
            category="cultural",
            name="admin_0_countries"
        )

        world = gpd.read_file(shpfilename)

    except Exception as e:
        print("Advertencia: no se pudo cargar Natural Earth con Cartopy.")
        print("Instala cartopy con:")
        print("conda install -c conda-forge cartopy")
        print(f"Motivo: {e}")
        world = None

    if world is None:
        print("No se dibujarán países vecinos en gris.")
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

    print(f"Capa regional Natural Earth cargada: {len(world)} geometrías")

    return world


# ============================================================
# 7. CÁLCULOS
# ============================================================

def compute_annual_and_period_total(
    files,
    model,
    period_label,
    start_year,
    end_year,
    chile_geom=None
):
    yearly_accumulator = {}

    for f in files:
        print(f"      Leyendo anual: {f.name}")

        try:
            ds = open_dataset_safe(f)
        except Exception as e:
            raise RuntimeError(f"No se pudo abrir {f}: {e}")

        try:
            var_name = select_variable(ds, PR_VAR_CANDIDATES)

            da = ds[var_name]
            da = standardize_coords(da)
            da = da.squeeze(drop=True)
            da = clean_for_concat(da)

            if "time" not in da.dims:
                raise ValueError(f"{f.name} no tiene dimensión time.")

            da_mm = convert_precip_to_mm_per_timestep(da)

            years = da_mm["time"].dt.year.values.astype(int)
            unique_years = sorted(np.unique(years))

            for yy in unique_years:
                if yy < start_year or yy > end_year:
                    continue

                idx = np.where(years == yy)[0]

                if len(idx) == 0:
                    continue

                print(f"         Acumulando año {yy}")

                sub = da_mm.isel(time=idx)

                # Evita crear máscaras booleanas gigantes
                annual_total = sub.sum(dim="time", skipna=False)
                annual_total = annual_total.load()
                annual_total = clean_for_concat(annual_total)

                if yy not in yearly_accumulator:
                    yearly_accumulator[yy] = annual_total
                else:
                    old = yearly_accumulator[yy]
                    old, annual_total = xr.align(
                        old,
                        annual_total,
                        join="inner"
                    )
                    yearly_accumulator[yy] = old + annual_total

        except Exception as e:
            try:
                ds.close()
            except Exception:
                pass

            raise RuntimeError(f"Falló archivo {f.name}: {e}")

        try:
            ds.close()
        except Exception:
            pass

    if not yearly_accumulator:
        return None, []

    annual_maps = []
    annual_years = []
    annual_rows = []

    for yy in sorted(yearly_accumulator.keys()):
        annual_total = yearly_accumulator[yy]
        annual_total = clean_for_concat(annual_total)
        annual_total = apply_chile_mask(annual_total, chile_geom)

        annual_total.name = f"pr_{period_label}_{model}_{yy}"
        annual_total.attrs["units"] = "mm/year"

        annual_maps.append(annual_total)
        annual_years.append(int(yy))

        annual_rows.append({
            "time_group": "annual",
            "variable": "pr",
            "model": model,
            "period": period_label,
            "year": int(yy),
            "season": "",
            "spatial_mean_precip_mm": spatial_mean(annual_total),
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

    period_mean.name = f"pr_{period_label}_{model}_annual_mean"
    period_mean.attrs["units"] = "mm/year"

    return period_mean, annual_rows


def compute_seasonal_period_totals(
    files,
    model,
    period_label,
    start_year,
    end_year,
    chile_geom=None
):
    seasonal_accumulator = {}

    for season_key in SEASONS.keys():
        seasonal_accumulator[season_key] = {}

    for f in files:
        print(f"      Leyendo estacional: {f.name}")

        try:
            ds = open_dataset_safe(f)
        except Exception as e:
            raise RuntimeError(f"No se pudo abrir {f}: {e}")

        try:
            var_name = select_variable(ds, PR_VAR_CANDIDATES)

            da = ds[var_name]
            da = standardize_coords(da)
            da = da.squeeze(drop=True)
            da = clean_for_concat(da)

            if "time" not in da.dims:
                raise ValueError(f"{f.name} no tiene dimensión time.")

            da_mm = convert_precip_to_mm_per_timestep(da)

            years = da_mm["time"].dt.year.values.astype(int)
            months = da_mm["time"].dt.month.values.astype(int)

            for season_key, season_info in SEASONS.items():
                season_months = np.array(season_info["months"])
                season_years = years.copy()

                if season_key == "verano_DJF":
                    season_years = np.where(months == 12, years + 1, years)

                mask_time = (
                    np.isin(months, season_months) &
                    (season_years >= start_year) &
                    (season_years <= end_year)
                )

                valid_season_years = sorted(np.unique(season_years[mask_time]))

                for sy in valid_season_years:
                    idx = np.where(mask_time & (season_years == sy))[0]

                    if len(idx) == 0:
                        continue

                    print(f"         Acumulando {season_key} {int(sy)}")

                    sub = da_mm.isel(time=idx)

                    # Evita crear máscaras booleanas gigantes
                    season_total = sub.sum(dim="time", skipna=False)
                    season_total = season_total.load()
                    season_total = clean_for_concat(season_total)

                    months_present = set(months[idx].astype(int).tolist())
                    sy = int(sy)

                    if sy not in seasonal_accumulator[season_key]:
                        seasonal_accumulator[season_key][sy] = {
                            "total": season_total,
                            "months_present": months_present,
                        }
                    else:
                        old = seasonal_accumulator[season_key][sy]["total"]
                        old, season_total = xr.align(
                            old,
                            season_total,
                            join="inner"
                        )

                        seasonal_accumulator[season_key][sy]["total"] = old + season_total
                        seasonal_accumulator[season_key][sy]["months_present"].update(months_present)

        except Exception as e:
            try:
                ds.close()
            except Exception:
                pass

            raise RuntimeError(f"Falló archivo estacional {f.name}: {e}")

        try:
            ds.close()
        except Exception:
            pass

    seasonal_means = {}
    seasonal_rows = []

    for season_key, season_info in SEASONS.items():
        acc = seasonal_accumulator[season_key]

        if not acc:
            seasonal_means[season_key] = None
            continue

        required_months = set(season_info["months"])

        season_maps = []
        season_years_out = []

        for sy in sorted(acc.keys()):
            months_present = acc[sy]["months_present"]

            if not required_months.issubset(months_present):
                print(
                    f"      Omitiendo {season_key} {sy} por estación incompleta. "
                    f"Meses presentes: {sorted(months_present)}"
                )
                continue

            season_total = acc[sy]["total"]
            season_total = clean_for_concat(season_total)
            season_total = apply_chile_mask(season_total, chile_geom)

            season_total.name = f"pr_{period_label}_{model}_{season_key}_{sy}"
            season_total.attrs["units"] = "mm/season"

            season_maps.append(season_total)
            season_years_out.append(int(sy))

            seasonal_rows.append({
                "time_group": "seasonal",
                "variable": "pr",
                "model": model,
                "period": period_label,
                "year": int(sy),
                "season": season_key,
                "spatial_mean_precip_mm": spatial_mean(season_total),
            })

        if len(season_maps) == 0:
            seasonal_means[season_key] = None
            continue

        season_stack = xr.concat(
            season_maps,
            dim=xr.DataArray(
                season_years_out,
                dims="year",
                name="year"
            ),
            join="inner",
            coords="minimal",
            compat="override"
        )

        period_mean = season_stack.mean(dim="year", skipna=True)
        period_mean = clean_for_concat(period_mean)
        period_mean = apply_chile_mask(period_mean, chile_geom)

        period_mean.name = f"pr_{period_label}_{model}_{season_key}_mean"
        period_mean.attrs["units"] = "mm/season"

        seasonal_means[season_key] = period_mean

    return seasonal_means, seasonal_rows


def compute_percent_change(delta_mm, hist_map, min_hist_for_pct=1.0):
    pct = 100.0 * delta_mm / hist_map.where(np.abs(hist_map) >= min_hist_for_pct)
    pct = clean_for_concat(pct)
    pct.attrs["units"] = "%"
    return pct


def add_summary_row(
    summary_rows,
    time_group,
    season,
    model,
    hist_map,
    fut_map,
    delta_mm_map,
    delta_pct_map,
    n_hist,
    n_fut
):
    summary_rows.append({
        "time_group": time_group,
        "season": season,
        "variable": "pr",
        "model": model,
        "n_years_historical": int(n_hist),
        "n_years_future": int(n_fut),
        "historical_spatial_mean_mm": spatial_mean(hist_map),
        "future_spatial_mean_mm": spatial_mean(fut_map),
        "delta_spatial_mean_mm": spatial_mean(delta_mm_map),
        "delta_spatial_mean_pct": spatial_mean(delta_pct_map),
    })


# ============================================================
# 8. COLORES Y MAPAS
# ============================================================

def make_precip_cmap():
    colors = [
        (0.00, "#7F0000"),
        (0.18, "#D7301F"),
        (0.35, "#FC8D59"),
        (0.50, "#FFFFFF"),
        (0.65, "#9ECAE1"),
        (0.82, "#3182BD"),
        (1.00, "#08306B"),
    ]

    return LinearSegmentedColormap.from_list(
        "precip_delta_cmap",
        colors,
        N=256
    )


def save_scale_png(output_path, vmax, label_text):
    cmap = make_precip_cmap()

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax
    )

    fig, ax = plt.subplots(figsize=(2.0, 6.0))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        extend="both"
    )

    cbar.set_label(
        label_text,
        fontsize=11,
        rotation=90,
        labelpad=12
    )

    if vmax == PR_MM_MAX:
        ticks = [-500, -250, 0, 250, 500]
    else:
        ticks = [-30, -20, -10, 0, 10, 20, 30]

    cbar.set_ticks(ticks)
    cbar.ax.tick_params(labelsize=10)

    ax.remove()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        transparent=False
    )

    plt.close(fig)


def get_xy_for_plot(da):
    da = da.squeeze()

    if "lat" in da.dims and "lon" in da.dims:
        da = da.transpose("lat", "lon")

    if "lon" not in da.coords or "lat" not in da.coords:
        raise ValueError("No se encontraron coordenadas lat/lon para graficar.")

    lon = da["lon"]
    lat = da["lat"]

    if lon.ndim == 1 and lat.ndim == 1:
        x, y = np.meshgrid(lon.values, lat.values)
    else:
        x, y = lon.values, lat.values

    z = da.values

    return x, y, z


def plot_precip_map(
    da,
    output_path,
    vmax,
    gdf_chile=None,
    gdf_region=None,
    agreement_mask=None
):
    cmap = make_precip_cmap()

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax
    )

    x, y, z = get_xy_for_plot(da)

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE)
    ax.set_facecolor(OCEAN_FACE)

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
            print("Advertencia: no se pudo dibujar capa regional.")
            print(f"Motivo: {e}")

    ax.pcolormesh(
        x,
        y,
        z,
        cmap=cmap,
        norm=norm,
        shading="auto",
        zorder=3
    )

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

    if gdf_chile is not None:
        try:
            gdf_chile.boundary.plot(
                ax=ax,
                linewidth=CHILE_BORDER_WIDTH,
                color=CHILE_BORDER_COLOR,
                zorder=5
            )
        except Exception as e:
            print("Advertencia: no se pudo dibujar borde de Chile.")
            print(f"Motivo: {e}")

    lon_min, lon_max, lat_min, lat_max = REGION_EXTENT
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_title("")
    ax.set_xlabel("Longitude", fontsize=11)
    ax.set_ylabel("Latitude", fontsize=11)
    ax.tick_params(axis="both", labelsize=10)

    fig.tight_layout()

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)


# ============================================================
# 9. MULTIMODELO
# ============================================================

def process_multimodel_group(
    deltas_dict,
    group_key,
    group_label,
    chile_geom,
    gdf_chile,
    gdf_region,
    threshold_rows,
    included_rows,
    failed_models
):
    for field_key, vmax, units_label in [
        ("delta_mm", PR_MM_MAX, "Precipitation change (mm)"),
        ("delta_pct", PR_PCT_MAX, "Precipitation change (%)"),
    ]:

        print("\n====================================================")
        print(f"MULTIMODELO HIGHRES PR: {field_key} - {group_key}")
        print("====================================================")

        if len(deltas_dict[group_key][field_key]) == 0:
            print(f"No hay modelos disponibles para {field_key} - {group_key}. Se omite.")
            continue

        models = sorted(deltas_dict[group_key][field_key].keys())

        included_rows.append({
            "group_key": group_key,
            "group_label": group_label,
            "field": field_key,
            "n_models_included": len(models),
            "models": "; ".join(models),
        })

        print(f"MODELOS INCLUIDOS EN MULTIMODELO {field_key} - {group_key}: {len(models)}")
        for m in models:
            print(f" - {m}")

        try:
            delta_list = []

            for m in models:
                da_model = deltas_dict[group_key][field_key][m]
                da_model = clean_for_concat(da_model)
                da_model = apply_chile_mask(da_model, chile_geom)
                delta_list.append(da_model)

            aligned = xr.align(
                *delta_list,
                join="inner"
            )

            aligned = [
                apply_chile_mask(clean_for_concat(a), chile_geom)
                for a in aligned
            ]

            stack = xr.concat(
                aligned,
                dim=xr.DataArray(
                    models,
                    dims="model",
                    name="model"
                ),
                coords="minimal",
                compat="override",
                join="inner"
            )

            stack.name = f"highres_pr_{field_key}_{group_key}_by_model"

            multimodel_mean = stack.mean(dim="model", skipna=True)
            multimodel_mean = clean_for_concat(multimodel_mean)
            multimodel_mean = apply_chile_mask(multimodel_mean, chile_geom)
            multimodel_mean.name = f"highres_pr_{field_key}_{group_key}_multimodel"

            n_models = len(models)
            threshold = agreement_threshold(n_models, AGREEMENT_PCT)

            n_positive = (stack > 0).sum(dim="model")
            n_negative = (stack < 0).sum(dim="model")

            agreement_mask = (n_positive >= threshold) | (n_negative >= threshold)
            agreement_mask = clean_for_concat(agreement_mask)
            agreement_mask = apply_chile_mask(agreement_mask, chile_geom)
            agreement_mask.name = f"highres_pr_{field_key}_{group_key}_agreement_hatching"

            actual_pct = int(np.ceil((threshold / n_models) * 100))

            print(f"Número de modelos: {n_models}")
            print(f"70% de modelos: {n_models * AGREEMENT_PCT}")
            print(f"Umbral usado para achurado: {threshold}")
            print(f"Porcentaje real equivalente: ≥{actual_pct}%")

            threshold_rows.append({
                "time_group": group_key,
                "time_group_label": group_label,
                "field": field_key,
                "n_models_used": n_models,
                "target_70_pct_value": n_models * AGREEMENT_PCT,
                "hatching_threshold_used": threshold,
                "actual_equivalent_percent": actual_pct,
                "rule": "standard rounding: 4.2->4, 4.6->5, 4.5->5",
            })

            nc_out = OUT_DIR / "netcdf"
            nc_out.mkdir(parents=True, exist_ok=True)

            stack_nc = nc_out / f"highres_pr_{field_key}_{group_key}_by_model.nc"
            multimodel_nc = nc_out / f"highres_pr_{field_key}_{group_key}_multimodel.nc"
            hatching_nc = nc_out / f"highres_pr_{field_key}_{group_key}_agreement_hatching.nc"

            stack.to_netcdf(stack_nc)
            multimodel_mean.to_netcdf(multimodel_nc)
            agreement_mask.fillna(0).astype("int8").to_netcdf(hatching_nc)

            print("\nNetCDF generados:")
            print(f" - {stack_nc}")
            print(f" - {multimodel_nc}")
            print(f" - {hatching_nc}")

            maps_root = "maps_mm" if field_key == "delta_mm" else "maps_percent"
            group_out = OUT_DIR / maps_root / group_key
            group_out.mkdir(parents=True, exist_ok=True)

            if MAKE_MODEL_MAPS:
                print("\nGenerando mapas individuales por modelo...")

                for model, da_delta in zip(models, aligned):
                    output_png = group_out / f"highres_pr_{field_key}_{group_key}_{safe_name(model)}.png"

                    plot_precip_map(
                        da=da_delta,
                        output_path=output_png,
                        vmax=vmax,
                        gdf_chile=gdf_chile,
                        gdf_region=gdf_region,
                        agreement_mask=None
                    )

                print(f"Mapas individuales guardados en: {group_out}")

            if MAKE_MULTIMODEL_MAPS:
                print("\nGenerando mapa multimodelo con achurado...")

                output_png = group_out / f"highres_pr_{field_key}_{group_key}_multimodel_hatched.png"

                plot_precip_map(
                    da=multimodel_mean,
                    output_path=output_png,
                    vmax=vmax,
                    gdf_chile=gdf_chile,
                    gdf_region=gdf_region,
                    agreement_mask=agreement_mask
                )

                print("Mapa multimodelo con achurado guardado en:")
                print(f" - {output_png}")

                output_png_no_hatch = group_out / f"highres_pr_{field_key}_{group_key}_multimodel_no_hatch.png"

                plot_precip_map(
                    da=multimodel_mean,
                    output_path=output_png_no_hatch,
                    vmax=vmax,
                    gdf_chile=gdf_chile,
                    gdf_region=gdf_region,
                    agreement_mask=None
                )

                print("Mapa multimodelo sin achurado guardado en:")
                print(f" - {output_png_no_hatch}")

        except Exception as e:
            register_failure(
                failed_models,
                model=f"MULTIMODEL_{field_key}_{group_key}",
                step="multimodel",
                reason=e
            )
            print(f"ERROR en multimodelo {field_key} - {group_key}: {e}")


# ============================================================
# 10. RESUMEN FINAL
# ============================================================

def print_final_summary(processed_models, skipped_models, failed_models, included_rows):
    print("\n====================================================")
    print("RESUMEN FINAL HIGHRES PRECIPITACIÓN")
    print("====================================================")

    print(f"\nModelos procesados con al menos un delta válido: {len(processed_models)}")
    for m in processed_models:
        print(f"  OK: {m}")

    print(f"\nModelos omitidos: {len(skipped_models)}")
    for item in skipped_models:
        print(f"  OMITIDO: {item['model']} | Motivo: {item['reason']}")

    print(f"\nModelos / procesos con error: {len(failed_models)}")
    for item in failed_models:
        print(
            f"  ERROR: {item['model']} | "
            f"Paso: {item['step']} | "
            f"Motivo: {item['reason']}"
        )

    print("\n====================================================")
    print("MODELOS INCLUIDOS EN CADA MULTIMODELO")
    print("====================================================")

    if len(included_rows) == 0:
        print("No se generó ningún multimodelo.")
    else:
        for row in included_rows:
            print(
                f"{row['field']} | {row['group_key']} "
                f"({row['group_label']}): "
                f"{row['n_models_included']} modelos"
            )
            print(f"  {row['models']}")

    unique_models = set()
    for row in included_rows:
        for m in row["models"].split("; "):
            if m.strip():
                unique_models.add(m.strip())

    print("\n====================================================")
    print(f"TOTAL ÚNICO DE MODELOS INCLUIDOS EN ALGÚN MULTIMODELO: {len(unique_models)}")
    print("====================================================")
    for m in sorted(unique_models):
        print(f"  {m}")

    print("====================================================\n")


# ============================================================
# 11. MAIN
# ============================================================

def main():
    print("====================================================")
    print("HIGHRES PRECIPITATION CHANGE MAPS")
    print("Annual + seasonal")
    print("Delta = Future - Historical")
    print("Units: mm and %")
    print("====================================================")

    print("\nRutas:")
    print(f"Histórico: {HIGHRES_HIST_DIR}")
    print(f"Futuro:    {HIGHRES_FUT_DIR}")
    print(f"Shapefile: {SHAPEFILE}")
    print(f"Salida:    {OUT_DIR}")

    print("\nPeriodos:")
    print(f"Histórico: {HIST_LABEL}")
    print(f"Futuro:    {FUT_LABEL}")

    processed_models = []
    skipped_models = []
    failed_models = []
    included_rows = []

    chile_geom = load_chile_geometry(SHAPEFILE)
    gdf_chile = load_shape_for_plot(SHAPEFILE)
    gdf_region = load_regional_layer()

    # Escalas aparte
    scale_mm_png = OUT_DIR / "highres_precip_scale_mm_500.png"
    scale_pct_png = OUT_DIR / "highres_precip_scale_percent_30.png"

    save_scale_png(scale_mm_png, PR_MM_MAX, "Precipitation change (mm)")
    save_scale_png(scale_pct_png, PR_PCT_MAX, "Precipitation change (%)")

    print("\nEscalas de colores guardadas en:")
    print(f" - {scale_mm_png}")
    print(f" - {scale_pct_png}")

    hist_models = find_highres_pr_files_by_model(HIGHRES_HIST_DIR, "historical")
    fut_models = find_highres_pr_files_by_model(HIGHRES_FUT_DIR, "future")

    all_models = sorted(
        set(hist_models.keys()) |
        set(fut_models.keys())
    )

    print("\n====================================================")
    print("DIAGNÓSTICO HIGHRES")
    print("====================================================")
    print(f"Modelos detectados en histórico: {len(hist_models)}")
    print(f"Modelos detectados en futuro:    {len(fut_models)}")
    print(f"Modelos totales detectados:      {len(all_models)}")

    for model in all_models:
        print(
            f"{model}: "
            f"hist={model in hist_models}, "
            f"fut={model in fut_models}"
        )

    print("====================================================\n")

    if not all_models:
        raise ValueError("No se encontraron modelos HighRes de precipitación.")

    summary_rows = []
    rows_all = []

    group_keys = ["annual"] + list(SEASONS.keys())

    deltas_by_group = {}
    for group_key in group_keys:
        deltas_by_group[group_key] = {
            "delta_mm": {},
            "delta_pct": {},
        }

    for model in all_models:
        print("\n----------------------------------------------------")
        print(f"Procesando modelo HighRes: {model}")
        print("----------------------------------------------------")

        if model not in hist_models:
            skipped_models.append({
                "model": model,
                "reason": "No tiene archivos históricos."
            })
            print(f"{model}: omitido porque no tiene histórico.")
            continue

        if model not in fut_models:
            skipped_models.append({
                "model": model,
                "reason": "No tiene archivos futuros."
            })
            print(f"{model}: omitido porque no tiene futuro.")
            continue

        hist_files = hist_models[model]
        fut_files = fut_models[model]

        try:
            print("   PR histórico anual")

            pr_hist_map, pr_hist_rows = compute_annual_and_period_total(
                files=hist_files,
                model=model,
                period_label="historical",
                start_year=HIST_START_YEAR,
                end_year=HIST_END_YEAR,
                chile_geom=chile_geom
            )

            print("   PR futuro anual")

            pr_fut_map, pr_fut_rows = compute_annual_and_period_total(
                files=fut_files,
                model=model,
                period_label="future",
                start_year=FUT_START_YEAR,
                end_year=FUT_END_YEAR,
                chile_geom=chile_geom
            )

            print("   PR histórico estacional")

            pr_hist_seasonal, pr_hist_seasonal_rows = compute_seasonal_period_totals(
                files=hist_files,
                model=model,
                period_label="historical",
                start_year=HIST_START_YEAR,
                end_year=HIST_END_YEAR,
                chile_geom=chile_geom
            )

            print("   PR futuro estacional")

            pr_fut_seasonal, pr_fut_seasonal_rows = compute_seasonal_period_totals(
                files=fut_files,
                model=model,
                period_label="future",
                start_year=FUT_START_YEAR,
                end_year=FUT_END_YEAR,
                chile_geom=chile_geom
            )

        except Exception as e:
            register_failure(
                failed_models,
                model=model,
                step="lectura/calculo_periodos",
                reason=e
            )
            print(f"ERROR en {model}: {e}")
            continue

        rows_all.extend(pr_hist_rows)
        rows_all.extend(pr_fut_rows)
        rows_all.extend(pr_hist_seasonal_rows)
        rows_all.extend(pr_fut_seasonal_rows)

        model_has_any_delta = False

        # Delta anual
        try:
            if pr_hist_map is not None and pr_fut_map is not None:
                pr_hist_map, pr_fut_map = xr.align(
                    pr_hist_map,
                    pr_fut_map,
                    join="inner"
                )

                delta_mm = pr_fut_map - pr_hist_map
                delta_mm = clean_for_concat(delta_mm)
                delta_mm = apply_chile_mask(delta_mm, chile_geom)
                delta_mm.name = f"pr_delta_mm_annual_{model}"
                delta_mm.attrs["units"] = "mm/year"

                delta_pct = compute_percent_change(delta_mm, pr_hist_map)
                delta_pct = apply_chile_mask(delta_pct, chile_geom)
                delta_pct.name = f"pr_delta_pct_annual_{model}"
                delta_pct.attrs["units"] = "%"

                deltas_by_group["annual"]["delta_mm"][model] = delta_mm
                deltas_by_group["annual"]["delta_pct"][model] = delta_pct

                add_summary_row(
                    summary_rows,
                    "annual",
                    "",
                    model,
                    pr_hist_map,
                    pr_fut_map,
                    delta_mm,
                    delta_pct,
                    len({r["year"] for r in pr_hist_rows}),
                    len({r["year"] for r in pr_fut_rows}),
                )

                model_has_any_delta = True
            else:
                skipped_models.append({
                    "model": model,
                    "reason": "No tuvo mapa anual histórico o futuro válido."
                })

        except Exception as e:
            register_failure(
                failed_models,
                model=model,
                step="delta_anual",
                reason=e
            )
            print(f"ERROR calculando delta anual en {model}: {e}")

        # Delta estacional
        for season_key in SEASONS.keys():
            try:
                hmap = pr_hist_seasonal.get(season_key)
                fmap = pr_fut_seasonal.get(season_key)

                if hmap is None or fmap is None:
                    print(f"{model}: no se calcula {season_key}, falta histórico o futuro.")
                    continue

                hmap, fmap = xr.align(hmap, fmap, join="inner")

                delta_mm = fmap - hmap
                delta_mm = clean_for_concat(delta_mm)
                delta_mm = apply_chile_mask(delta_mm, chile_geom)
                delta_mm.name = f"pr_delta_mm_{season_key}_{model}"
                delta_mm.attrs["units"] = "mm/season"

                delta_pct = compute_percent_change(delta_mm, hmap)
                delta_pct = apply_chile_mask(delta_pct, chile_geom)
                delta_pct.name = f"pr_delta_pct_{season_key}_{model}"
                delta_pct.attrs["units"] = "%"

                deltas_by_group[season_key]["delta_mm"][model] = delta_mm
                deltas_by_group[season_key]["delta_pct"][model] = delta_pct

                add_summary_row(
                    summary_rows,
                    "seasonal",
                    season_key,
                    model,
                    hmap,
                    fmap,
                    delta_mm,
                    delta_pct,
                    len({r["year"] for r in pr_hist_seasonal_rows if r["season"] == season_key}),
                    len({r["year"] for r in pr_fut_seasonal_rows if r["season"] == season_key}),
                )

                model_has_any_delta = True

            except Exception as e:
                register_failure(
                    failed_models,
                    model=model,
                    step=f"delta_estacional_{season_key}",
                    reason=e
                )
                print(f"ERROR calculando {season_key} en {model}: {e}")

        if model_has_any_delta:
            processed_models.append(model)
        else:
            skipped_models.append({
                "model": model,
                "reason": "No se pudo generar ningún delta válido."
            })

    # CSV base
    summary_df = pd.DataFrame(summary_rows)
    rows_df = pd.DataFrame(rows_all)
    failed_df = pd.DataFrame(failed_models)
    skipped_df = pd.DataFrame(skipped_models)

    summary_csv = OUT_DIR / "highres_precipitation_summary_annual_seasonal.csv"
    rows_csv = OUT_DIR / "highres_precipitation_annual_seasonal_by_model.csv"
    failed_csv = OUT_DIR / "highres_precipitation_failed_models.csv"
    skipped_csv = OUT_DIR / "highres_precipitation_skipped_models.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    rows_df.to_csv(rows_csv, index=False, encoding="utf-8-sig")
    failed_df.to_csv(failed_csv, index=False, encoding="utf-8-sig")
    skipped_df.to_csv(skipped_csv, index=False, encoding="utf-8-sig")

    print("\nCSV generados:")
    print(f" - {summary_csv}")
    print(f" - {rows_csv}")
    print(f" - {failed_csv}")
    print(f" - {skipped_csv}")

    # Multimodelo
    threshold_rows = []

    process_multimodel_group(
        deltas_dict=deltas_by_group,
        group_key="annual",
        group_label="Annual",
        chile_geom=chile_geom,
        gdf_chile=gdf_chile,
        gdf_region=gdf_region,
        threshold_rows=threshold_rows,
        included_rows=included_rows,
        failed_models=failed_models
    )

    for season_key, season_info in SEASONS.items():
        process_multimodel_group(
            deltas_dict=deltas_by_group,
            group_key=season_key,
            group_label=season_info["label"],
            chile_geom=chile_geom,
            gdf_chile=gdf_chile,
            gdf_region=gdf_region,
            threshold_rows=threshold_rows,
            included_rows=included_rows,
            failed_models=failed_models
        )

    # CSV achurado e incluidos
    if len(threshold_rows) > 0:
        threshold_df = pd.DataFrame(threshold_rows)
        threshold_csv = OUT_DIR / "highres_precipitation_hatching_threshold_annual_seasonal.csv"

        threshold_df.to_csv(
            threshold_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nCSV de umbral de achurado generado:")
        print(f" - {threshold_csv}")

    included_csv = OUT_DIR / "highres_precipitation_models_included_multimodel.csv"
    pd.DataFrame(included_rows).to_csv(
        included_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCSV de modelos incluidos en multimodelo:")
    print(f" - {included_csv}")

    # Actualizar errores por si falló multimodelo
    pd.DataFrame(failed_models).to_csv(
        failed_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print_final_summary(
        processed_models=processed_models,
        skipped_models=skipped_models,
        failed_models=failed_models,
        included_rows=included_rows
    )

    print("\n====================================================")
    print("PROCESO TERMINADO")
    print("====================================================")
    print(f"Carpeta de salida: {OUT_DIR}")


if __name__ == "__main__":
    main()