# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 10:49:44 2026

@author: catar
"""

# -*- coding: utf-8 -*-
"""
CMIP6 PRECIPITATION CHANGE MAPS - ANNUAL + SEASONAL

Calcula:
  - pr: cambio de precipitación

Entrega:
  - cambio absoluto: futuro - histórico, en mm/año
  - cambio porcentual: ((futuro - histórico) / histórico) * 100

Estructura esperada:

F:/CMIP6/pr_historica/
  MODELO/
    RESAMPLE_FILES/
      pr/
        Regrid_005_conservative/
          *.nc

F:/CMIP6/pr_futuro/
  MODELO/
    RESAMPLE_FILES/
      pr/
        Regrid_005_conservative/
          *.nc

Método:
  1. Calcula acumulado anual por modelo y periodo.
  2. Calcula acumulado estacional por modelo y periodo.
  3. Calcula promedio del periodo histórico.
  4. Calcula promedio del periodo futuro.
  5. Calcula:
       delta_mm = futuro - histórico
       delta_pct = (delta_mm / histórico) * 100
  6. Calcula multimodelo.
  7. Aplica achurado donde >=70% de modelos coinciden en signo.

Mapas:
  - Chile con datos.
  - Países vecinos en gris usando Natural Earth.
  - Océano blanco.
  - Máscara de Chile.
  - Sin título.
  - Ejes en inglés.
  - Sin barra de color al lado del mapa.
  - Escalas de color guardadas en PNG aparte.

Escalas:
  - mm/año:  -500 a 500
  - %:       -30 a 30

Convención de colores:
  - rojo = más seco / menos precipitación
  - azul = más húmedo / más precipitación
  - blanco = 0
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, Normalize
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

PR_FUT_DIR = Path(
    r"F:\CMIP5\Pr_futuro"
)

PR_HIST_DIR = Path(
    r"F:\CMIP6\Pr_hist"
)

SHAPEFILE = Path(
    r"C:/Users/catar/OneDrive - Universidad Técnica Federico Santa María/Universidad/2026/Seminario de Tesis I/Provincias/Provincias.shp"
)

OUT_DIR = Path(
    r"F:\CMIP6\salida_precipitacion"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Periodos
HIST_START_YEAR = 1980
HIST_END_YEAR   = 2010

FUT_START_YEAR  = 2080
FUT_END_YEAR    = 2099

HIST_LABEL = f"{HIST_START_YEAR}-{HIST_END_YEAR}"
FUT_LABEL  = f"{FUT_START_YEAR}-{FUT_END_YEAR}"

# Variables posibles dentro de los NetCDF
PR_VAR_CANDIDATES = ["pr", "PR", "precip", "precipitation"]

# Extensión del mapa
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

# Estaciones climáticas
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
    """
    Redondeo normal:
      4.2 -> 4
      4.5 -> 5
      4.6 -> 5
    """
    return int(np.floor(float(x) + 0.5))


def agreement_threshold(n_models, pct=0.70):
    return max(1, round_half_up(n_models * pct))


def safe_name(text):
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text))


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
    """
    Promedio espacial ponderado por cos(lat), ignorando NaN.
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
# 3. CONVERSIÓN DE PRECIPITACIÓN
# ============================================================

def convert_precip_to_mm_per_timestep(da):
    """
    Convierte precipitación a mm por paso temporal, listo para sumar.

    Casos comunes:
    - kg m-2 s-1  -> mm/s  -> se multiplica por 86400 para obtener mm/día
      (asumiendo datos diarios, como suele ocurrir en CMIP6 day)
    - mm/day o similares -> se usa tal cual
    - mm -> se usa tal cual
    """

    units = str(da.attrs.get("units", "")).lower().strip()

    rate_units = [
        "kg m-2 s-1",
        "kg m^-2 s^-1",
        "kg/m2/s",
        "kg m-2 sec-1",
        "mm s-1",
        "mm/s",
    ]

    day_units = [
        "mm/day",
        "mm d-1",
        "mm/day ",
        "kg m-2 day-1",
        "kg/m2/day",
        "mm per day",
    ]

    amount_units = [
        "mm",
        "kg m-2",
        "kg/m2",
    ]

    if units in rate_units or "s-1" in units or "/s" in units:
        out = da * 86400.0
        out.attrs["units"] = "mm_per_timestep"
        return out

    if units in day_units or "day" in units or "d-1" in units:
        out = da.copy()
        out.attrs["units"] = "mm_per_timestep"
        return out

    if units in amount_units:
        out = da.copy()
        out.attrs["units"] = "mm_per_timestep"
        return out

    # Heurística simple si no hay units claras
    try:
        sample = float(da.isel(time=0).mean(skipna=True).values)
        if np.isfinite(sample) and sample < 0.05:
            out = da * 86400.0
            out.attrs["units"] = "mm_per_timestep"
            return out
    except Exception:
        pass

    out = da.copy()
    out.attrs["units"] = "mm_per_timestep"
    return out


# ============================================================
# 4. DETECCIÓN DE ARCHIVOS CMIP6
# ============================================================

def find_cmip6_files_by_model(base_dir, variable):
    """
    Busca archivos CMIP6 de precipitación con estructura simple:

    base_dir/
      MODELO/
        *.nc
        *.nc4

    Es decir, NO busca:
      RESAMPLE_FILES/pr/Regrid_005_conservative

    Solo entra a la carpeta del modelo y busca NetCDF dentro de ella.
    """

    base_dir = Path(base_dir)

    if not base_dir.exists():
        print(f"Advertencia: no existe la carpeta base: {base_dir}")
        return {}

    by_model = {}

    model_dirs = sorted([
        p for p in base_dir.iterdir()
        if p.is_dir()
    ])

    print("\n====================================================")
    print(f"Buscando archivos de {variable} en carpetas por modelo:")
    print(base_dir)
    print("====================================================")

    for model_dir in model_dirs:

        model_name = model_dir.name

        # Busca archivos directamente dentro de la carpeta del modelo.
        # Si quieres buscar también en subcarpetas internas, cambia glob por rglob.
        files = sorted(
            list(model_dir.glob("*.nc")) +
            list(model_dir.glob("*.nc4"))
        )

        # Respaldo: si no encuentra directo, busca en subcarpetas,
        # pero SIN exigir RESAMPLE_FILES/Regrid.
        if len(files) == 0:
            files = sorted(
                list(model_dir.rglob("*.nc")) +
                list(model_dir.rglob("*.nc4"))
            )

        if len(files) == 0:
            print(f" - {model_name}: no hay archivos NetCDF.")
            continue

        # Filtro suave para quedarnos con precipitación si el nombre lo indica.
        pr_files = []

        for f in files:
            name = f.name.lower()

            if (
                name.startswith("pr_") or
                "_pr_" in name or
                "precip" in name or
                variable.lower() in name
            ):
                pr_files.append(f)

        # Si el filtro no encuentra nada, usa todos los NetCDF encontrados.
        # Esto sirve cuando los archivos no tienen 'pr' en el nombre,
        # pero la carpeta del modelo solo contiene precipitación.
        if len(pr_files) > 0:
            files = pr_files

        by_model[model_name] = files

        print(f" - {model_name}: {len(files)} archivos encontrados")

    return by_model

# ============================================================
# 5. MÁSCARA DE CHILE
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
# 6. SHAPEFILES / NATURAL EARTH
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
    """
    Carga países vecinos desde Natural Earth usando Cartopy.
    """
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
# 7. CÁLCULO ANUAL Y ESTACIONAL
# ============================================================

def compute_annual_and_period_total(
    files,
    var_candidates,
    model,
    period_label,
    start_year,
    end_year,
    chile_geom=None
):
    """
    Calcula acumulado anual y promedio del periodo (mm/año).
    """

    yearly_accumulator = {}

    for f in files:
        print(f"      Leyendo: {f.name}")

        try:
            ds = open_dataset_safe(f)
        except Exception as e:
            print(f"      Advertencia: no se pudo abrir {f.name}. Se omite.")
            print(f"      Motivo: {e}")
            continue

        with ds:
            try:
                var_name = select_variable(ds, var_candidates)

                da = ds[var_name]
                da = standardize_coords(da)
                da = da.squeeze(drop=True)
                da = clean_for_concat(da)

                if "time" not in da.dims:
                    print(f"      Advertencia: {f.name} no tiene dimensión time. Se omite.")
                    continue

                da_mm = convert_precip_to_mm_per_timestep(da)

                years = da_mm["time"].dt.year.values.astype(int)
                unique_years = sorted(np.unique(years))

                for yy in unique_years:

                    if yy < start_year or yy > end_year:
                        continue

                    idx = np.where(years == yy)[0]

                    if len(idx) == 0:
                        continue

                    print(f"         Acumulando precipitación del año {yy}")

                    sub = da_mm.isel(time=idx)
                    annual_total = sub.sum(dim="time", skipna=True).load()
                    annual_total = clean_for_concat(annual_total)

                    if yy not in yearly_accumulator:
                        yearly_accumulator[yy] = annual_total
                    else:
                        old = yearly_accumulator[yy]
                        old, annual_total = xr.align(old, annual_total, join="inner")
                        yearly_accumulator[yy] = old + annual_total

            except Exception as e:
                print(f"      Advertencia: falló archivo {f.name}. Se omite.")
                print(f"      Motivo: {e}")
                continue

    if not yearly_accumulator:
        print(
            f"      Advertencia: no hay años válidos para "
            f"{model} - {period_label} en {start_year}-{end_year}."
        )
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
    var_candidates,
    model,
    period_label,
    start_year,
    end_year,
    chile_geom=None
):
    """
    Calcula acumulado estacional por año y promedio del periodo.
    """

    seasonal_accumulator = {}
    for season_key in SEASONS.keys():
        seasonal_accumulator[season_key] = {}

    for f in files:
        print(f"      Leyendo estacional: {f.name}")

        try:
            ds = open_dataset_safe(f)
        except Exception as e:
            print(f"      Advertencia: no se pudo abrir {f.name}. Se omite.")
            print(f"      Motivo: {e}")
            continue

        with ds:
            try:
                var_name = select_variable(ds, var_candidates)

                da = ds[var_name]
                da = standardize_coords(da)
                da = da.squeeze(drop=True)
                da = clean_for_concat(da)

                if "time" not in da.dims:
                    print(f"      Advertencia: {f.name} no tiene dimensión time. Se omite.")
                    continue

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

                        print(f"         Acumulando {season_key} para {int(sy)}")

                        sub = da_mm.isel(time=idx)
                        season_total = sub.sum(dim="time", skipna=True).load()
                        season_total = clean_for_concat(season_total)

                        sy = int(sy)

                        if sy not in seasonal_accumulator[season_key]:
                            seasonal_accumulator[season_key][sy] = season_total
                        else:
                            old = seasonal_accumulator[season_key][sy]
                            old, season_total = xr.align(old, season_total, join="inner")
                            seasonal_accumulator[season_key][sy] = old + season_total

            except Exception as e:
                print(f"      Advertencia: falló archivo estacional {f.name}. Se omite.")
                print(f"      Motivo: {e}")
                continue

    seasonal_means = {}
    seasonal_rows = []

    for season_key in SEASONS.keys():

        acc = seasonal_accumulator[season_key]

        if not acc:
            print(
                f"      Advertencia: no hay datos válidos para "
                f"{model} - {period_label} - {season_key}."
            )
            seasonal_means[season_key] = None
            continue

        season_maps = []
        season_years_out = []

        for sy in sorted(acc.keys()):
            season_total = acc[sy]
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
    """
    Cambio porcentual = (delta / histórico) * 100
    Evita divisiones inestables donde histórico es muy pequeño.
    """
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
    """
    Escala divergente:
      negativo = rojo (más seco)
      0 = blanco
      positivo = azul (más húmedo)
    """
    colors = [
        (0.00, "#7F0000"),  # rojo oscuro
        (0.18, "#D7301F"),  # rojo
        (0.35, "#FC8D59"),  # naranja/rojo claro
        (0.50, "#FFFFFF"),  # cero
        (0.65, "#9ECAE1"),  # celeste
        (0.82, "#3182BD"),  # azul
        (1.00, "#08306B"),  # azul oscuro
    ]

    return LinearSegmentedColormap.from_list(
        "precip_delta_cmap",
        colors,
        N=256
    )


def save_scale_png(output_path, vmax, label_text):
    """
    Guarda una barra de color vertical en PNG aparte.
    """
    cmap = make_precip_cmap()
    norm = Normalize(vmin=-vmax, vmax=vmax)

    fig, ax = plt.subplots(figsize=(2.0, 6.0))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical"
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
    """
    Grafica mapa sin barra de color lateral.
    """
    cmap = make_precip_cmap()

    norm = TwoSlopeNorm(
        vmin=-vmax,
        vcenter=0,
        vmax=vmax
    )

    x, y, z = get_xy_for_plot(da)

    fig, ax = plt.subplots(figsize=MAP_FIGSIZE)
    ax.set_facecolor(OCEAN_FACE)

    # Países vecinos en gris
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

    # Datos de Chile
    ax.pcolormesh(
        x,
        y,
        z,
        cmap=cmap,
        norm=norm,
        shading="auto",
        zorder=3
    )

    # Achurado
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

    # Bordes de Chile
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
# 9. MULTIMODELO + ACHURADO
# ============================================================

def process_multimodel_group(
    deltas_dict,
    group_key,
    group_label,
    chile_geom,
    gdf_chile,
    gdf_region,
    threshold_rows
):
    """
    Procesa multimodelo para annual o para cada estación.
    Genera:
      - cambio absoluto mm/año
      - cambio porcentual %
    """

    for field_key, vmax, units_label in [
        ("delta_mm", PR_MM_MAX, "Precipitation change (mm/year)"),
        ("delta_pct", PR_PCT_MAX, "Precipitation change (%)"),
    ]:

        print("\n====================================================")
        print(f"MULTIMODELO CMIP6 PR: {field_key} - {group_key}")
        print("====================================================")

        if len(deltas_dict[group_key][field_key]) == 0:
            print(f"No hay modelos disponibles para {field_key} - {group_key}. Se omite.")
            continue

        models = sorted(deltas_dict[group_key][field_key].keys())

        print(f"Modelos usados para {field_key} - {group_key}: {len(models)}")
        for m in models:
            print(f" - {m}")

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

        stack.name = f"cmip6_pr_{field_key}_{group_key}_by_model"
        stack.attrs["description"] = (
            f"CMIP6 precipitation change by model. "
            f"Field: {field_key}. Time group: {group_label}. "
            f"Delta = future ({FUT_LABEL}) - historical ({HIST_LABEL})"
        )

        multimodel_mean = stack.mean(dim="model", skipna=True)
        multimodel_mean = clean_for_concat(multimodel_mean)
        multimodel_mean = apply_chile_mask(multimodel_mean, chile_geom)

        multimodel_mean.name = f"cmip6_pr_{field_key}_{group_key}_multimodel"

        n_models = len(models)
        threshold = agreement_threshold(n_models, AGREEMENT_PCT)

        n_positive = (stack > 0).sum(dim="model")
        n_negative = (stack < 0).sum(dim="model")

        agreement_mask = (n_positive >= threshold) | (n_negative >= threshold)
        agreement_mask = clean_for_concat(agreement_mask)
        agreement_mask = apply_chile_mask(agreement_mask, chile_geom)

        agreement_mask.name = f"cmip6_pr_{field_key}_{group_key}_agreement_hatching"

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

        # NetCDF
        nc_out = OUT_DIR / "netcdf"
        nc_out.mkdir(parents=True, exist_ok=True)

        stack_nc = nc_out / f"cmip6_pr_{field_key}_{group_key}_by_model.nc"
        multimodel_nc = nc_out / f"cmip6_pr_{field_key}_{group_key}_multimodel.nc"
        hatching_nc = nc_out / f"cmip6_pr_{field_key}_{group_key}_agreement_hatching.nc"

        stack.to_netcdf(stack_nc)
        multimodel_mean.to_netcdf(multimodel_nc)
        agreement_mask.fillna(0).astype("int8").to_netcdf(hatching_nc)

        print("\nNetCDF generados:")
        print(f" - {stack_nc}")
        print(f" - {multimodel_nc}")
        print(f" - {hatching_nc}")

        # Mapas
        maps_root = "maps_mm_year" if field_key == "delta_mm" else "maps_percent"
        group_out = OUT_DIR / maps_root / group_key
        group_out.mkdir(parents=True, exist_ok=True)

        if MAKE_MODEL_MAPS:
            print("\nGenerando mapas individuales por modelo...")

            for model, da_delta in zip(models, aligned):

                output_png = group_out / f"cmip6_pr_{field_key}_{group_key}_{safe_name(model)}.png"

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

            output_png = group_out / f"cmip6_pr_{field_key}_{group_key}_multimodel_hatched.png"

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

            output_png_no_hatch = group_out / f"cmip6_pr_{field_key}_{group_key}_multimodel_no_hatch.png"

            plot_precip_map(
                da=multimodel_mean,
                output_path=output_png_no_hatch,
                vmax=vmax,
                gdf_chile=gdf_chile,
                gdf_region=gdf_region,
                agreement_mask=None
            )

            print("Mapa multimodelo sin achurado guardado en:")
            print(f" - {output_png}")


# ============================================================
# 10. PROCESO PRINCIPAL
# ============================================================

def main():
    print("====================================================")
    print("CMIP6 PRECIPITATION CHANGE MAPS")
    print("Annual + seasonal")
    print("Delta = Future - Historical")
    print("Units: mm/year and %")
    print("====================================================")

    print("\nCarpetas:")
    print(f"Histórico: {PR_HIST_DIR}")
    print(f"Futuro:    {PR_FUT_DIR}")
    print(f"Salida:    {OUT_DIR}")

    print("\nPeriodos:")
    print(f"Histórico: {HIST_LABEL}")
    print(f"Futuro:    {FUT_LABEL}")

    chile_geom = load_chile_geometry(SHAPEFILE)
    gdf_chile = load_shape_for_plot(SHAPEFILE)
    gdf_region = load_regional_layer()

    # Guardar escalas aparte
    scale_mm_png = OUT_DIR / "cmip6_precip_scale_mm_year_500.png"
    scale_pct_png = OUT_DIR / "cmip6_precip_scale_percent_30.png"

    save_scale_png(scale_mm_png, PR_MM_MAX, "Precipitation change (mm/year)")
    save_scale_png(scale_pct_png, PR_PCT_MAX, "Precipitation change (%)")

    print("\nEscalas de colores guardadas en:")
    print(f" - {scale_mm_png}")
    print(f" - {scale_pct_png}")

    pr_hist_models = find_cmip6_files_by_model(PR_HIST_DIR, "pr")
    pr_fut_models  = find_cmip6_files_by_model(PR_FUT_DIR, "pr")

    print("\n====================================================")
    print("DIAGNÓSTICO DE DISPONIBILIDAD CMIP6")
    print("====================================================")

    all_models = sorted(
        set(pr_hist_models.keys()) |
        set(pr_fut_models.keys())
    )

    for model in all_models:
        print(
            f"{model}: "
            f"pr_hist={model in pr_hist_models}, "
            f"pr_fut={model in pr_fut_models}"
        )

    print("====================================================\n")

    if not all_models:
        raise ValueError("No se encontraron modelos CMIP6 de precipitación.")

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
        print(f"Procesando modelo: {model}")
        print("----------------------------------------------------")

        has_pr = (
            model in pr_hist_models and
            model in pr_fut_models
        )

        if not has_pr:
            print(f"{model}: no tiene pr completo. Se omite.")
            continue

        # ====================================================
        # PR ANUAL
        # ====================================================

        print("   PR histórico anual")

        pr_hist_map, pr_hist_rows = compute_annual_and_period_total(
            files=pr_hist_models[model],
            var_candidates=PR_VAR_CANDIDATES,
            model=model,
            period_label="historical",
            start_year=HIST_START_YEAR,
            end_year=HIST_END_YEAR,
            chile_geom=chile_geom
        )

        print("   PR futuro anual")

        pr_fut_map, pr_fut_rows = compute_annual_and_period_total(
            files=pr_fut_models[model],
            var_candidates=PR_VAR_CANDIDATES,
            model=model,
            period_label="future",
            start_year=FUT_START_YEAR,
            end_year=FUT_END_YEAR,
            chile_geom=chile_geom
        )

        # ====================================================
        # PR ESTACIONAL
        # ====================================================

        print("   PR histórico estacional")

        pr_hist_seasonal, pr_hist_seasonal_rows = compute_seasonal_period_totals(
            files=pr_hist_models[model],
            var_candidates=PR_VAR_CANDIDATES,
            model=model,
            period_label="historical",
            start_year=HIST_START_YEAR,
            end_year=HIST_END_YEAR,
            chile_geom=chile_geom
        )

        print("   PR futuro estacional")

        pr_fut_seasonal, pr_fut_seasonal_rows = compute_seasonal_period_totals(
            files=pr_fut_models[model],
            var_candidates=PR_VAR_CANDIDATES,
            model=model,
            period_label="future",
            start_year=FUT_START_YEAR,
            end_year=FUT_END_YEAR,
            chile_geom=chile_geom
        )

        rows_all.extend(pr_hist_rows)
        rows_all.extend(pr_fut_rows)
        rows_all.extend(pr_hist_seasonal_rows)
        rows_all.extend(pr_fut_seasonal_rows)

        # ====================================================
        # DELTA ANUAL
        # ====================================================

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

        # ====================================================
        # DELTA ESTACIONAL
        # ====================================================

        for season_key in SEASONS.keys():

            hmap = pr_hist_seasonal.get(season_key)
            fmap = pr_fut_seasonal.get(season_key)

            if hmap is None or fmap is None:
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

    # ========================================================
    # GUARDAR CSV
    # ========================================================

    summary_df = pd.DataFrame(summary_rows)
    rows_df = pd.DataFrame(rows_all)

    summary_csv = OUT_DIR / "cmip6_precipitation_summary_annual_seasonal.csv"
    rows_csv = OUT_DIR / "cmip6_precipitation_annual_seasonal_by_model.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    rows_df.to_csv(rows_csv, index=False, encoding="utf-8-sig")

    print("\nCSV generados:")
    print(f" - {summary_csv}")
    print(f" - {rows_csv}")

    # ========================================================
    # MULTIMODELO ANUAL + ESTACIONAL
    # ========================================================

    threshold_rows = []

    process_multimodel_group(
        deltas_dict=deltas_by_group,
        group_key="annual",
        group_label="Annual",
        chile_geom=chile_geom,
        gdf_chile=gdf_chile,
        gdf_region=gdf_region,
        threshold_rows=threshold_rows
    )

    for season_key, season_info in SEASONS.items():

        process_multimodel_group(
            deltas_dict=deltas_by_group,
            group_key=season_key,
            group_label=season_info["label"],
            chile_geom=chile_geom,
            gdf_chile=gdf_chile,
            gdf_region=gdf_region,
            threshold_rows=threshold_rows
        )

    # ========================================================
    # CSV ACHURADO
    # ========================================================

    if len(threshold_rows) > 0:

        threshold_df = pd.DataFrame(threshold_rows)

        threshold_csv = OUT_DIR / "cmip6_precipitation_hatching_threshold_annual_seasonal.csv"

        threshold_df.to_csv(
            threshold_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nCSV de umbral de achurado generado:")
        print(f" - {threshold_csv}")

    print("\n====================================================")
    print("PROCESO TERMINADO")
    print("====================================================")
    print(f"Carpeta de salida: {OUT_DIR}")


if __name__ == "__main__":
    main()