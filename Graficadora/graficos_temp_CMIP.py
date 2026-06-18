# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 20:01:20 2026

@author: catar
"""

# -*- coding: utf-8 -*-
"""
CMIP6 TEMPERATURE CHANGE MAPS - ANNUAL + SEASONAL

Calcula:
  - tasmax: cambio de temperatura máxima
  - tasmin: cambio de temperatura mínima
  - tmean: cambio de temperatura media = (tasmax + tasmin) / 2

Estructura esperada:

F:/CMIP6/tas_historica/
  MODELO/
    RESAMPLE_FILES/
      tasmax/
        Regrid_005_conservative/
          *.nc
      tasmin/
        Regrid_005_conservative/
          *.nc

F:/CMIP6/tas_futuro/
  MODELO/
    RESAMPLE_FILES/
      tasmax/
        Regrid_005_conservative/
          *.nc
      tasmin/
        Regrid_005_conservative/
          *.nc

Método:
  1. Calcula promedio anual por modelo, variable y periodo.
  2. Calcula promedio estacional por modelo, variable y periodo.
  3. Calcula promedio histórico.
  4. Calcula promedio futuro.
  5. Calcula delta = futuro - histórico, en °C.
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
  - Escala de color guardada en PNG aparte.
  - Escala parte en 0 °C:
      0 = blanco
      amarillo -> naranja -> rojo -> café -> negro
      rojo en 5 °C.
"""

from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
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

CMIP6_FUT_DIR = Path(
    r"F:\CMIP5\tas_futuro"
)

CMIP6_HIST_DIR = Path(
    r"F:\CMIP5\tas_historico"
)

SHAPEFILE = Path(
    r"C:/Users/catar/OneDrive - Universidad Técnica Federico Santa María/Universidad/2026/Seminario de Tesis I/Provincias/Provincias.shp"
)

OUT_DIR = Path(
    r"F:\CMIP5\salida_temperatura"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Periodos
HIST_START_YEAR = 1980
HIST_END_YEAR   = 2010

FUT_START_YEAR  = 2020
FUT_END_YEAR    = 2050

HIST_LABEL = f"{HIST_START_YEAR}-{HIST_END_YEAR}"
FUT_LABEL  = f"{FUT_START_YEAR}-{FUT_END_YEAR}"

# Variables posibles dentro de los NetCDF
TASMAX_VAR_CANDIDATES = ["tasmax", "TASMAX", "Tasmax", "tmax", "TMAX"]
TASMIN_VAR_CANDIDATES = ["tasmin", "TASMIN", "Tasmin", "tmin", "TMIN"]

# Extensión del mapa
REGION_EXTENT = (-78.5, -64.5, -56.5, -15.5)

# Escala de temperatura que parte en cero
TEMP_MIN = 0
TEMP_MAX = 8
TEMP_RED_VALUE = 5

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


def convert_mean_to_celsius(mean_da, original_da):
    """
    Convierte solo el mapa promedio a Celsius.
    Evita cargar todo el NetCDF completo.
    """
    if is_kelvin_temperature(original_da):
        mean_da = mean_da - 273.15
    else:
        try:
            sample = float(mean_da.mean(skipna=True).values)

            if np.isfinite(sample) and sample > 100:
                mean_da = mean_da - 273.15

        except Exception:
            pass

    mean_da.attrs["units"] = "degC"

    return mean_da


def spatial_mean(da):
    """
    Promedio espacial ponderado por cos(lat), ignorando NaN.
    Como la máscara deja océano en NaN, el océano no entra.
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
# 3. DETECCIÓN DE ARCHIVOS CMIP6
# ============================================================

def find_cmip6_files_by_model(base_dir, variable):
    """
    Busca archivos CMIP6 con estructura:

    base_dir/
      MODELO/
        RESAMPLE_FILES/
          tasmax/
            Regrid_005_conservative/
              *.nc

    o

    base_dir/
      MODELO/
        RESAMPLE_FILES/
          tasmin/
            Regrid_005_conservative/
              *.nc
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
    print(f"Buscando {variable} en:")
    print(base_dir)
    print("====================================================")

    for model_dir in model_dirs:

        model_name = model_dir.name

        var_dir = (
            model_dir /
            "RESAMPLE_FILES" /
            variable /
            "Regrid_005_conservative"
        )

        if not var_dir.exists():
            print(f" - {model_name}: no existe {variable}/Regrid_005_conservative")
            continue

        files = sorted(
            list(var_dir.glob("*.nc")) +
            list(var_dir.glob("*.nc4"))
        )

        if len(files) == 0:
            print(f" - {model_name}: carpeta existe, pero no hay archivos {variable}")
            continue

        by_model[model_name] = files

        print(f" - {model_name}: {len(files)} archivos {variable}")

    return by_model


# ============================================================
# 4. MÁSCARA DE CHILE
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
# 5. SHAPEFILES / NATURAL EARTH
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
# 6. CÁLCULO ANUAL Y ESTACIONAL
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
    Calcula promedio anual y promedio de periodo.
    Maneja archivos mensuales, anuales o multi-anuales.
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

                years = da["time"].dt.year.values.astype(int)
                unique_years = sorted(np.unique(years))

                for yy in unique_years:

                    if yy < start_year or yy > end_year:
                        continue

                    idx = np.where(years == yy)[0]

                    if len(idx) == 0:
                        continue

                    print(f"         Agregando datos al año {yy}")

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
                print(f"      Advertencia: falló archivo {f.name}. Se omite.")
                print(f"      Motivo: {e}")
                continue

    if not yearly_accumulator:
        print(
            f"      Advertencia: no hay años válidos para "
            f"{model} - {variable_label} - {period_label} "
            f"en {start_year}-{end_year}."
        )
        return None, []

    annual_maps = []
    annual_years = []
    annual_rows = []

    for yy in sorted(yearly_accumulator.keys()):

        print(f"      Calculando promedio anual para {yy}")

        total_sum = yearly_accumulator[yy]["sum"]
        total_count = yearly_accumulator[yy]["count"]
        ref_da = yearly_accumulator[yy]["ref_da"]

        annual_mean = total_sum / total_count.where(total_count > 0)

        annual_mean = convert_mean_to_celsius(
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
            "time_group": "annual",
            "variable": variable_label,
            "model": model,
            "period": period_label,
            "year": int(yy),
            "season": "",
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

    period_mean.name = f"{variable_label}_{period_label}_{model}_annual_mean"
    period_mean.attrs["units"] = "degC"

    return period_mean, annual_rows


def compute_seasonal_period_means(
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
    Calcula promedio estacional por año y promedio de periodo.

    Estaciones:
      verano_DJF: diciembre, enero, febrero
      otono_MAM: marzo, abril, mayo
      invierno_JJA: junio, julio, agosto
      primavera_SON: septiembre, octubre, noviembre

    Para DJF:
      - diciembre se asigna al año siguiente.
      - ejemplo: diciembre 1979 + enero 1980 + febrero 1980 = verano_DJF 1980.
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

                years = da["time"].dt.year.values.astype(int)
                months = da["time"].dt.month.values.astype(int)

                for season_key, season_info in SEASONS.items():

                    season_months = np.array(season_info["months"])

                    season_years = years.copy()

                    # En verano austral, diciembre pertenece al verano del año siguiente
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

                        print(f"         Agregando {season_key} al año-estación {int(sy)}")

                        sub = da.isel(time=idx)

                        sub_sum = sub.sum(dim="time", skipna=True).load()
                        sub_count = sub.count(dim="time").load()

                        sub_sum = clean_for_concat(sub_sum)
                        sub_count = clean_for_concat(sub_count)

                        sy = int(sy)

                        if sy not in seasonal_accumulator[season_key]:
                            seasonal_accumulator[season_key][sy] = {
                                "sum": sub_sum,
                                "count": sub_count,
                                "ref_da": da,
                            }
                        else:
                            old_sum = seasonal_accumulator[season_key][sy]["sum"]
                            old_count = seasonal_accumulator[season_key][sy]["count"]

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

                            seasonal_accumulator[season_key][sy]["sum"] = old_sum + sub_sum
                            seasonal_accumulator[season_key][sy]["count"] = old_count + sub_count

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
                f"{model} - {variable_label} - {period_label} - {season_key}."
            )
            seasonal_means[season_key] = None
            continue

        season_maps = []
        season_years_out = []

        for sy in sorted(acc.keys()):

            print(f"      Calculando promedio {season_key} para {sy}")

            total_sum = acc[sy]["sum"]
            total_count = acc[sy]["count"]
            ref_da = acc[sy]["ref_da"]

            season_mean = total_sum / total_count.where(total_count > 0)

            season_mean = convert_mean_to_celsius(
                season_mean,
                ref_da
            )

            season_mean = clean_for_concat(season_mean)
            season_mean = apply_chile_mask(season_mean, chile_geom)

            season_mean.name = f"{variable_label}_{period_label}_{model}_{season_key}_{sy}"
            season_mean.attrs["units"] = "degC"

            season_maps.append(season_mean)
            season_years_out.append(int(sy))

            seasonal_rows.append({
                "time_group": "seasonal",
                "variable": variable_label,
                "model": model,
                "period": period_label,
                "year": int(sy),
                "season": season_key,
                "spatial_mean_temperature_c": spatial_mean(season_mean),
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

        period_mean.name = f"{variable_label}_{period_label}_{model}_{season_key}_mean"
        period_mean.attrs["units"] = "degC"

        seasonal_means[season_key] = period_mean

    return seasonal_means, seasonal_rows


def build_tmean_annual_rows(tasmax_rows, tasmin_rows, model, period_label):
    df_max = pd.DataFrame(tasmax_rows)
    df_min = pd.DataFrame(tasmin_rows)

    if df_max.empty or df_min.empty:
        return []

    df_max = df_max[
        (df_max["time_group"] == "annual")
    ][["year", "spatial_mean_temperature_c"]].rename(
        columns={"spatial_mean_temperature_c": "tasmax_c"}
    )

    df_min = df_min[
        (df_min["time_group"] == "annual")
    ][["year", "spatial_mean_temperature_c"]].rename(
        columns={"spatial_mean_temperature_c": "tasmin_c"}
    )

    df = pd.merge(df_max, df_min, on="year", how="inner")

    df["spatial_mean_temperature_c"] = (
        df["tasmax_c"] + df["tasmin_c"]
    ) / 2

    rows = []

    for _, r in df.iterrows():
        rows.append({
            "time_group": "annual",
            "variable": "tmean",
            "model": model,
            "period": period_label,
            "year": int(r["year"]),
            "season": "",
            "spatial_mean_temperature_c": float(r["spatial_mean_temperature_c"]),
        })

    return rows


def build_tmean_seasonal_rows(tasmax_rows, tasmin_rows, model, period_label):
    df_max = pd.DataFrame(tasmax_rows)
    df_min = pd.DataFrame(tasmin_rows)

    if df_max.empty or df_min.empty:
        return []

    df_max = df_max[
        (df_max["time_group"] == "seasonal")
    ][["year", "season", "spatial_mean_temperature_c"]].rename(
        columns={"spatial_mean_temperature_c": "tasmax_c"}
    )

    df_min = df_min[
        (df_min["time_group"] == "seasonal")
    ][["year", "season", "spatial_mean_temperature_c"]].rename(
        columns={"spatial_mean_temperature_c": "tasmin_c"}
    )

    df = pd.merge(df_max, df_min, on=["year", "season"], how="inner")

    df["spatial_mean_temperature_c"] = (
        df["tasmax_c"] + df["tasmin_c"]
    ) / 2

    rows = []

    for _, r in df.iterrows():
        rows.append({
            "time_group": "seasonal",
            "variable": "tmean",
            "model": model,
            "period": period_label,
            "year": int(r["year"]),
            "season": r["season"],
            "spatial_mean_temperature_c": float(r["spatial_mean_temperature_c"]),
        })

    return rows


def add_summary_row(
    summary_rows,
    time_group,
    season,
    variable,
    model,
    hist_map,
    fut_map,
    delta_map,
    n_hist,
    n_fut
):
    summary_rows.append({
        "time_group": time_group,
        "season": season,
        "variable": variable,
        "model": model,
        "n_years_historical": int(n_hist),
        "n_years_future": int(n_fut),
        "historical_spatial_mean_c": spatial_mean(hist_map),
        "future_spatial_mean_c": spatial_mean(fut_map),
        "delta_spatial_mean_c": spatial_mean(delta_map),
    })


# ============================================================
# 7. COLORES Y MAPAS
# ============================================================

def make_temperature_cmap():
    """
    Escala positiva que parte en 0:
      0 °C = blanco
      luego amarillo -> naranja -> rojo -> café -> negro
      rojo en 5 °C
    """
    red_pos = TEMP_RED_VALUE / TEMP_MAX
    red_pos = min(max(red_pos, 0.55), 0.90)
    brown_pos = min(red_pos + 0.15, 0.97)

    colors = [
        (0.00, "#FFFFFF"),      # 0 = blanco
        (0.20, "#FFF7BC"),      # amarillo claro
        (0.40, "#FEC44F"),      # amarillo/naranja
        (0.55, "#F16913"),      # naranja
        (red_pos, "#D7301F"),   # rojo en 5 °C
        (brown_pos, "#8C510A"), # café
        (1.00, "#000000"),      # negro
    ]

    colors = sorted(colors, key=lambda x: x[0])

    cmap = LinearSegmentedColormap.from_list(
        "temperature_positive_cmap",
        colors,
        N=256
    )

    # Todo valor menor a 0 se verá blanco
    cmap.set_under("#FFFFFF")

    # Todo valor mayor al máximo se verá negro
    cmap.set_over("#000000")

    return cmap


def save_temperature_scale_png(output_path):
    """
    Guarda la escala de colores en un PNG aparte.
    """
    cmap = make_temperature_cmap()
    norm = Normalize(vmin=TEMP_MIN, vmax=TEMP_MAX)

    fig, ax = plt.subplots(figsize=(2.0, 6.0))

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        extend="max"
    )

    cbar.set_label(
        "Temperature change (°C)",
        fontsize=11,
        rotation=90,
        labelpad=12
    )

    ticks = list(range(TEMP_MIN, TEMP_MAX + 1, 1))
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


def plot_temperature_map(
    da,
    output_path,
    gdf_chile=None,
    gdf_region=None,
    agreement_mask=None
):
    """
    Grafica mapa sin barra de color lateral.
    La barra se guarda aparte con save_temperature_scale_png().
    """
    cmap = make_temperature_cmap()

    norm = Normalize(
        vmin=TEMP_MIN,
        vmax=TEMP_MAX
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
# 8. MULTIMODELO + ACHURADO
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
    """

    for variable in ["tasmax", "tasmin", "tmean"]:

        print("\n====================================================")
        print(f"MULTIMODELO CMIP6: {variable} - {group_key}")
        print("====================================================")

        if len(deltas_dict[group_key][variable]) == 0:
            print(f"No hay modelos disponibles para {variable} - {group_key}. Se omite.")
            continue

        models = sorted(deltas_dict[group_key][variable].keys())

        print(f"Modelos usados para {variable} - {group_key}: {len(models)}")
        for m in models:
            print(f" - {m}")

        delta_list = []

        for m in models:
            da_model = deltas_dict[group_key][variable][m]
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

        stack.name = f"cmip6_{variable}_delta_{group_key}_by_model"
        stack.attrs["units"] = "degC"
        stack.attrs["description"] = (
            f"CMIP6 {variable} temperature change by model. "
            f"Time group: {group_label}. "
            f"Delta = future ({FUT_LABEL}) - historical ({HIST_LABEL})"
        )

        multimodel_mean = stack.mean(dim="model", skipna=True)
        multimodel_mean = clean_for_concat(multimodel_mean)
        multimodel_mean = apply_chile_mask(multimodel_mean, chile_geom)

        multimodel_mean.name = f"cmip6_{variable}_delta_{group_key}_multimodel"
        multimodel_mean.attrs["units"] = "degC"
        multimodel_mean.attrs["description"] = (
            f"CMIP6 multi-model mean {variable} temperature change. "
            f"Time group: {group_label}. "
            f"Delta = future ({FUT_LABEL}) - historical ({HIST_LABEL})"
        )

        n_models = len(models)
        threshold = agreement_threshold(n_models, AGREEMENT_PCT)

        n_positive = (stack > 0).sum(dim="model")
        n_negative = (stack < 0).sum(dim="model")

        agreement_mask = (n_positive >= threshold) | (n_negative >= threshold)
        agreement_mask = clean_for_concat(agreement_mask)
        agreement_mask = apply_chile_mask(agreement_mask, chile_geom)

        agreement_mask.name = f"cmip6_{variable}_agreement_hatching_{group_key}"
        agreement_mask.attrs["description"] = (
            f"Grid cells where at least {threshold} of {n_models} models "
            f"agree in the sign of change. Time group: {group_label}."
        )

        actual_pct = int(np.ceil((threshold / n_models) * 100))

        print(f"Número de modelos: {n_models}")
        print(f"70% de modelos: {n_models * AGREEMENT_PCT}")
        print(f"Umbral usado para achurado: {threshold}")
        print(f"Porcentaje real equivalente: ≥{actual_pct}%")

        threshold_rows.append({
            "time_group": group_key,
            "time_group_label": group_label,
            "variable": variable,
            "n_models_used": n_models,
            "target_70_pct_value": n_models * AGREEMENT_PCT,
            "hatching_threshold_used": threshold,
            "actual_equivalent_percent": actual_pct,
            "rule": "standard rounding: 4.2->4, 4.6->5, 4.5->5",
        })

        # NetCDF
        nc_out = OUT_DIR / "netcdf"
        nc_out.mkdir(parents=True, exist_ok=True)

        stack_nc = nc_out / f"cmip6_{variable}_delta_{group_key}_by_model.nc"
        multimodel_nc = nc_out / f"cmip6_{variable}_delta_{group_key}_multimodel.nc"
        hatching_nc = nc_out / f"cmip6_{variable}_agreement_hatching_{group_key}.nc"

        stack.to_netcdf(stack_nc)
        multimodel_mean.to_netcdf(multimodel_nc)
        agreement_mask.fillna(0).astype("int8").to_netcdf(hatching_nc)

        print("\nNetCDF generados:")
        print(f" - {stack_nc}")
        print(f" - {multimodel_nc}")
        print(f" - {hatching_nc}")

        # Mapas
        var_out = OUT_DIR / f"maps_{variable}" / group_key
        var_out.mkdir(parents=True, exist_ok=True)

        if MAKE_MODEL_MAPS:

            print("\nGenerando mapas individuales por modelo...")

            for model, da_delta in zip(models, aligned):

                output_png = var_out / f"cmip6_{variable}_delta_{group_key}_{safe_name(model)}.png"

                plot_temperature_map(
                    da=da_delta,
                    output_path=output_png,
                    gdf_chile=gdf_chile,
                    gdf_region=gdf_region,
                    agreement_mask=None
                )

            print(f"Mapas individuales guardados en: {var_out}")

        if MAKE_MULTIMODEL_MAPS:

            print("\nGenerando mapa multimodelo con achurado...")

            output_png = var_out / f"cmip6_{variable}_delta_{group_key}_multimodel_hatched.png"

            plot_temperature_map(
                da=multimodel_mean,
                output_path=output_png,
                gdf_chile=gdf_chile,
                gdf_region=gdf_region,
                agreement_mask=agreement_mask
            )

            print("Mapa multimodelo con achurado guardado en:")
            print(f" - {output_png}")

            output_png_no_hatch = var_out / f"cmip6_{variable}_delta_{group_key}_multimodel_no_hatch.png"

            plot_temperature_map(
                da=multimodel_mean,
                output_path=output_png_no_hatch,
                gdf_chile=gdf_chile,
                gdf_region=gdf_region,
                agreement_mask=None
            )

            print("Mapa multimodelo sin achurado guardado en:")
            print(f" - {output_png_no_hatch}")


# ============================================================
# 9. PROCESO PRINCIPAL
# ============================================================

def main():
    print("====================================================")
    print("CMIP6 TEMPERATURE CHANGE MAPS")
    print("Annual + seasonal")
    print("Delta = Future - Historical")
    print("Unit: °C")
    print("====================================================")

    print("\nCarpetas:")
    print(f"Histórico: {CMIP6_HIST_DIR}")
    print(f"Futuro:    {CMIP6_FUT_DIR}")
    print(f"Salida:    {OUT_DIR}")

    print("\nPeriodos:")
    print(f"Histórico: {HIST_LABEL}")
    print(f"Futuro:    {FUT_LABEL}")

    chile_geom = load_chile_geometry(SHAPEFILE)
    gdf_chile = load_shape_for_plot(SHAPEFILE)
    gdf_region = load_regional_layer()

    # Guardar escala aparte
    scale_png = OUT_DIR / "cmip6_temperature_scale_0_to_8C.png"
    save_temperature_scale_png(scale_png)

    print("\nEscala de colores guardada en:")
    print(f" - {scale_png}")

    tasmax_hist_models = find_cmip6_files_by_model(CMIP6_HIST_DIR, "tasmax")
    tasmax_fut_models  = find_cmip6_files_by_model(CMIP6_FUT_DIR, "tasmax")

    tasmin_hist_models = find_cmip6_files_by_model(CMIP6_HIST_DIR, "tasmin")
    tasmin_fut_models  = find_cmip6_files_by_model(CMIP6_FUT_DIR, "tasmin")

    print("\n====================================================")
    print("DIAGNÓSTICO DE DISPONIBILIDAD CMIP6")
    print("====================================================")

    all_models = sorted(
        set(tasmax_hist_models.keys()) |
        set(tasmax_fut_models.keys()) |
        set(tasmin_hist_models.keys()) |
        set(tasmin_fut_models.keys())
    )

    for model in all_models:
        print(
            f"{model}: "
            f"tasmax_hist={model in tasmax_hist_models}, "
            f"tasmax_fut={model in tasmax_fut_models}, "
            f"tasmin_hist={model in tasmin_hist_models}, "
            f"tasmin_fut={model in tasmin_fut_models}"
        )

    print("====================================================\n")

    if not all_models:
        raise ValueError("No se encontraron modelos CMIP6.")

    summary_rows = []
    annual_rows_all = []

    group_keys = ["annual"] + list(SEASONS.keys())

    deltas_by_group = {}

    for group_key in group_keys:
        deltas_by_group[group_key] = {
            "tasmax": {},
            "tasmin": {},
            "tmean": {},
        }

    for model in all_models:

        print("\n----------------------------------------------------")
        print(f"Procesando modelo: {model}")
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
            print(f"{model}: no tiene tasmax ni tasmin completo. Se omite.")
            continue

        tasmax_hist_map = None
        tasmax_fut_map = None
        tasmax_hist_rows = []
        tasmax_fut_rows = []
        tasmax_hist_seasonal = {}
        tasmax_fut_seasonal = {}

        tasmin_hist_map = None
        tasmin_fut_map = None
        tasmin_hist_rows = []
        tasmin_fut_rows = []
        tasmin_hist_seasonal = {}
        tasmin_fut_seasonal = {}

        # ====================================================
        # TASMAX
        # ====================================================

        if has_tasmax:
            print("   TASMAX histórico anual")

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

            print("   TASMAX futuro anual")

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

            print("   TASMAX histórico estacional")

            tasmax_hist_seasonal, tasmax_hist_seasonal_rows = compute_seasonal_period_means(
                files=tasmax_hist_models[model],
                var_candidates=TASMAX_VAR_CANDIDATES,
                model=model,
                variable_label="tasmax",
                period_label="historical",
                start_year=HIST_START_YEAR,
                end_year=HIST_END_YEAR,
                chile_geom=chile_geom
            )

            print("   TASMAX futuro estacional")

            tasmax_fut_seasonal, tasmax_fut_seasonal_rows = compute_seasonal_period_means(
                files=tasmax_fut_models[model],
                var_candidates=TASMAX_VAR_CANDIDATES,
                model=model,
                variable_label="tasmax",
                period_label="future",
                start_year=FUT_START_YEAR,
                end_year=FUT_END_YEAR,
                chile_geom=chile_geom
            )

            tasmax_hist_rows.extend(tasmax_hist_seasonal_rows)
            tasmax_fut_rows.extend(tasmax_fut_seasonal_rows)

            if tasmax_hist_map is not None and tasmax_fut_map is not None:

                tasmax_hist_map, tasmax_fut_map = xr.align(
                    tasmax_hist_map,
                    tasmax_fut_map,
                    join="inner"
                )

                tasmax_delta = tasmax_fut_map - tasmax_hist_map
                tasmax_delta = clean_for_concat(tasmax_delta)
                tasmax_delta = apply_chile_mask(tasmax_delta, chile_geom)
                tasmax_delta.name = f"tasmax_delta_annual_{model}"
                tasmax_delta.attrs["units"] = "degC"

                deltas_by_group["annual"]["tasmax"][model] = tasmax_delta

                add_summary_row(
                    summary_rows,
                    "annual",
                    "",
                    "tasmax",
                    model,
                    tasmax_hist_map,
                    tasmax_fut_map,
                    tasmax_delta,
                    len({r["year"] for r in tasmax_hist_rows if r["time_group"] == "annual"}),
                    len({r["year"] for r in tasmax_fut_rows if r["time_group"] == "annual"}),
                )

            for season_key in SEASONS.keys():

                hmap = tasmax_hist_seasonal.get(season_key)
                fmap = tasmax_fut_seasonal.get(season_key)

                if hmap is None or fmap is None:
                    continue

                hmap, fmap = xr.align(hmap, fmap, join="inner")

                delta = fmap - hmap
                delta = clean_for_concat(delta)
                delta = apply_chile_mask(delta, chile_geom)
                delta.name = f"tasmax_delta_{season_key}_{model}"
                delta.attrs["units"] = "degC"

                deltas_by_group[season_key]["tasmax"][model] = delta

                add_summary_row(
                    summary_rows,
                    "seasonal",
                    season_key,
                    "tasmax",
                    model,
                    hmap,
                    fmap,
                    delta,
                    len({r["year"] for r in tasmax_hist_rows if r["season"] == season_key}),
                    len({r["year"] for r in tasmax_fut_rows if r["season"] == season_key}),
                )

            annual_rows_all.extend(tasmax_hist_rows)
            annual_rows_all.extend(tasmax_fut_rows)

        # ====================================================
        # TASMIN
        # ====================================================

        if has_tasmin:
            print("   TASMIN histórico anual")

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

            print("   TASMIN futuro anual")

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

            print("   TASMIN histórico estacional")

            tasmin_hist_seasonal, tasmin_hist_seasonal_rows = compute_seasonal_period_means(
                files=tasmin_hist_models[model],
                var_candidates=TASMIN_VAR_CANDIDATES,
                model=model,
                variable_label="tasmin",
                period_label="historical",
                start_year=HIST_START_YEAR,
                end_year=HIST_END_YEAR,
                chile_geom=chile_geom
            )

            print("   TASMIN futuro estacional")

            tasmin_fut_seasonal, tasmin_fut_seasonal_rows = compute_seasonal_period_means(
                files=tasmin_fut_models[model],
                var_candidates=TASMIN_VAR_CANDIDATES,
                model=model,
                variable_label="tasmin",
                period_label="future",
                start_year=FUT_START_YEAR,
                end_year=FUT_END_YEAR,
                chile_geom=chile_geom
            )

            tasmin_hist_rows.extend(tasmin_hist_seasonal_rows)
            tasmin_fut_rows.extend(tasmin_fut_seasonal_rows)

            if tasmin_hist_map is not None and tasmin_fut_map is not None:

                tasmin_hist_map, tasmin_fut_map = xr.align(
                    tasmin_hist_map,
                    tasmin_fut_map,
                    join="inner"
                )

                tasmin_delta = tasmin_fut_map - tasmin_hist_map
                tasmin_delta = clean_for_concat(tasmin_delta)
                tasmin_delta = apply_chile_mask(tasmin_delta, chile_geom)
                tasmin_delta.name = f"tasmin_delta_annual_{model}"
                tasmin_delta.attrs["units"] = "degC"

                deltas_by_group["annual"]["tasmin"][model] = tasmin_delta

                add_summary_row(
                    summary_rows,
                    "annual",
                    "",
                    "tasmin",
                    model,
                    tasmin_hist_map,
                    tasmin_fut_map,
                    tasmin_delta,
                    len({r["year"] for r in tasmin_hist_rows if r["time_group"] == "annual"}),
                    len({r["year"] for r in tasmin_fut_rows if r["time_group"] == "annual"}),
                )

            for season_key in SEASONS.keys():

                hmap = tasmin_hist_seasonal.get(season_key)
                fmap = tasmin_fut_seasonal.get(season_key)

                if hmap is None or fmap is None:
                    continue

                hmap, fmap = xr.align(hmap, fmap, join="inner")

                delta = fmap - hmap
                delta = clean_for_concat(delta)
                delta = apply_chile_mask(delta, chile_geom)
                delta.name = f"tasmin_delta_{season_key}_{model}"
                delta.attrs["units"] = "degC"

                deltas_by_group[season_key]["tasmin"][model] = delta

                add_summary_row(
                    summary_rows,
                    "seasonal",
                    season_key,
                    "tasmin",
                    model,
                    hmap,
                    fmap,
                    delta,
                    len({r["year"] for r in tasmin_hist_rows if r["season"] == season_key}),
                    len({r["year"] for r in tasmin_fut_rows if r["season"] == season_key}),
                )

            annual_rows_all.extend(tasmin_hist_rows)
            annual_rows_all.extend(tasmin_fut_rows)

        # ====================================================
        # TMEAN ANUAL
        # ====================================================

        if (
            tasmax_hist_map is not None and
            tasmax_fut_map is not None and
            tasmin_hist_map is not None and
            tasmin_fut_map is not None
        ):
            print("   Calculando TMEAN anual = (TASMAX + TASMIN) / 2")

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

            tmean_delta = tmean_fut_map - tmean_hist_map
            tmean_delta = clean_for_concat(tmean_delta)
            tmean_delta = apply_chile_mask(tmean_delta, chile_geom)
            tmean_delta.name = f"tmean_delta_annual_{model}"
            tmean_delta.attrs["units"] = "degC"

            deltas_by_group["annual"]["tmean"][model] = tmean_delta

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
                "annual",
                "",
                "tmean",
                model,
                tmean_hist_map,
                tmean_fut_map,
                tmean_delta,
                min(
                    len({r["year"] for r in tasmax_hist_rows if r["time_group"] == "annual"}),
                    len({r["year"] for r in tasmin_hist_rows if r["time_group"] == "annual"}),
                ),
                min(
                    len({r["year"] for r in tasmax_fut_rows if r["time_group"] == "annual"}),
                    len({r["year"] for r in tasmin_fut_rows if r["time_group"] == "annual"}),
                ),
            )

        # ====================================================
        # TMEAN ESTACIONAL
        # ====================================================

        if has_tasmax and has_tasmin:

            print("   Calculando TMEAN estacional = (TASMAX + TASMIN) / 2")

            for season_key in SEASONS.keys():

                tx_h = tasmax_hist_seasonal.get(season_key)
                tx_f = tasmax_fut_seasonal.get(season_key)
                tn_h = tasmin_hist_seasonal.get(season_key)
                tn_f = tasmin_fut_seasonal.get(season_key)

                if tx_h is None or tx_f is None or tn_h is None or tn_f is None:
                    print(f"      TMEAN {season_key} no se calcula porque falta tasmax o tasmin.")
                    continue

                tx_h, tn_h = xr.align(tx_h, tn_h, join="inner")
                tx_f, tn_f = xr.align(tx_f, tn_f, join="inner")

                tmean_h = (tx_h + tn_h) / 2
                tmean_f = (tx_f + tn_f) / 2

                tmean_delta = tmean_f - tmean_h
                tmean_delta = clean_for_concat(tmean_delta)
                tmean_delta = apply_chile_mask(tmean_delta, chile_geom)
                tmean_delta.name = f"tmean_delta_{season_key}_{model}"
                tmean_delta.attrs["units"] = "degC"

                deltas_by_group[season_key]["tmean"][model] = tmean_delta

                add_summary_row(
                    summary_rows,
                    "seasonal",
                    season_key,
                    "tmean",
                    model,
                    tmean_h,
                    tmean_f,
                    tmean_delta,
                    len({r["year"] for r in tasmax_hist_rows if r["season"] == season_key}),
                    len({r["year"] for r in tasmax_fut_rows if r["season"] == season_key}),
                )

            annual_rows_all.extend(
                build_tmean_seasonal_rows(
                    tasmax_hist_rows,
                    tasmin_hist_rows,
                    model,
                    "historical"
                )
            )

            annual_rows_all.extend(
                build_tmean_seasonal_rows(
                    tasmax_fut_rows,
                    tasmin_fut_rows,
                    model,
                    "future"
                )
            )

    # ========================================================
    # GUARDAR CSV
    # ========================================================

    summary_df = pd.DataFrame(summary_rows)
    annual_df = pd.DataFrame(annual_rows_all)

    summary_csv = OUT_DIR / "cmip6_temperature_summary_annual_seasonal.csv"
    annual_csv = OUT_DIR / "cmip6_temperature_annual_seasonal_by_model.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    annual_df.to_csv(annual_csv, index=False, encoding="utf-8-sig")

    print("\nCSV generados:")
    print(f" - {summary_csv}")
    print(f" - {annual_csv}")

    # ========================================================
    # MULTIMODELO ANUAL + ESTACIONAL
    # ========================================================

    threshold_rows = []

    # Anual
    process_multimodel_group(
        deltas_dict=deltas_by_group,
        group_key="annual",
        group_label="Annual",
        chile_geom=chile_geom,
        gdf_chile=gdf_chile,
        gdf_region=gdf_region,
        threshold_rows=threshold_rows
    )

    # Estaciones
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

        threshold_csv = OUT_DIR / "cmip6_temperature_hatching_threshold_annual_seasonal.csv"

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