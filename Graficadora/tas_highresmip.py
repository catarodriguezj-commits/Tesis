# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 17:57:31 2026

@author: catar
"""

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
HIGHRES TEMPERATURE CHANGE MAPS - ANNUAL + SEASONAL - FIXED

Calcula:
  - tasmax
  - tasmin
  - tmean = (tasmax + tasmin) / 2

Incluye:
  - mapas anuales
  - mapas estacionales
  - multimodelo con achurado >=70%
  - escala en PNG aparte
  - resumen final con modelos incluidos
  - permite estaciones incompletas si faltan meses
  - evita error zero-size array por xr.align(join='inner')
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

TASMAX_HIST_DIR = Path(
    r"E:\Cata\HighRes\tasmax_hist\RESAMPLE_FILES\Regrid_005_conservative"
)

TASMAX_FUT_DIR = Path(
    r"E:\Cata\HighRes\tasmax_futuro\RESAMPLE_FILES\Regrid_005_conservative"
)

TASMIN_HIST_DIR = Path(
    r"E:\Cata\HighRes\tasmin_historico\RESAMPLE_FILES\Regrid_005_conservative"
)

TASMIN_FUT_DIR = Path(
    r"E:\Cata\HighRes\tasmin_futuro\RESAMPLE_FILES\Regrid_005_conservative"
)

SHAPEFILE = Path(
    r"C:\Users\catar\OneDrive - Universidad Técnica Federico Santa María\Universidad\2026\Seminario de Tesis I\Provincias\Provincias.shp"
)

OUT_DIR = Path(
    r"E:\Cata\HighRes\salida_temperatura"
)

OUT_DIR.mkdir(parents=True, exist_ok=True)

HIST_START_YEAR = 1980
HIST_END_YEAR   = 2010

FUT_START_YEAR  = 2020
FUT_END_YEAR    = 2050

HIST_LABEL = f"{HIST_START_YEAR}-{HIST_END_YEAR}"
FUT_LABEL  = f"{FUT_START_YEAR}-{FUT_END_YEAR}"

TASMAX_VAR_CANDIDATES = ["tasmax", "TASMAX", "Tasmax", "tmax", "TMAX"]
TASMIN_VAR_CANDIDATES = ["tasmin", "TASMIN", "Tasmin", "tmin", "TMIN"]

REGION_EXTENT = (-78.5, -64.5, -56.5, -15.5)

# Escala temperatura
TEMP_NEG_MIN = -2
TEMP_POS_MAX = 8
TEMP_RED_VALUE = 5

AGREEMENT_PCT = 0.70

MAKE_MODEL_MAPS = True
MAKE_MULTIMODEL_MAPS = True

MAP_FIGSIZE = (4.2, 10.5)

OTHER_COUNTRIES_FACE = "#D9D9D9"
OTHER_COUNTRIES_EDGE = "#9E9E9E"
OCEAN_FACE = "white"

CHILE_BORDER_COLOR = "black"
CHILE_BORDER_WIDTH = 0.6
OTHER_COUNTRIES_WIDTH = 0.5

PREFER_REGRID_FILES = True

# IMPORTANTE:
# False = calcula estaciones aunque falte algún mes.
# Ejemplo: si verano tiene solo enero-febrero, igual lo usa.
REQUIRE_COMPLETE_SEASONS = False

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


def has_zero_size(da):
    try:
        return any(size == 0 for size in da.sizes.values())
    except Exception:
        return False


def align_maps(a, b):
    """
    Alinea dos mapas sin destruir la grilla.

    Si tienen misma forma, copia coordenadas del primero al segundo.
    Esto evita que xr.align(join='inner') deje lat/lon vacíos por
    diferencias mínimas entre coordenadas.
    """
    a = clean_for_concat(a)
    b = clean_for_concat(b)

    if set(a.dims) == set(b.dims):
        same_sizes = True

        for d in a.dims:
            if a.sizes.get(d) != b.sizes.get(d):
                same_sizes = False
                break

        if same_sizes:
            try:
                b = b.transpose(*a.dims)
            except Exception:
                pass

            for coord in ["lat", "lon", "x", "y"]:
                if coord in a.coords and coord in b.coords:
                    try:
                        if a[coord].shape == b[coord].shape:
                            b = b.assign_coords({coord: a[coord]})
                    except Exception:
                        pass

            return a, b

    a2, b2 = xr.align(a, b, join="inner")

    if has_zero_size(a2) or has_zero_size(b2):
        raise ValueError(
            "La grilla quedó vacía después de xr.align(join='inner'). "
            "Probablemente las coordenadas no calzan exactamente."
        )

    return a2, b2


def align_many_for_stack(maps):
    """
    Alinea muchos mapas usando el primero como referencia.
    """
    if len(maps) == 0:
        return []

    base = clean_for_concat(maps[0])
    out = [base]

    for m in maps[1:]:
        _, m2 = align_maps(base, m)
        out.append(m2)

    return out


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


def is_kelvin_temperature_from_units(units):
    units = str(units).lower().strip()

    kelvin_units = [
        "k",
        "kelvin",
        "degrees_k",
        "degree_k",
        "degree kelvin",
        "degrees kelvin",
    ]

    return units in kelvin_units


def convert_temperature_map_to_celsius(da, source_units=""):
    """
    Convierte una temperatura media a Celsius.
    Si no detecta unidades, usa heurística: valores >100 son Kelvin.
    """
    out = da.copy()

    if is_kelvin_temperature_from_units(source_units):
        out = out - 273.15
    else:
        try:
            sample = float(out.mean(skipna=True).values)
            if np.isfinite(sample) and sample > 100:
                out = out - 273.15
        except Exception:
            pass

    out.attrs["units"] = "degC"

    return out


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
# 3. BUSCAR ARCHIVOS HIGHRES
# ============================================================

def is_temperature_file(path, variable):
    name = path.name.lower()
    full = str(path).lower()
    variable = variable.lower()

    return (
        name.startswith(variable + "_") or
        f"_{variable}_" in name or
        f"/{variable}/" in full or
        f"\\{variable}\\" in full
    )


def is_output_or_auxiliary_file(path):
    text = str(path).lower()

    bad_keys = [
        "salida_temperatura",
        "maps_",
        "netcdf",
        "hatching",
        "summary",
        "scale",
        ".png",
        ".csv"
    ]

    return any(k in text for k in bad_keys)


def extract_highres_model_name(path, base_dir, variable):
    path = Path(path)
    base_dir = Path(base_dir)

    try:
        rel = path.relative_to(base_dir)
        parts = rel.parts

        if len(parts) > 1:
            first = parts[0]

            generic = {
                "tasmax", "tasmin", "tas", "TASMAX", "TASMIN",
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
        rf"{variable}_day_(.*?)_(historical|hist-1950|highres-future|highressst-present|highressst-future|future|ssp|rcp)",
        rf"{variable}_(.*?)_(historical|hist-1950|highres-future|highressst-present|highressst-future|future|ssp|rcp)",
    ]

    for pat in patterns:
        m = re.search(pat, name, flags=re.IGNORECASE)
        if m:
            return m.group(1)

    return path.parent.name


def prefer_regrid_if_available(files):
    if not PREFER_REGRID_FILES:
        return sorted(files)

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


def find_highres_temp_files_by_model(base_dir, variable, period_label):
    base_dir = Path(base_dir)

    if not base_dir.exists():
        print(f"Advertencia: no existe la carpeta: {base_dir}")
        return {}

    all_nc = sorted(
        list(base_dir.rglob("*.nc")) +
        list(base_dir.rglob("*.nc4"))
    )

    all_nc = [
        f for f in all_nc
        if not is_output_or_auxiliary_file(f)
    ]

    temp_files = [
        f for f in all_nc
        if is_temperature_file(f, variable)
    ]

    if len(temp_files) == 0:
        temp_files = all_nc

    by_model = {}

    for f in temp_files:
        model = extract_highres_model_name(f, base_dir, variable)

        if model not in by_model:
            by_model[model] = []

        by_model[model].append(f)

    for model in list(by_model.keys()):
        by_model[model] = prefer_regrid_if_available(by_model[model])

    print("\n====================================================")
    print(f"ARCHIVOS HIGHRES {variable.upper()} {period_label.upper()}")
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
# 4. MÁSCARA CHILE
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

    if has_zero_size(da_ref):
        print("Advertencia: grilla vacía al construir máscara de Chile.")
        return xr.full_like(da_ref, False, dtype=bool)

    x, y = get_xy_for_dataarray(da_ref)

    if x.size == 0 or y.size == 0:
        print("Advertencia: coordenadas vacías al construir máscara de Chile.")
        return xr.full_like(da_ref, False, dtype=bool)

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

    if has_zero_size(da):
        return da

    mask = build_chile_mask(da, chile_geom)

    try:
        return da.where(mask)
    except Exception:
        da_aligned, mask_aligned = align_maps(da, mask)
        return da_aligned.where(mask_aligned)


# ============================================================
# 5. CAPAS MAPA
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
    yearly_data = {}

    for f in files:
        print(f"      Leyendo anual: {f.name}")

        try:
            ds = open_dataset_safe(f)
        except Exception as e:
            raise RuntimeError(f"No se pudo abrir {f}: {e}")

        try:
            var_name = select_variable(ds, var_candidates)

            da = ds[var_name]
            source_units = str(da.attrs.get("units", ""))

            da = standardize_coords(da)
            da = da.squeeze(drop=True)
            da = clean_for_concat(da)

            if "time" not in da.dims:
                raise ValueError(f"{f.name} no tiene dimensión time.")

            years = da["time"].dt.year.values.astype(int)
            unique_years = sorted(np.unique(years))

            for yy in unique_years:
                if yy < start_year or yy > end_year:
                    continue

                idx = np.where(years == yy)[0]

                if len(idx) == 0:
                    continue

                print(f"         Procesando año {yy} con {len(idx)} pasos de tiempo")

                sub = da.isel(time=idx)

                annual_sum = sub.sum(dim="time", skipna=False)
                annual_sum = annual_sum.load()
                annual_sum = clean_for_concat(annual_sum)

                yy = int(yy)

                if yy not in yearly_data:
                    yearly_data[yy] = {
                        "sum": annual_sum,
                        "count": len(idx),
                        "units": source_units,
                    }
                else:
                    old_sum = yearly_data[yy]["sum"]
                    old_sum, annual_sum = align_maps(old_sum, annual_sum)
                    yearly_data[yy]["sum"] = old_sum + annual_sum
                    yearly_data[yy]["count"] += len(idx)

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

    if not yearly_data:
        return None, []

    annual_maps = []
    annual_years = []
    annual_rows = []

    for yy in sorted(yearly_data.keys()):
        item = yearly_data[yy]

        annual_mean = item["sum"] / item["count"]
        annual_mean = convert_temperature_map_to_celsius(
            annual_mean,
            source_units=item["units"]
        )

        annual_mean = clean_for_concat(annual_mean)
        annual_mean = apply_chile_mask(annual_mean, chile_geom)

        if has_zero_size(annual_mean):
            print(f"      Advertencia: {variable_label} {model} {yy} quedó vacío. Se omite.")
            continue

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

    if len(annual_maps) == 0:
        return None, []

    annual_maps = align_many_for_stack(annual_maps)

    annual_stack = xr.concat(
        annual_maps,
        dim=xr.DataArray(
            annual_years,
            dims="year",
            name="year"
        ),
        join="override",
        coords="minimal",
        compat="override"
    )

    period_mean = annual_stack.mean(dim="year", skipna=True)
    period_mean = clean_for_concat(period_mean)
    period_mean = apply_chile_mask(period_mean, chile_geom)

    if has_zero_size(period_mean):
        return None, annual_rows

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
    seasonal_data = {}

    for season_key in SEASONS.keys():
        seasonal_data[season_key] = {}

    for f in files:
        print(f"      Leyendo estacional: {f.name}")

        try:
            ds = open_dataset_safe(f)
        except Exception as e:
            raise RuntimeError(f"No se pudo abrir {f}: {e}")

        try:
            var_name = select_variable(ds, var_candidates)

            da = ds[var_name]
            source_units = str(da.attrs.get("units", ""))

            da = standardize_coords(da)
            da = da.squeeze(drop=True)
            da = clean_for_concat(da)

            if "time" not in da.dims:
                raise ValueError(f"{f.name} no tiene dimensión time.")

            years = da["time"].dt.year.values.astype(int)
            months = da["time"].dt.month.values.astype(int)

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

                    print(
                        f"         Procesando {season_key} {int(sy)} "
                        f"con {len(idx)} pasos de tiempo"
                    )

                    sub = da.isel(time=idx)

                    season_sum = sub.sum(dim="time", skipna=False)
                    season_sum = season_sum.load()
                    season_sum = clean_for_concat(season_sum)

                    months_present = set(months[idx].astype(int).tolist())
                    sy = int(sy)

                    if sy not in seasonal_data[season_key]:
                        seasonal_data[season_key][sy] = {
                            "sum": season_sum,
                            "count": len(idx),
                            "months_present": months_present,
                            "units": source_units,
                        }
                    else:
                        old_sum = seasonal_data[season_key][sy]["sum"]
                        old_sum, season_sum = align_maps(old_sum, season_sum)
                        seasonal_data[season_key][sy]["sum"] = old_sum + season_sum
                        seasonal_data[season_key][sy]["count"] += len(idx)
                        seasonal_data[season_key][sy]["months_present"].update(months_present)

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
        acc = seasonal_data[season_key]

        if not acc:
            seasonal_means[season_key] = None
            continue

        required_months = set(season_info["months"])

        season_maps = []
        season_years_out = []

        for sy in sorted(acc.keys()):
            item = acc[sy]
            months_present = item["months_present"]

            if REQUIRE_COMPLETE_SEASONS:
                if not required_months.issubset(months_present):
                    print(
                        f"      Omitiendo {season_key} {sy} por estación incompleta. "
                        f"Meses presentes: {sorted(months_present)}"
                    )
                    continue
            else:
                if not required_months.issubset(months_present):
                    print(
                        f"      Usando {season_key} {sy} aunque esté incompleta. "
                        f"Meses presentes: {sorted(months_present)}"
                    )

            season_mean = item["sum"] / item["count"]
            season_mean = convert_temperature_map_to_celsius(
                season_mean,
                source_units=item["units"]
            )

            season_mean = clean_for_concat(season_mean)
            season_mean = apply_chile_mask(season_mean, chile_geom)

            if has_zero_size(season_mean):
                print(f"      Advertencia: {season_key} {sy} quedó vacío. Se omite.")
                continue

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

        if len(season_maps) == 0:
            seasonal_means[season_key] = None
            continue

        season_maps = align_many_for_stack(season_maps)

        season_stack = xr.concat(
            season_maps,
            dim=xr.DataArray(
                season_years_out,
                dims="year",
                name="year"
            ),
            join="override",
            coords="minimal",
            compat="override"
        )

        period_mean = season_stack.mean(dim="year", skipna=True)
        period_mean = clean_for_concat(period_mean)
        period_mean = apply_chile_mask(period_mean, chile_geom)

        if has_zero_size(period_mean):
            seasonal_means[season_key] = None
            continue

        period_mean.name = f"{variable_label}_{period_label}_{model}_{season_key}_mean"
        period_mean.attrs["units"] = "degC"

        seasonal_means[season_key] = period_mean

    return seasonal_means, seasonal_rows


def build_tmean_rows(tasmax_rows, tasmin_rows, model, period_label):
    df_max = pd.DataFrame(tasmax_rows)
    df_min = pd.DataFrame(tasmin_rows)

    if df_max.empty or df_min.empty:
        return []

    df_max = df_max[[
        "time_group",
        "year",
        "season",
        "spatial_mean_temperature_c"
    ]].rename(
        columns={"spatial_mean_temperature_c": "tasmax_c"}
    )

    df_min = df_min[[
        "time_group",
        "year",
        "season",
        "spatial_mean_temperature_c"
    ]].rename(
        columns={"spatial_mean_temperature_c": "tasmin_c"}
    )

    df = pd.merge(
        df_max,
        df_min,
        on=["time_group", "year", "season"],
        how="inner"
    )

    df["spatial_mean_temperature_c"] = (
        df["tasmax_c"] + df["tasmin_c"]
    ) / 2

    rows = []

    for _, r in df.iterrows():
        rows.append({
            "time_group": r["time_group"],
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
    red_pos = 0.5 + 0.5 * (TEMP_RED_VALUE / TEMP_POS_MAX)
    red_pos = min(max(red_pos, 0.70), 0.92)
    brown_pos = min(red_pos + 0.10, 0.97)

    colors = [
        (0.00, "#08306B"),
        (0.20, "#2171B5"),
        (0.38, "#9ECAE1"),
        (0.50, "#FFFFFF"),
        (0.60, "#FFF7BC"),
        (0.70, "#FEC44F"),
        (0.76, "#F16913"),
        (red_pos, "#D7301F"),
        (brown_pos, "#8C510A"),
        (1.00, "#000000"),
    ]

    colors = sorted(colors, key=lambda x: x[0])

    cmap = LinearSegmentedColormap.from_list(
        "temperature_delta_cmap",
        colors,
        N=256
    )

    cmap.set_under("#08306B")
    cmap.set_over("#000000")

    return cmap


def save_temperature_scale_png(output_path):
    cmap = make_temperature_cmap()

    norm = TwoSlopeNorm(
        vmin=TEMP_NEG_MIN,
        vcenter=0,
        vmax=TEMP_POS_MAX
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
        "Temperature change (°C)",
        fontsize=11,
        rotation=90,
        labelpad=12
    )

    ticks = list(range(TEMP_NEG_MIN, TEMP_POS_MAX + 1, 1))
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
    da = clean_for_concat(da)

    if has_zero_size(da):
        print(f"Advertencia: mapa vacío. No se grafica: {output_path}")
        return

    cmap = make_temperature_cmap()

    norm = TwoSlopeNorm(
        vmin=TEMP_NEG_MIN,
        vcenter=0,
        vmax=TEMP_POS_MAX
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
        agreement_mask = clean_for_concat(agreement_mask)

        if not has_zero_size(agreement_mask):
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
# 8. MULTIMODELO
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
    for variable in ["tasmax", "tasmin", "tmean"]:

        print("\n====================================================")
        print(f"MULTIMODELO HIGHRES TEMPERATURA: {variable} - {group_key}")
        print("====================================================")

        if len(deltas_dict[group_key][variable]) == 0:
            print(f"No hay modelos disponibles para {variable} - {group_key}. Se omite.")
            continue

        models = sorted(deltas_dict[group_key][variable].keys())

        included_rows.append({
            "group_key": group_key,
            "group_label": group_label,
            "variable": variable,
            "n_models_included": len(models),
            "models": "; ".join(models),
        })

        print(f"MODELOS INCLUIDOS EN MULTIMODELO {variable} - {group_key}: {len(models)}")
        for m in models:
            print(f" - {m}")

        try:
            delta_list = []

            for m in models:
                da_model = deltas_dict[group_key][variable][m]
                da_model = clean_for_concat(da_model)
                da_model = apply_chile_mask(da_model, chile_geom)

                if not has_zero_size(da_model):
                    delta_list.append(da_model)

            if len(delta_list) == 0:
                print(f"No quedaron mapas válidos para {variable} - {group_key}.")
                continue

            aligned = align_many_for_stack(delta_list)

            aligned = [
                apply_chile_mask(clean_for_concat(a), chile_geom)
                for a in aligned
            ]

            stack = xr.concat(
                aligned,
                dim=xr.DataArray(
                    models[:len(aligned)],
                    dims="model",
                    name="model"
                ),
                coords="minimal",
                compat="override",
                join="override"
            )

            stack.name = f"highres_{variable}_delta_{group_key}_by_model"
            stack.attrs["units"] = "degC"

            multimodel_mean = stack.mean(dim="model", skipna=True)
            multimodel_mean = clean_for_concat(multimodel_mean)
            multimodel_mean = apply_chile_mask(multimodel_mean, chile_geom)
            multimodel_mean.name = f"highres_{variable}_delta_{group_key}_multimodel"
            multimodel_mean.attrs["units"] = "degC"

            n_models = int(stack.sizes["model"])
            threshold = agreement_threshold(n_models, AGREEMENT_PCT)

            n_positive = (stack > 0).sum(dim="model")
            n_negative = (stack < 0).sum(dim="model")

            agreement_mask = (n_positive >= threshold) | (n_negative >= threshold)
            agreement_mask = clean_for_concat(agreement_mask)
            agreement_mask = apply_chile_mask(agreement_mask, chile_geom)
            agreement_mask.name = f"highres_{variable}_agreement_hatching_{group_key}"

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

            nc_out = OUT_DIR / "netcdf"
            nc_out.mkdir(parents=True, exist_ok=True)

            stack_nc = nc_out / f"highres_{variable}_delta_{group_key}_by_model.nc"
            multimodel_nc = nc_out / f"highres_{variable}_delta_{group_key}_multimodel.nc"
            hatching_nc = nc_out / f"highres_{variable}_agreement_hatching_{group_key}.nc"

            stack.to_netcdf(stack_nc)
            multimodel_mean.to_netcdf(multimodel_nc)
            agreement_mask.fillna(0).astype("int8").to_netcdf(hatching_nc)

            print("\nNetCDF generados:")
            print(f" - {stack_nc}")
            print(f" - {multimodel_nc}")
            print(f" - {hatching_nc}")

            group_out = OUT_DIR / f"maps_{variable}" / group_key
            group_out.mkdir(parents=True, exist_ok=True)

            if MAKE_MODEL_MAPS:
                print("\nGenerando mapas individuales por modelo...")

                for model_name, da_delta in zip(models, aligned):
                    output_png = group_out / f"highres_{variable}_delta_{group_key}_{safe_name(model_name)}.png"

                    plot_temperature_map(
                        da=da_delta,
                        output_path=output_png,
                        gdf_chile=gdf_chile,
                        gdf_region=gdf_region,
                        agreement_mask=None
                    )

                print(f"Mapas individuales guardados en: {group_out}")

            if MAKE_MULTIMODEL_MAPS:
                print("\nGenerando mapa multimodelo con achurado...")

                output_png = group_out / f"highres_{variable}_delta_{group_key}_multimodel_hatched.png"

                plot_temperature_map(
                    da=multimodel_mean,
                    output_path=output_png,
                    gdf_chile=gdf_chile,
                    gdf_region=gdf_region,
                    agreement_mask=agreement_mask
                )

                print("Mapa multimodelo con achurado guardado en:")
                print(f" - {output_png}")

                output_png_no_hatch = group_out / f"highres_{variable}_delta_{group_key}_multimodel_no_hatch.png"

                plot_temperature_map(
                    da=multimodel_mean,
                    output_path=output_png_no_hatch,
                    gdf_chile=gdf_chile,
                    gdf_region=gdf_region,
                    agreement_mask=None
                )

                print("Mapa multimodelo sin achurado guardado en:")
                print(f" - {output_png_no_hatch}")

        except Exception as e:
            register_failure(
                failed_models,
                model=f"MULTIMODEL_{variable}_{group_key}",
                step="multimodel",
                reason=e
            )
            print(f"ERROR en multimodelo {variable} - {group_key}: {e}")


# ============================================================
# 9. RESUMEN FINAL
# ============================================================

def print_final_summary(processed_models, skipped_models, failed_models, included_rows):
    print("\n====================================================")
    print("RESUMEN FINAL HIGHRES TEMPERATURA")
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
                f"{row['variable']} | {row['group_key']} "
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
# 10. MAIN
# ============================================================

def main():
    print("====================================================")
    print("HIGHRES TEMPERATURE CHANGE MAPS - FIXED")
    print("Annual + seasonal")
    print("Delta = Future - Historical")
    print("Units: °C")
    print("====================================================")

    print("\nRutas:")
    print(f"TASMAX histórico: {TASMAX_HIST_DIR}")
    print(f"TASMAX futuro:    {TASMAX_FUT_DIR}")
    print(f"TASMIN histórico: {TASMIN_HIST_DIR}")
    print(f"TASMIN futuro:    {TASMIN_FUT_DIR}")
    print(f"Shapefile:        {SHAPEFILE}")
    print(f"Salida:           {OUT_DIR}")

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

    scale_png = OUT_DIR / "highres_temperature_scale_minus2_to_8C.png"
    save_temperature_scale_png(scale_png)

    print("\nEscala de colores guardada en:")
    print(f" - {scale_png}")

    tasmax_hist_models = find_highres_temp_files_by_model(
        TASMAX_HIST_DIR,
        "tasmax",
        "historical"
    )

    tasmax_fut_models = find_highres_temp_files_by_model(
        TASMAX_FUT_DIR,
        "tasmax",
        "future"
    )

    tasmin_hist_models = find_highres_temp_files_by_model(
        TASMIN_HIST_DIR,
        "tasmin",
        "historical"
    )

    tasmin_fut_models = find_highres_temp_files_by_model(
        TASMIN_FUT_DIR,
        "tasmin",
        "future"
    )

    all_models = sorted(
        set(tasmax_hist_models.keys()) |
        set(tasmax_fut_models.keys()) |
        set(tasmin_hist_models.keys()) |
        set(tasmin_fut_models.keys())
    )

    print("\n====================================================")
    print("DIAGNÓSTICO HIGHRES TEMPERATURA")
    print("====================================================")
    print(f"Modelos tasmax histórico: {len(tasmax_hist_models)}")
    print(f"Modelos tasmax futuro:    {len(tasmax_fut_models)}")
    print(f"Modelos tasmin histórico: {len(tasmin_hist_models)}")
    print(f"Modelos tasmin futuro:    {len(tasmin_fut_models)}")
    print(f"Modelos totales detectados: {len(all_models)}")

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
        raise ValueError("No se encontraron modelos HighRes de temperatura.")

    summary_rows = []
    rows_all = []

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
        print(f"Procesando modelo HighRes: {model}")
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
            skipped_models.append({
                "model": model,
                "reason": "No tiene tasmax ni tasmin completo."
            })
            print(f"{model}: omitido porque no tiene tasmax ni tasmin completo.")
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

        model_has_any_delta = False

        # ====================================================
        # TASMAX
        # ====================================================

        if has_tasmax:
            try:
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

                rows_all.extend(tasmax_hist_rows)
                rows_all.extend(tasmax_fut_rows)

            except Exception as e:
                register_failure(
                    failed_models,
                    model=model,
                    step="tasmax_lectura_calculo",
                    reason=e
                )
                print(f"ERROR en TASMAX {model}: {e}")
                tasmax_hist_map = None
                tasmax_fut_map = None

            try:
                if tasmax_hist_map is not None and tasmax_fut_map is not None:
                    tasmax_hist_map, tasmax_fut_map = align_maps(
                        tasmax_hist_map,
                        tasmax_fut_map
                    )

                    delta = tasmax_fut_map - tasmax_hist_map
                    delta = clean_for_concat(delta)
                    delta = apply_chile_mask(delta, chile_geom)

                    if not has_zero_size(delta):
                        delta.name = f"tasmax_delta_annual_{model}"
                        delta.attrs["units"] = "degC"

                        deltas_by_group["annual"]["tasmax"][model] = delta

                        add_summary_row(
                            summary_rows,
                            "annual",
                            "",
                            "tasmax",
                            model,
                            tasmax_hist_map,
                            tasmax_fut_map,
                            delta,
                            len({r["year"] for r in tasmax_hist_rows if r["time_group"] == "annual"}),
                            len({r["year"] for r in tasmax_fut_rows if r["time_group"] == "annual"}),
                        )

                        model_has_any_delta = True

                for season_key in SEASONS.keys():
                    hmap = tasmax_hist_seasonal.get(season_key)
                    fmap = tasmax_fut_seasonal.get(season_key)

                    if hmap is None or fmap is None:
                        print(f"{model}: TASMAX sin {season_key} completo para delta.")
                        continue

                    hmap, fmap = align_maps(hmap, fmap)

                    delta = fmap - hmap
                    delta = clean_for_concat(delta)
                    delta = apply_chile_mask(delta, chile_geom)

                    if has_zero_size(delta):
                        print(f"{model}: TASMAX {season_key} quedó vacío. Se omite.")
                        continue

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

                    model_has_any_delta = True

            except Exception as e:
                register_failure(
                    failed_models,
                    model=model,
                    step="tasmax_delta",
                    reason=e
                )
                print(f"ERROR calculando delta TASMAX en {model}: {e}")

        else:
            print(f"{model}: no tiene tasmax histórico y futuro completo.")

        # ====================================================
        # TASMIN
        # ====================================================

        if has_tasmin:
            try:
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

                rows_all.extend(tasmin_hist_rows)
                rows_all.extend(tasmin_fut_rows)

            except Exception as e:
                register_failure(
                    failed_models,
                    model=model,
                    step="tasmin_lectura_calculo",
                    reason=e
                )
                print(f"ERROR en TASMIN {model}: {e}")
                tasmin_hist_map = None
                tasmin_fut_map = None

            try:
                if tasmin_hist_map is not None and tasmin_fut_map is not None:
                    tasmin_hist_map, tasmin_fut_map = align_maps(
                        tasmin_hist_map,
                        tasmin_fut_map
                    )

                    delta = tasmin_fut_map - tasmin_hist_map
                    delta = clean_for_concat(delta)
                    delta = apply_chile_mask(delta, chile_geom)

                    if not has_zero_size(delta):
                        delta.name = f"tasmin_delta_annual_{model}"
                        delta.attrs["units"] = "degC"

                        deltas_by_group["annual"]["tasmin"][model] = delta

                        add_summary_row(
                            summary_rows,
                            "annual",
                            "",
                            "tasmin",
                            model,
                            tasmin_hist_map,
                            tasmin_fut_map,
                            delta,
                            len({r["year"] for r in tasmin_hist_rows if r["time_group"] == "annual"}),
                            len({r["year"] for r in tasmin_fut_rows if r["time_group"] == "annual"}),
                        )

                        model_has_any_delta = True

                for season_key in SEASONS.keys():
                    hmap = tasmin_hist_seasonal.get(season_key)
                    fmap = tasmin_fut_seasonal.get(season_key)

                    if hmap is None or fmap is None:
                        print(f"{model}: TASMIN sin {season_key} completo para delta.")
                        continue

                    hmap, fmap = align_maps(hmap, fmap)

                    delta = fmap - hmap
                    delta = clean_for_concat(delta)
                    delta = apply_chile_mask(delta, chile_geom)

                    if has_zero_size(delta):
                        print(f"{model}: TASMIN {season_key} quedó vacío. Se omite.")
                        continue

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

                    model_has_any_delta = True

            except Exception as e:
                register_failure(
                    failed_models,
                    model=model,
                    step="tasmin_delta",
                    reason=e
                )
                print(f"ERROR calculando delta TASMIN en {model}: {e}")

        else:
            print(f"{model}: no tiene tasmin histórico y futuro completo.")

        # ====================================================
        # TMEAN
        # ====================================================

        try:
            if (
                tasmax_hist_map is not None and
                tasmax_fut_map is not None and
                tasmin_hist_map is not None and
                tasmin_fut_map is not None
            ):
                print("   Calculando TMEAN anual")

                tasmax_hist_map, tasmin_hist_map = align_maps(
                    tasmax_hist_map,
                    tasmin_hist_map
                )

                tasmax_fut_map, tasmin_fut_map = align_maps(
                    tasmax_fut_map,
                    tasmin_fut_map
                )

                tmean_hist = (tasmax_hist_map + tasmin_hist_map) / 2
                tmean_fut = (tasmax_fut_map + tasmin_fut_map) / 2

                delta = tmean_fut - tmean_hist
                delta = clean_for_concat(delta)
                delta = apply_chile_mask(delta, chile_geom)

                if not has_zero_size(delta):
                    delta.name = f"tmean_delta_annual_{model}"
                    delta.attrs["units"] = "degC"

                    deltas_by_group["annual"]["tmean"][model] = delta

                    rows_all.extend(
                        build_tmean_rows(
                            tasmax_hist_rows,
                            tasmin_hist_rows,
                            model,
                            "historical"
                        )
                    )

                    rows_all.extend(
                        build_tmean_rows(
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
                        tmean_hist,
                        tmean_fut,
                        delta,
                        min(
                            len({r["year"] for r in tasmax_hist_rows if r["time_group"] == "annual"}),
                            len({r["year"] for r in tasmin_hist_rows if r["time_group"] == "annual"}),
                        ),
                        min(
                            len({r["year"] for r in tasmax_fut_rows if r["time_group"] == "annual"}),
                            len({r["year"] for r in tasmin_fut_rows if r["time_group"] == "annual"}),
                        ),
                    )

                    model_has_any_delta = True

                for season_key in SEASONS.keys():
                    tx_h = tasmax_hist_seasonal.get(season_key)
                    tx_f = tasmax_fut_seasonal.get(season_key)
                    tn_h = tasmin_hist_seasonal.get(season_key)
                    tn_f = tasmin_fut_seasonal.get(season_key)

                    if tx_h is None or tx_f is None or tn_h is None or tn_f is None:
                        print(f"{model}: no se calcula TMEAN {season_key}, falta mapa.")
                        continue

                    tx_h, tn_h = align_maps(tx_h, tn_h)
                    tx_f, tn_f = align_maps(tx_f, tn_f)

                    tmean_h = (tx_h + tn_h) / 2
                    tmean_f = (tx_f + tn_f) / 2

                    delta = tmean_f - tmean_h
                    delta = clean_for_concat(delta)
                    delta = apply_chile_mask(delta, chile_geom)

                    if has_zero_size(delta):
                        print(f"{model}: TMEAN {season_key} quedó vacío. Se omite.")
                        continue

                    delta.name = f"tmean_delta_{season_key}_{model}"
                    delta.attrs["units"] = "degC"

                    deltas_by_group[season_key]["tmean"][model] = delta

                    add_summary_row(
                        summary_rows,
                        "seasonal",
                        season_key,
                        "tmean",
                        model,
                        tmean_h,
                        tmean_f,
                        delta,
                        len({r["year"] for r in tasmax_hist_rows if r["season"] == season_key}),
                        len({r["year"] for r in tasmax_fut_rows if r["season"] == season_key}),
                    )

                    model_has_any_delta = True

        except Exception as e:
            register_failure(
                failed_models,
                model=model,
                step="tmean_delta",
                reason=e
            )
            print(f"ERROR calculando TMEAN en {model}: {e}")

        if model_has_any_delta:
            processed_models.append(model)
        else:
            skipped_models.append({
                "model": model,
                "reason": "No se pudo generar ningún delta válido."
            })

    # ========================================================
    # CSV BASE
    # ========================================================

    summary_df = pd.DataFrame(summary_rows)
    rows_df = pd.DataFrame(rows_all)
    failed_df = pd.DataFrame(failed_models)
    skipped_df = pd.DataFrame(skipped_models)

    summary_csv = OUT_DIR / "highres_temperature_summary_annual_seasonal.csv"
    rows_csv = OUT_DIR / "highres_temperature_annual_seasonal_by_model.csv"
    failed_csv = OUT_DIR / "highres_temperature_failed_models.csv"
    skipped_csv = OUT_DIR / "highres_temperature_skipped_models.csv"

    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")
    rows_df.to_csv(rows_csv, index=False, encoding="utf-8-sig")
    failed_df.to_csv(failed_csv, index=False, encoding="utf-8-sig")
    skipped_df.to_csv(skipped_csv, index=False, encoding="utf-8-sig")

    print("\nCSV generados:")
    print(f" - {summary_csv}")
    print(f" - {rows_csv}")
    print(f" - {failed_csv}")
    print(f" - {skipped_csv}")

    # ========================================================
    # MULTIMODELO
    # ========================================================

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

    if len(threshold_rows) > 0:
        threshold_df = pd.DataFrame(threshold_rows)
        threshold_csv = OUT_DIR / "highres_temperature_hatching_threshold_annual_seasonal.csv"

        threshold_df.to_csv(
            threshold_csv,
            index=False,
            encoding="utf-8-sig"
        )

        print("\nCSV de umbral de achurado generado:")
        print(f" - {threshold_csv}")

    included_csv = OUT_DIR / "highres_temperature_models_included_multimodel.csv"
    pd.DataFrame(included_rows).to_csv(
        included_csv,
        index=False,
        encoding="utf-8-sig"
    )

    print("\nCSV de modelos incluidos en multimodelo:")
    print(f" - {included_csv}")

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