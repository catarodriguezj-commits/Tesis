# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 09:51:40 2026

@author: catar
"""

# -*- coding: utf-8 -*-
"""
PRECIPITACIÓN HIGHRES

Calcula:
  - Acumulado anual de precipitación por modelo.
  - Promedio histórico de acumulados anuales.
  - Promedio futuro de acumulados anuales.
  - Diferencia absoluta: futuro - histórico, en mm/año.
  - Cambio porcentual: (futuro - histórico) / histórico * 100.
  - Promedio multimodelo.
  - Achurado donde los modelos coinciden en el signo del cambio.
  - Máscara de Chile para no considerar océano.
  - CSV anual y resumen.
  - Mapas PNG.
  - NetCDF de resultados.

Colores:
  - Más precipitación: azul.
  - Menos precipitación: rojo/naranjo.
  - Cero: blanco.
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


# ============================================================
# 1. CONFIGURACIÓN
# ============================================================

# Carpeta con archivos históricos HighRes
HIST_DIR = Path(r"E:\Cata\HighRes\pr_hist\RESAMPLE_FILES\Regrid_005_conservative")

# Carpeta con archivos futuros HighRes
FUT_DIR = Path(r"E:\Cata\HighRes\pr_futuro\RESAMPLE_FILES\Regrid_005_conservative")

# Shapefile de Chile. Debe apuntar al archivo .shp
SHAPEFILE = Path(r"C:/Users/catar/OneDrive - Universidad Técnica Federico Santa María/Universidad/2026/Seminario de Tesis I/Provincias/Provincias.shp")

# Carpeta de salida
OUT_DIR = Path(r"E:/Cata/HighRes/graf_pr")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Periodos
HIST_START_YEAR = 1980
HIST_END_YEAR   = 2010

FUT_START_YEAR  = 2020
FUT_END_YEAR    = 2050

HIST_LABEL = f"{HIST_START_YEAR}-{HIST_END_YEAR}"
FUT_LABEL  = f"{FUT_START_YEAR}-{FUT_END_YEAR}"

# Variable de precipitación
PR_VAR_CANDIDATES = ["pr", "precip", "precipitation", "PR"]

# Crear mapas
MAKE_MAPS = True

# Extensión aproximada Chile continental: lon_min, lon_max, lat_min, lat_max
CHILE_EXTENT = (-76, -66, -56, -16)

# Escala porcentual fija: -10% a +10%
PCT_LIM_FIXED = 10

# Escala mm/año.
# Si lo dejas en None, calcula una escala ajustada automáticamente.
# Si quieres forzar, por ejemplo:
# DELTA_MM_LIM_FIXED = 300
DELTA_MM_LIM_FIXED = None

# Percentil para ajustar escala de mm/año y evitar extremos
MM_SCALE_PERCENTILE = 97

# Redondeo de escala mm/año a múltiplos de este valor
MM_SCALE_ROUND_TO = 50

# Achurado
AGREEMENT_PCT = 0.70


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
    """
    Umbral de modelos para achurado.
    """
    return max(1, round_half_up(n_models * pct))


def safe_name(text):
    """
    Limpia nombres para archivos.
    """
    return re.sub(r"[^A-Za-z0-9_\-]+", "_", str(text))


def extract_model_name(file_path):
    """
    Extrae el nombre del modelo desde nombres tipo:

    pr_day_HadGEM3-GC31-LL_highres-future_r1i1p1f1_gn_...
    pr_day_ACCESS1-3_historical_r1i1p1_...

    Devuelve el bloque después de pr_day_.
    """
    name = Path(file_path).name
    name = re.sub(r"\.nc4?$", "", name, flags=re.IGNORECASE)

    parts = name.split("_")

    if len(parts) >= 3 and parts[0].lower() == "pr":
        return parts[2]

    pattern = r"pr_(?:day|Amon|mon|3hr|6hr)_([^_]+)_"
    match = re.search(pattern, name, flags=re.IGNORECASE)

    if match:
        return match.group(1)

    return Path(file_path).parent.name


def find_files_by_model(folder):
    """
    Busca archivos .nc y .nc4 y los agrupa por modelo.
    """
    folder = Path(folder)

    if not folder.exists():
        raise FileNotFoundError(f"No existe la carpeta: {folder}")

    files = sorted(list(folder.rglob("*.nc")) + list(folder.rglob("*.nc4")))

    by_model = {}

    for f in files:
        if "pr" not in f.name.lower():
            continue

        model = extract_model_name(f)
        by_model.setdefault(model, []).append(f)

    return by_model


def open_dataset_safe(path):
    """
    Abre NetCDF de forma robusta.
    """
    try:
        return xr.open_dataset(path, decode_times=True, use_cftime=True)
    except TypeError:
        return xr.open_dataset(path, decode_times=True)
    except Exception:
        return xr.open_dataset(path, decode_times=True)


def standardize_coords(da):
    """
    Normaliza nombres frecuentes de coordenadas.
    """
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

    # Longitud 0-360 a -180/180 si lon es 1D
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
    Elimina coordenadas auxiliares conflictivas: height, bounds, time_bnds, etc.
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
    """
    Selecciona variable de precipitación.
    """
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


def precip_multiplier_to_mm_per_day(da):
    """
    Determina factor para pasar precipitación a mm/día.

    CMIP normalmente viene en:
      kg m-2 s-1 = mm/s
    Entonces:
      mm/día = kg m-2 s-1 * 86400
    """
    units = str(da.attrs.get("units", "")).lower().strip()
    units_clean = units.replace(" ", "").replace("^", "")

    if units_clean in [
        "kgm-2s-1",
        "kg/m2/s",
        "kgm**-2s**-1",
        "kgm-2sec-1",
        "mm/s",
        "mms-1"
    ]:
        return 86400.0

    if units_clean in [
        "mm/day",
        "mm/d",
        "mmday-1",
        "mm/day-1",
        "mm"
    ]:
        return 1.0

    # Respaldo típico CMIP: si la variable se llama pr y no sabemos,
    # asumimos kg m-2 s-1
    if "kg" in units and "s" in units:
        return 86400.0

    return 1.0


def spatial_mean(da):
    """
    Promedio espacial ponderado por cos(lat), ignorando NaN.
    Como la máscara de Chile deja océano en NaN, el océano no entra.
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
# 3. MÁSCARA DE CHILE ROBUSTA
# Arregla geometrías inválidas y, si no puede unirlas,
# usa una lista de polígonos para aplicar la máscara igual.
# ============================================================

_CHILE_GEOM = None
_MASK_CACHE = {}


def fix_geometry_safe(geom):
    """
    Intenta reparar geometrías inválidas del shapefile.
    Sirve para errores tipo:
    TopologyException: side location conflict
    """
    if geom is None:
        return None

    try:
        if geom.is_empty:
            return None
    except Exception:
        return None

    # Intento 1: shapely.make_valid, si existe
    try:
        if hasattr(shapely, "make_valid"):
            geom = shapely.make_valid(geom)
    except Exception:
        pass

    # Intento 2: buffer(0), clásico para reparar polígonos
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
    """
    Si la geometría reparada queda como GeometryCollection,
    extrae solo Polygon y MultiPolygon.
    """
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
    Carga shapefile de Chile de forma robusta.

    Si las geometrías son válidas, intenta unirlas.
    Si la unión falla por problemas topológicos, devuelve una lista
    de polígonos limpios y la máscara se aplica igual.
    """
    global _CHILE_GEOM

    if shapefile is None:
        print("No se entregó shapefile. No se aplicará máscara de Chile.")
        return None

    if gpd is None:
        print("Advertencia: geopandas no está instalado. No se aplicará máscara.")
        return None

    shapefile = Path(shapefile)

    if not shapefile.exists():
        print(f"Advertencia: no existe shapefile: {shapefile}")
        print("No se aplicará máscara de Chile.")
        return None

    print("Cargando shapefile de Chile...")

    try:
        gdf = gpd.read_file(shapefile)
    except Exception as e:
        print("Advertencia: no se pudo leer el shapefile.")
        print(f"Motivo: {e}")
        print("No se aplicará máscara de Chile.")
        return None

    try:
        if gdf.crs is not None:
            gdf = gdf.to_crs("EPSG:4326")
    except Exception as e:
        print("Advertencia: no se pudo reproyectar el shapefile a EPSG:4326.")
        print(f"Motivo: {e}")

    # Eliminar geometrías nulas
    gdf = gdf[gdf.geometry.notnull()].copy()

    if len(gdf) == 0:
        print("Advertencia: el shapefile no tiene geometrías válidas.")
        return None

    cleaned_polygons = []

    print("Reparando geometrías del shapefile...")

    for geom in gdf.geometry:
        fixed = fix_geometry_safe(geom)
        parts = keep_polygonal_parts(fixed)
        cleaned_polygons.extend(parts)

    if len(cleaned_polygons) == 0:
        print("Advertencia: no quedaron polígonos válidos después de reparar.")
        print("No se aplicará máscara de Chile.")
        return None

    print(f"Polígonos válidos para máscara: {len(cleaned_polygons)}")

    # Intentar unir. Si falla, se usa la lista de polígonos.
    try:
        from shapely.ops import unary_union

        geom_union = unary_union(cleaned_polygons)

        try:
            geom_union = fix_geometry_safe(geom_union)
        except Exception:
            pass

        if geom_union is not None:
            _CHILE_GEOM = geom_union
            print("Máscara de Chile cargada correctamente usando geometría unida.")
            return geom_union

    except Exception as e:
        print("Advertencia: no se pudieron unir las geometrías del shapefile.")
        print(f"Motivo: {e}")
        print("Se usará la lista de polígonos por separado para la máscara.")

    _CHILE_GEOM = cleaned_polygons

    print("Máscara de Chile cargada correctamente usando polígonos separados.")

    return cleaned_polygons


def get_xy_for_dataarray(da):
    """
    Devuelve X, Y 2D para una grilla lat/lon.
    """
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
    """
    Evalúa si los puntos están dentro de una geometría.
    Funciona con una geometría única o con lista de polígonos.
    """
    if geom is None:
        return np.ones(x.shape, dtype=bool)

    # Si geom es lista de polígonos, hacemos OR entre todos
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
                            temp[i, j] = g.contains(Point(float(x[i, j]), float(y[i, j])))

                    mask |= temp

            except Exception:
                continue

        return mask

    # Si es una sola geometría
    try:
        if shapely is not None and hasattr(shapely, "contains_xy"):
            return shapely.contains_xy(geom, x, y)
    except Exception:
        pass

    # Respaldo lento
    from shapely.geometry import Point

    mask = np.zeros(x.shape, dtype=bool)

    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            try:
                mask[i, j] = geom.contains(Point(float(x[i, j]), float(y[i, j])))
            except Exception:
                mask[i, j] = False

    return mask


def build_chile_mask(da, chile_geom):
    """
    Construye máscara booleana para Chile.
    True = dentro de Chile.
    False = fuera de Chile.
    """
    da = standardize_coords(da)
    da = da.squeeze()

    if chile_geom is None:
        return xr.ones_like(da, dtype=bool)

    if "lat" in da.dims and "lon" in da.dims:
        da_ref = da.transpose("lat", "lon")
    else:
        da_ref = da

    x, y = get_xy_for_dataarray(da_ref)

    # Cache para no reconstruir la máscara en cada año/modelo
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
    """
    Aplica máscara de Chile.
    Todo lo que queda fuera de Chile queda como NaN.
    """
    if chile_geom is None:
        return da

    da = standardize_coords(da)
    da = clean_for_concat(da)

    if "lat" not in da.coords or "lon" not in da.coords:
        return da

    mask = build_chile_mask(da, chile_geom)

    try:
        da_masked = da.where(mask)
    except Exception:
        da_aligned, mask_aligned = xr.align(da, mask, join="inner")
        da_masked = da_aligned.where(mask_aligned)

    return da_masked


# ============================================================
# 4. CÁLCULO ANUAL DE PRECIPITACIÓN
# ============================================================

def compute_annual_accum_and_period_mean(
    files,
    model,
    period_label,
    start_year,
    end_year,
    chile_geom=None
):
    """
    Calcula precipitación acumulada anual y promedio del periodo.

    Lógica:
      1. Abre cada archivo.
      2. Detecta años.
      3. Procesa año por año.
      4. Convierte a mm/día si viene en kg m-2 s-1.
      5. Suma días del año = mm/año.
      6. Aplica máscara de Chile.
      7. Usa cada año como un punto.
      8. Promedia los acumulados anuales del periodo.
    """
    annual_maps = []
    annual_years = []
    annual_rows = []
    processed_years = set()

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
                var_name = select_variable(ds, PR_VAR_CANDIDATES)

                da = ds[var_name]
                da = standardize_coords(da)
                da = da.squeeze(drop=True)
                da = clean_for_concat(da)

                if "time" not in da.dims:
                    print(f"      Advertencia: {f.name} no tiene dimensión time. Se omite.")
                    continue

                factor = precip_multiplier_to_mm_per_day(da)

                years = da["time"].dt.year.values.astype(int)
                unique_years = sorted(np.unique(years))

                for yy in unique_years:
                    if yy < start_year or yy > end_year:
                        continue

                    if yy in processed_years:
                        print(f"      Advertencia: año {yy} duplicado en {model}. Se omite.")
                        continue

                    idx = np.where(years == yy)[0]

                    if len(idx) == 0:
                        continue

                    print(f"         Procesando año {yy}")

                    sub = da.isel(time=idx)

                    # Acumulado anual en mm/año
                    annual_accum = sub.sum(dim="time", skipna=True) * factor
                    annual_accum = annual_accum.load()

                    annual_accum = clean_for_concat(annual_accum)
                    annual_accum = apply_chile_mask(annual_accum, chile_geom)

                    annual_accum.name = f"pr_{period_label}_{model}_{yy}"
                    annual_accum.attrs["units"] = "mm/year"

                    annual_maps.append(annual_accum)
                    annual_years.append(int(yy))
                    processed_years.add(int(yy))

                    annual_rows.append({
                        "modelo": model,
                        "periodo": period_label,
                        "anio": int(yy),
                        "precipitacion_media_espacial_mm_anio": spatial_mean(annual_accum),
                    })

            except Exception as e:
                print(f"      Advertencia: falló archivo {f.name}. Se omite.")
                print(f"      Motivo: {e}")
                continue

    if not annual_maps:
        print(
            f"      Advertencia: no hay años válidos para {model} - {period_label} "
            f"en {start_year}-{end_year}."
        )
        return None, []

    annual_stack = xr.concat(
        annual_maps,
        dim=xr.DataArray(annual_years, dims="year", name="year"),
        join="inner",
        coords="minimal",
        compat="override"
    )

    period_mean = annual_stack.mean(dim="year", skipna=True)
    period_mean = clean_for_concat(period_mean)
    period_mean = apply_chile_mask(period_mean, chile_geom)

    period_mean.name = f"pr_{period_label}_{model}_mean"
    period_mean.attrs["units"] = "mm/year"

    return period_mean, annual_rows


# ============================================================
# 5. COLORES Y MAPAS
# ============================================================

def make_precip_delta_cmap():
    """
    Colormap para cambio de precipitación:

      Menos precipitación  -> rojo/naranjo
      Cero                -> blanco
      Más precipitación   -> azul
    """
    colors = [
        (0.00, "#7F0000"),  # rojo oscuro, mucho menos
        (0.18, "#D7301F"),  # rojo
        (0.34, "#FDB863"),  # naranjo
        (0.50, "#FFFFFF"),  # cero
        (0.66, "#C6DBEF"),  # celeste claro
        (0.82, "#4292C6"),  # azul
        (1.00, "#08306B"),  # azul oscuro, mucho más
    ]

    return LinearSegmentedColormap.from_list(
        "precip_delta_less_red_more_blue",
        colors,
        N=256
    )


def get_xy_for_plot(da):
    """
    Prepara X, Y, Z para pcolormesh.
    """
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


def calculate_mm_limit(dataarrays):
    """
    Escala automática ajustada para mapas en mm/año.
    Evita que valores extremos fijen una escala demasiado amplia.
    """
    if DELTA_MM_LIM_FIXED is not None:
        return float(abs(DELTA_MM_LIM_FIXED))

    values = []

    for da in dataarrays:
        arr = np.asarray(da.values)
        arr = arr[np.isfinite(arr)]

        if arr.size > 0:
            values.append(arr)

    if not values:
        return 100.0

    all_values = np.concatenate(values)

    lim = np.nanpercentile(np.abs(all_values), MM_SCALE_PERCENTILE)

    if not np.isfinite(lim) or lim <= 0:
        lim = 100.0

    lim = np.ceil(lim / MM_SCALE_ROUND_TO) * MM_SCALE_ROUND_TO

    return float(lim)


def plot_precip_map(
    da,
    title,
    output_path,
    lim,
    units_label,
    gdf=None,
    agreement_mask=None
):
    """
    Mapa estilo Chile vertical, cero blanco y achurado.
    """
    cmap = make_precip_delta_cmap()
    norm = TwoSlopeNorm(vmin=-lim, vcenter=0, vmax=lim)

    x, y, z = get_xy_for_plot(da)

    fig, ax = plt.subplots(figsize=(8, 13))

    im = ax.pcolormesh(
        x,
        y,
        z,
        cmap=cmap,
        norm=norm,
        shading="auto"
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
            alpha=0
        )

    if gdf is not None:
        gdf.boundary.plot(
            ax=ax,
            linewidth=0.7,
            color="black"
        )

    lon_min, lon_max, lat_min, lat_max = CHILE_EXTENT
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)

    ax.set_title(title, fontsize=15, pad=10)
    ax.set_xlabel("Longitud", fontsize=12)
    ax.set_ylabel("Latitud", fontsize=12)
    ax.tick_params(axis="both", labelsize=11)

    cbar = fig.colorbar(
        im,
        ax=ax,
        shrink=0.82,
        pad=0.08,
        extend="both"
    )

    cbar.set_label(
        units_label,
        fontsize=12,
        rotation=90,
        labelpad=18
    )

    cbar.ax.tick_params(labelsize=11)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def load_shape_for_plot(shapefile):
    """
    Carga shapefile para dibujar borde en mapas.
    """
    if shapefile is None:
        return None

    if gpd is None:
        return None

    shapefile = Path(shapefile)

    if not shapefile.exists():
        return None

    gdf = gpd.read_file(shapefile)

    try:
        if gdf.crs is not None:
            gdf = gdf.to_crs("EPSG:4326")
    except Exception:
        pass

    return gdf


# ============================================================
# 6. PROCESAMIENTO PRINCIPAL
# ============================================================

def main():
    print("====================================================")
    print("PRECIPITACIÓN HIGHRES")
    print("Acumulado anual -> promedio periodo -> delta")
    print("Delta = futuro - histórico")
    print("Más precipitación = azul | Menos precipitación = rojo")
    print("Máscara de Chile activada")
    print("====================================================")

    print("\nCarpetas:")
    print(f"Histórico: {HIST_DIR}")
    print(f"Futuro:    {FUT_DIR}")
    print(f"Salida:    {OUT_DIR}")

    print("\nPeriodos:")
    print(f"Histórico: {HIST_LABEL}")
    print(f"Futuro:    {FUT_LABEL}")

    chile_geom = load_chile_geometry(SHAPEFILE)
    gdf_plot = load_shape_for_plot(SHAPEFILE)

    print("\nBuscando modelos históricos...")
    hist_models = find_files_by_model(HIST_DIR)

    print("Buscando modelos futuros...")
    fut_models = find_files_by_model(FUT_DIR)

    common_models = sorted(set(hist_models.keys()) & set(fut_models.keys()))

    if not common_models:
        print("\nModelos históricos encontrados:")
        print(sorted(hist_models.keys()))

        print("\nModelos futuros encontrados:")
        print(sorted(fut_models.keys()))

        raise ValueError("No se encontraron modelos comunes.")

    print(f"\nModelos comunes encontrados: {len(common_models)}")
    for m in common_models:
        print(f" - {m}")

    annual_rows_all = []
    summary_rows = []

    delta_mm_by_model = {}
    delta_pct_by_model = {}

    hist_mean_by_model = {}
    fut_mean_by_model = {}

    for model in common_models:
        print("\n----------------------------------------------------")
        print(f"Procesando modelo: {model}")
        print("----------------------------------------------------")

        print("   Histórico")
        hist_mean, hist_rows = compute_annual_accum_and_period_mean(
            files=hist_models[model],
            model=model,
            period_label="historico",
            start_year=HIST_START_YEAR,
            end_year=HIST_END_YEAR,
            chile_geom=chile_geom
        )

        print("   Futuro")
        fut_mean, fut_rows = compute_annual_accum_and_period_mean(
            files=fut_models[model],
            model=model,
            period_label="futuro",
            start_year=FUT_START_YEAR,
            end_year=FUT_END_YEAR,
            chile_geom=chile_geom
        )

        if hist_mean is None or fut_mean is None:
            print(f"   Advertencia: se omite {model} porque falta histórico o futuro válido.")
            continue

        try:
            hist_mean, fut_mean = xr.align(hist_mean, fut_mean, join="inner")

            delta_mm = fut_mean - hist_mean
            delta_mm = clean_for_concat(delta_mm)
            delta_mm = apply_chile_mask(delta_mm, chile_geom)
            delta_mm.name = f"delta_pr_mm_{model}"
            delta_mm.attrs["units"] = "mm/year"

            delta_pct = xr.where(
                np.abs(hist_mean) > 1e-6,
                (delta_mm / hist_mean) * 100.0,
                np.nan
            )

            delta_pct = clean_for_concat(delta_pct)
            delta_pct = apply_chile_mask(delta_pct, chile_geom)
            delta_pct.name = f"delta_pr_pct_{model}"
            delta_pct.attrs["units"] = "%"

            hist_mean_by_model[model] = hist_mean
            fut_mean_by_model[model] = fut_mean

            delta_mm_by_model[model] = delta_mm
            delta_pct_by_model[model] = delta_pct

            annual_rows_all.extend(hist_rows)
            annual_rows_all.extend(fut_rows)

            summary_rows.append({
                "modelo": model,
                "n_anios_historico": len({r["anio"] for r in hist_rows}),
                "n_anios_futuro": len({r["anio"] for r in fut_rows}),
                "historico_promedio_espacial_mm_anio": spatial_mean(hist_mean),
                "futuro_promedio_espacial_mm_anio": spatial_mean(fut_mean),
                "delta_futuro_menos_historico_mm_anio": spatial_mean(delta_mm),
                "delta_porcentual_espacial_pct": spatial_mean(delta_pct),
            })

        except Exception as e:
            print(f"   Advertencia: falló cálculo de delta para {model}. Se omite.")
            print(f"   Motivo: {e}")
            continue

    if not delta_mm_by_model:
        raise ValueError("No se pudo procesar ningún modelo válido.")

    # ========================================================
    # CSV
    # ========================================================

    annual_df = pd.DataFrame(annual_rows_all)
    summary_df = pd.DataFrame(summary_rows)

    annual_csv = OUT_DIR / "precipitacion_anual_modelos.csv"
    summary_csv = OUT_DIR / "precipitacion_resumen_modelos.csv"

    annual_df.to_csv(annual_csv, index=False, encoding="utf-8-sig")
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8-sig")

    print("\nCSV generados:")
    print(f" - {annual_csv}")
    print(f" - {summary_csv}")

    # ========================================================
    # MULTIMODELO
    # ========================================================

    models = sorted(delta_mm_by_model.keys())

    delta_mm_list = [clean_for_concat(delta_mm_by_model[m]) for m in models]
    delta_pct_list = [clean_for_concat(delta_pct_by_model[m]) for m in models]

    aligned_mm = xr.align(*delta_mm_list, join="inner")
    aligned_pct = xr.align(*delta_pct_list, join="inner")

    aligned_mm = [apply_chile_mask(clean_for_concat(a), chile_geom) for a in aligned_mm]
    aligned_pct = [apply_chile_mask(clean_for_concat(a), chile_geom) for a in aligned_pct]

    stack_mm = xr.concat(
        aligned_mm,
        dim=xr.DataArray(models, dims="model", name="model"),
        coords="minimal",
        compat="override",
        join="inner"
    )

    stack_pct = xr.concat(
        aligned_pct,
        dim=xr.DataArray(models, dims="model", name="model"),
        coords="minimal",
        compat="override",
        join="inner"
    )

    stack_mm.name = "delta_pr_mm_por_modelo"
    stack_pct.name = "delta_pr_pct_por_modelo"

    multimodel_mm = stack_mm.mean(dim="model", skipna=True)
    multimodel_pct = stack_pct.mean(dim="model", skipna=True)

    multimodel_mm = apply_chile_mask(clean_for_concat(multimodel_mm), chile_geom)
    multimodel_pct = apply_chile_mask(clean_for_concat(multimodel_pct), chile_geom)

    multimodel_mm.name = "delta_pr_mm_multimodelo"
    multimodel_pct.name = "delta_pr_pct_multimodelo"

    n_models = len(models)
    threshold = agreement_threshold(n_models, AGREEMENT_PCT)

    n_positive = (stack_mm > 0).sum(dim="model")
    n_negative = (stack_mm < 0).sum(dim="model")

    agreement_mask = (n_positive >= threshold) | (n_negative >= threshold)
    agreement_mask = apply_chile_mask(clean_for_concat(agreement_mask), chile_geom)
    agreement_mask.name = "achurado_acuerdo_signo"

    actual_pct = int(np.ceil((threshold / n_models) * 100))

    threshold_df = pd.DataFrame([{
        "n_modelos_usados": n_models,
        "valor_70_pct": n_models * AGREEMENT_PCT,
        "umbral_usado_para_achurado": threshold,
        "porcentaje_real_equivalente": actual_pct,
        "regla": "redondeo normal: 4.2->4, 4.6->5, 4.5->5"
    }])

    threshold_csv = OUT_DIR / "precipitacion_umbral_achurado.csv"
    threshold_df.to_csv(threshold_csv, index=False, encoding="utf-8-sig")

    print("\nAchurado:")
    print(f"Modelos usados: {n_models}")
    print(f"70% de modelos: {n_models * AGREEMENT_PCT}")
    print(f"Umbral usado: {threshold}")
    print(f"Porcentaje real: ≥{actual_pct}%")

    # ========================================================
    # NetCDF
    # ========================================================

    stack_mm.to_netcdf(OUT_DIR / "precipitacion_delta_mm_por_modelo.nc")
    stack_pct.to_netcdf(OUT_DIR / "precipitacion_delta_pct_por_modelo.nc")
    multimodel_mm.to_netcdf(OUT_DIR / "precipitacion_delta_mm_multimodelo.nc")
    multimodel_pct.to_netcdf(OUT_DIR / "precipitacion_delta_pct_multimodelo.nc")
    agreement_mask.astype("int8").to_netcdf(OUT_DIR / "precipitacion_achurado_acuerdo.nc")

    print("\nNetCDF generados.")

    # ========================================================
    # MAPAS
    # ========================================================

    if MAKE_MAPS:
        maps_mm_dir = OUT_DIR / "mapas_delta_mm"
        maps_pct_dir = OUT_DIR / "mapas_delta_pct"

        maps_mm_dir.mkdir(parents=True, exist_ok=True)
        maps_pct_dir.mkdir(parents=True, exist_ok=True)

        mm_lim = calculate_mm_limit(list(aligned_mm) + [multimodel_mm])
        pct_lim = PCT_LIM_FIXED

        print(f"\nEscala mm/año usada: {-mm_lim} a {mm_lim}")
        print(f"Escala % usada: {-pct_lim}% a {pct_lim}%")

        # Mapas por modelo en mm
        for model, da in zip(models, aligned_mm):
            title = (
                f"Diferencia promedio de precipitación\n"
                f"{FUT_LABEL} - {HIST_LABEL}, modelo {model}"
            )

            output_png = maps_mm_dir / f"delta_pr_mm_{safe_name(model)}.png"

            plot_precip_map(
                da=da,
                title=title,
                output_path=output_png,
                lim=mm_lim,
                units_label="Futuro - Histórico (mm/año)",
                gdf=gdf_plot,
                agreement_mask=None
            )

        # Mapa multimodelo en mm con achurado
        title = (
            f"Diferencia promedio multimodelo de precipitación\n"
            f"{FUT_LABEL} - {HIST_LABEL}, n={n_models}\n"
            f"Achurado: ≥{actual_pct}% modelos coinciden en signo"
        )

        output_png = maps_mm_dir / "delta_pr_mm_multimodelo_achurado.png"

        plot_precip_map(
            da=multimodel_mm,
            title=title,
            output_path=output_png,
            lim=mm_lim,
            units_label="Futuro - Histórico (mm/año)",
            gdf=gdf_plot,
            agreement_mask=agreement_mask
        )

        # Mapas por modelo en %
        for model, da in zip(models, aligned_pct):
            title = (
                f"Cambio porcentual promedio de precipitación\n"
                f"{FUT_LABEL} - {HIST_LABEL}, modelo {model}"
            )

            output_png = maps_pct_dir / f"delta_pr_pct_{safe_name(model)}.png"

            plot_precip_map(
                da=da,
                title=title,
                output_path=output_png,
                lim=pct_lim,
                units_label="Cambio porcentual (%)",
                gdf=gdf_plot,
                agreement_mask=None
            )

        # Mapa multimodelo en % con achurado
        title = (
            f"Cambio porcentual promedio multimodelo de precipitación\n"
            f"{FUT_LABEL} - {HIST_LABEL}, n={n_models}\n"
            f"Achurado: ≥{actual_pct}% modelos coinciden en signo"
        )

        output_png = maps_pct_dir / "delta_pr_pct_multimodelo_achurado.png"

        plot_precip_map(
            da=multimodel_pct,
            title=title,
            output_path=output_png,
            lim=pct_lim,
            units_label="Cambio porcentual (%)",
            gdf=gdf_plot,
            agreement_mask=agreement_mask
        )

        print("\nMapas generados:")
        print(f" - {maps_mm_dir}")
        print(f" - {maps_pct_dir}")

    print("\n====================================================")
    print("PROCESO TERMINADO")
    print("====================================================")
    print(f"Carpeta de salida: {OUT_DIR}")


if __name__ == "__main__":
    main()