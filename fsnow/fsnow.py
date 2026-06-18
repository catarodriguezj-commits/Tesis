# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:20:09 2026

@author: catar
"""

import os
import re
import glob
import warnings
import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter

warnings.filterwarnings("ignore")


# ============================================================
# 1) CONFIGURACIÓN
# ============================================================

PR_DIR = "/home/miguel/Escritorio/Cata/CR2MET/CR2MET_pr_v2.5_day_1960-2021_005deg/pr"
TXN_DIR = "/home/miguel/Escritorio/Cata/CR2MET/CR2MET_txn_v2.5_day_1960-2021_005deg/txn"

PR_GLOB = os.path.join(PR_DIR, "**", "*.nc")
TXN_GLOB = os.path.join(TXN_DIR, "**", "*.nc")

# Shapefile. Si no existe esta ruta, el código intentará buscar Provincias.shp automáticamente.
SHAPE_CHILE = "/home/miguel/Escritorio/Cata/Provincias/Provincias.shp"

OUT_DIR = "/home/miguel/Escritorio/Cata/CR2MET/salida_fsnow"
os.makedirs(OUT_DIR, exist_ok=True)

OUT_NC = os.path.join(OUT_DIR, "CR2MET_v25_fsnow_1980_2014.nc")
OUT_PNG = os.path.join(OUT_DIR, "CR2MET_v25_fsnow_1980_2014_paper_style.png")
OUT_PDF = os.path.join(OUT_DIR, "CR2MET_v25_fsnow_1980_2014_paper_style.pdf")
OUT_CSV = os.path.join(OUT_DIR, "CR2MET_v25_fsnow_resumen_1980_2014.csv")

# Periodo histórico usado en el paper.
# En el suplemento se trabaja 1980-2014 como 34 water years.
USE_WATER_YEAR = True

if USE_WATER_YEAR:
    START_DATE = "1980-04-01"
    END_DATE = "2014-03-31"
else:
    START_DATE = "1980-01-01"
    END_DATE = "2014-12-31"

# Umbral para nieve usado por Vásquez et al.: precipitación cuando Tmean <= 2°C
T_SNOW = 2.0

# Para evitar valores raros en zonas con precipitación casi cero
MIN_P_ANNUAL = 1.0  # mm/año

# Dominio visual parecido al paper
MAP_EXTENT = [-76.5, -66.0, -56.8, -17.0]  # lon_min, lon_max, lat_min, lat_max

# Chunks para no reventar memoria
CHUNKS = {
    "time": 365,
    "lat": 200,
    "lon": 200,
    "latitude": 200,
    "longitude": 200,
}


# ============================================================
# 2) FUNCIONES AUXILIARES
# ============================================================

def list_nc_files(pattern, label):
    files = sorted(glob.glob(pattern, recursive=True))
    if len(files) == 0:
        raise FileNotFoundError(
            f"No encontré archivos NetCDF para {label} con el patrón:\n{pattern}"
        )

    print(f"\n{label}: {len(files)} archivo(s) encontrado(s)")
    for f in files[:5]:
        print("  -", os.path.basename(f))
    if len(files) > 5:
        print("  ...")

    return files


def open_mfdataset_safe(files, label):
    try:
        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            chunks=CHUNKS,
            parallel=True,
            decode_times=True,
            use_cftime=True
        )
    except Exception as e:
        print(f"\nAdvertencia: no pude abrir {label} con dask/chunks.")
        print("Intentaré abrir sin chunks.")
        print("Error original:", e)
        ds = xr.open_mfdataset(
            files,
            combine="by_coords",
            decode_times=True,
            use_cftime=True
        )

    return fix_coords(ds)


def fix_coords(ds):
    rename = {}
    all_names = set(list(ds.dims)) | set(list(ds.coords))

    for name in all_names:
        low = name.lower()
        if low in ["latitude", "latitud", "y"]:
            rename[name] = "lat"
        elif low in ["longitude", "longitud", "x"]:
            rename[name] = "lon"

    if rename:
        ds = ds.rename(rename)

    if "lon" in ds.coords:
        if float(ds["lon"].max()) > 180:
            ds = ds.assign_coords(lon=(((ds["lon"] + 180) % 360) - 180))
            ds = ds.sortby("lon")

    if "lat" in ds.coords:
        ds = ds.sortby("lat")

    return ds


def score_var_name(varname, candidates):
    name = varname.lower()
    score = 0
    for c in candidates:
        c = c.lower()
        if name == c:
            score += 100
        elif re.search(rf"(^|[_\-.]){re.escape(c)}($|[_\-.])", name):
            score += 50
        elif c in name:
            score += 10
    return score


def pick_variable(ds, candidates, label):
    data_vars = list(ds.data_vars)

    if len(data_vars) == 0:
        raise ValueError(f"El dataset de {label} no tiene variables de datos.")

    scores = [(v, score_var_name(v, candidates)) for v in data_vars]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)

    if scores[0][1] > 0:
        var = scores[0][0]
        print(f"Variable usada para {label}: {var}")
        return ds[var]

    if len(data_vars) == 1:
        var = data_vars[0]
        print(f"Variable usada para {label}: {var}")
        return ds[var]

    raise ValueError(
        f"No pude identificar la variable para {label}.\n"
        f"Variables disponibles: {data_vars}\n"
        f"Candidatas: {candidates}"
    )


def open_precipitation():
    files = list_nc_files(PR_GLOB, "precipitación CR2MET")
    ds = open_mfdataset_safe(files, "precipitación")
    pr = pick_variable(
        ds,
        candidates=["pr", "precip", "precipitation", "ppt", "prec"],
        label="precipitación"
    )
    return precip_to_mmday(pr)


def filename_tokens(path):
    base = os.path.basename(path).lower()
    return re.split(r"[^a-z0-9]+", base)


def filter_files_by_tokens(files, wanted_tokens):
    out = []
    for f in files:
        tokens = filename_tokens(f)
        if any(w in tokens for w in wanted_tokens):
            out.append(f)
    return out


def find_var_by_candidates(ds, candidates):
    data_vars = list(ds.data_vars)
    scores = [(v, score_var_name(v, candidates)) for v in data_vars]
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    if scores and scores[0][1] > 0:
        return scores[0][0]
    return None


def try_select_from_variable_dimension(ds, tmax_candidates, tmin_candidates):
    """
    Caso raro: un archivo con una sola variable y una dimensión tipo 'variable'
    donde vienen tx y tn juntos.
    """
    for varname in ds.data_vars:
        da = ds[varname]

        for dim in da.dims:
            if dim in ["time", "lat", "lon"]:
                continue

            if dim in da.coords:
                vals = [str(v).lower() for v in da.coords[dim].values]

                tmax_idx = None
                tmin_idx = None

                for i, v in enumerate(vals):
                    if any(c.lower() in v for c in tmax_candidates):
                        tmax_idx = i
                    if any(c.lower() in v for c in tmin_candidates):
                        tmin_idx = i

                if tmax_idx is not None and tmin_idx is not None:
                    print(f"Temperaturas encontradas dentro de la variable {varname}, dimensión {dim}")
                    tmax = da.isel({dim: tmax_idx})
                    tmin = da.isel({dim: tmin_idx})
                    return tmax, tmin

    return None, None


def open_temperature_txn():
    files = list_nc_files(TXN_GLOB, "temperatura CR2MET TXN")

    tmax_candidates = ["tmax", "tasmax", "tx", "Tmax", "TX", "maximum_temperature", "temp_max"]
    tmin_candidates = ["tmin", "tasmin", "tn", "Tmin", "TN", "minimum_temperature", "temp_min"]

    # Primero intenta abrir todo TXN junto.
    ds_all = open_mfdataset_safe(files, "temperatura TXN")

    tmax_var = find_var_by_candidates(ds_all, tmax_candidates)
    tmin_var = find_var_by_candidates(ds_all, tmin_candidates)

    if tmax_var is not None and tmin_var is not None and tmax_var != tmin_var:
        print(f"Variable usada para Tmax: {tmax_var}")
        print(f"Variable usada para Tmin: {tmin_var}")
        tmax = ds_all[tmax_var]
        tmin = ds_all[tmin_var]
        return temp_to_celsius(tmax, "Tmax"), temp_to_celsius(tmin, "Tmin")

    # Caso especial: una variable contiene ambas temperaturas.
    tmax, tmin = try_select_from_variable_dimension(ds_all, tmax_candidates, tmin_candidates)
    if tmax is not None and tmin is not None:
        return temp_to_celsius(tmax, "Tmax"), temp_to_celsius(tmin, "Tmin")

    # Si no pudo, intenta separar archivos por nombre.
    tmax_files = filter_files_by_tokens(files, ["tmax", "tasmax", "tx"])
    tmin_files = filter_files_by_tokens(files, ["tmin", "tasmin", "tn"])

    if len(tmax_files) > 0 and len(tmin_files) > 0:
        print("\nTemperaturas separadas por nombre de archivo.")
        ds_tx = open_mfdataset_safe(tmax_files, "Tmax")
        ds_tn = open_mfdataset_safe(tmin_files, "Tmin")

        tmax = pick_variable(ds_tx, tmax_candidates, "Tmax")
        tmin = pick_variable(ds_tn, tmin_candidates, "Tmin")

        return temp_to_celsius(tmax, "Tmax"), temp_to_celsius(tmin, "Tmin")

    raise ValueError(
        "\nNo pude identificar Tmax y Tmin dentro de TXN.\n"
        f"Variables encontradas: {list(ds_all.data_vars)}\n"
        "Abre un archivo con: ncdump -h archivo.nc | head -80\n"
        "y revisamos los nombres exactos."
    )


def precip_to_mmday(da):
    units = str(da.attrs.get("units", "")).lower().replace(" ", "")

    if ("kg" in units and "s-1" in units) or ("kgm-2s-1" in units):
        print("Convirtiendo precipitación de kg m-2 s-1 a mm/día")
        da = da * 86400.0
    elif "mm/s" in units or "mms-1" in units:
        print("Convirtiendo precipitación de mm/s a mm/día")
        da = da * 86400.0
    elif units in ["m", "m/day", "md-1", "m/d", "m/dia", "m/día"]:
        print("Convirtiendo precipitación de m/día a mm/día")
        da = da * 1000.0
    else:
        print("Precipitación asumida en mm/día o mm/día equivalente")

    da.attrs["units"] = "mm/day"
    return da


def temp_to_celsius(da, label):
    units = str(da.attrs.get("units", "")).lower()

    try:
        sample = float(
            da.isel(time=slice(0, min(20, da.sizes["time"])))
            .mean(skipna=True)
            .compute()
        )
    except Exception:
        sample = float(
            da.isel(time=slice(0, min(20, da.sizes["time"])))
            .mean(skipna=True)
        )

    if units.strip() in ["k", "kelvin"] or "kelvin" in units or sample > 100:
        print(f"Convirtiendo {label} de Kelvin a °C")
        da = da - 273.15
    else:
        print(f"{label} asumida en °C")

    da.attrs["units"] = "degC"
    return da


def ensure_lat_lon(da, label):
    if "lat" not in da.coords or "lon" not in da.coords:
        raise ValueError(
            f"{label} no tiene coordenadas lat/lon después de renombrar.\n"
            f"Dims: {da.dims}\nCoords: {list(da.coords)}"
        )

    if "lat" in da.dims and "lon" in da.dims:
        other_dims = [d for d in da.dims if d not in ["lat", "lon"]]
        da = da.transpose(*other_dims, "lat", "lon")

    return da


def add_year_coord(da):
    if USE_WATER_YEAR:
        wy = xr.where(
            da["time"].dt.month >= 4,
            da["time"].dt.year,
            da["time"].dt.year - 1
        )
        da = da.assign_coords(water_year=("time", wy.data))
        return da, "water_year"

    yr = da["time"].dt.year
    da = da.assign_coords(year=("time", yr.data))
    return da, "year"


def find_chile_shape():
    if SHAPE_CHILE is not None and os.path.exists(SHAPE_CHILE):
        return SHAPE_CHILE

    print("\nNo encontré SHAPE_CHILE en la ruta configurada.")
    print("Buscando Provincias.shp automáticamente dentro de /home/miguel/Escritorio/Cata ...")

    candidates = []

    common = [
        "/home/miguel/Escritorio/Cata/Provincias/Provincias.shp",
        "/home/miguel/Escritorio/Cata/CMIP5/Provincias/Provincias.shp",
        "/media/miguel/Disco/CMIP5/Provincias/Provincias.shp",
        "/media/miguel/Disco/Datos/CMIP5/Provincias/Provincias.shp",
    ]

    for p in common:
        if os.path.exists(p):
            candidates.append(p)

    candidates += glob.glob(
        "/home/miguel/Escritorio/Cata/**/Provincias.shp",
        recursive=True
    )

    candidates += glob.glob(
        "/home/miguel/Escritorio/Cata/**/*limite*internacional*.shp",
        recursive=True
    )

    candidates = list(dict.fromkeys(candidates))

    if len(candidates) == 0:
        raise FileNotFoundError(
            "No encontré shapefile de Chile.\n"
            "Edita la variable SHAPE_CHILE al inicio del script."
        )

    print("Usaré este shapefile:")
    print(candidates[0])
    return candidates[0]


def read_shape(path):
    gdf = gpd.read_file(path)

    if gdf.crs is None:
        print("Advertencia: el shapefile no tenía CRS. Asumo EPSG:4326.")
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:4326")
    gdf = gdf[gdf.geometry.notnull()]
    gdf = gdf[~gdf.geometry.is_empty]

    try:
        gdf["geometry"] = gdf.geometry.make_valid()
    except Exception:
        gdf["geometry"] = gdf.buffer(0)

    return gdf


def union_geometry(gdf):
    try:
        return gdf.geometry.union_all()
    except Exception:
        return gdf.geometry.unary_union


def mask_to_shape(da, shape_path):
    print("\nAplicando máscara espacial de Chile...")

    import shapely

    gdf = read_shape(shape_path)
    geom = union_geometry(gdf)

    da = ensure_lat_lon(da, "variable para máscara")

    lon = da["lon"].values
    lat = da["lat"].values

    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("El código espera coordenadas lat/lon 1D.")

    xx, yy = np.meshgrid(lon, lat)

    try:
        mask = shapely.contains_xy(geom, xx, yy)
    except Exception:
        from shapely import vectorized
        mask = vectorized.contains(geom, xx, yy)

    mask_da = xr.DataArray(
        mask,
        coords={"lat": lat, "lon": lon},
        dims=("lat", "lon")
    )

    return da.where(mask_da), gdf


def fmt_lon(x, pos=None):
    if x < 0:
        return f"{abs(x):.0f}°W"
    return f"{x:.0f}°E"


def fmt_lat(y, pos=None):
    if y < 0:
        return f"{abs(y):.0f}°S"
    return f"{y:.0f}°N"


def fsnow_cmap_paper_like():
    bounds = np.array([
        0.00, 0.05, 0.10, 0.20, 0.30, 0.40,
        0.50, 0.60, 0.70, 0.80, 0.90, 1.00
    ])

    colors = [
        "#ffffff",
        "#f2dfb3",
        "#d9902f",
        "#f1c93b",
        "#e5e62d",
        "#8dd744",
        "#2faa48",
        "#006b3a",
        "#276bb0",
        "#7a2ebf",
        "#000000",
    ]

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(bounds, cmap.N, clip=True)
    return cmap, norm, bounds


def plot_fsnow_paper_style(fsnow_da, chile_gdf):
    print("\nGenerando figura estilo paper...")

    fsnow_plot = ensure_lat_lon(fsnow_da, "fsnow").load()

    cmap, norm, bounds = fsnow_cmap_paper_like()

    plt.rcParams.update({
        "font.size": 9,
        "axes.linewidth": 0.8,
        "savefig.dpi": 400
    })

    fig = plt.figure(figsize=(3.25, 7.8))
    ax = fig.add_axes([0.14, 0.08, 0.58, 0.86])

    ax.set_facecolor("#d9d9d9")

    lon = fsnow_plot["lon"].values
    lat = fsnow_plot["lat"].values
    z = fsnow_plot.values

    mesh = ax.pcolormesh(
        lon,
        lat,
        z,
        cmap=cmap,
        norm=norm,
        shading="auto",
        zorder=2
    )

    chile_gdf.boundary.plot(
        ax=ax,
        color="0.25",
        linewidth=0.35,
        zorder=3
    )

    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    ax.set_xticks([-75, -72, -69])
    ax.set_yticks([-55, -50, -45, -40, -35, -30, -25, -20])

    ax.xaxis.set_major_formatter(FuncFormatter(fmt_lon))
    ax.yaxis.set_major_formatter(FuncFormatter(fmt_lat))

    ax.tick_params(direction="out", length=3, width=0.7)

    ax.text(
        0.03, 0.985,
        "d) Snowfall Frac.",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold"
    )

    cax = fig.add_axes([0.77, 0.30, 0.055, 0.42])
    cb = fig.colorbar(
        mesh,
        cax=cax,
        ticks=np.arange(0, 1.01, 0.1),
        boundaries=bounds
    )
    cb.ax.tick_params(labelsize=8, length=2.5, width=0.6)
    cb.outline.set_linewidth(0.6)

    fig.savefig(OUT_PNG, bbox_inches="tight", dpi=400)
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)

    print("Figura PNG guardada en:")
    print(OUT_PNG)
    print("Figura PDF guardada en:")
    print(OUT_PDF)


def save_summary_csv(fsnow_da, sf_da, p_da):
    import pandas as pd

    vals = {
        "period_start": START_DATE,
        "period_end": END_DATE,
        "water_year": USE_WATER_YEAR,
        "snow_threshold_degC": T_SNOW,
        "min_precip_mm_year_for_fsnow": MIN_P_ANNUAL,
        "fsnow_min": float(fsnow_da.min(skipna=True).compute()),
        "fsnow_max": float(fsnow_da.max(skipna=True).compute()),
        "fsnow_mean_spatial": float(fsnow_da.mean(skipna=True).compute()),
        "snowfall_mean_spatial_mm_year": float(sf_da.mean(skipna=True).compute()),
        "precip_mean_spatial_mm_year": float(p_da.mean(skipna=True).compute()),
    }

    df = pd.DataFrame([vals])
    df.to_csv(OUT_CSV, index=False)
    print("CSV resumen guardado en:")
    print(OUT_CSV)


# ============================================================
# 3) EJECUCIÓN
# ============================================================

print("\n======================================================")
print("CALCULO DE FSNOW CON CR2MET v2.5")
print("======================================================")
print(f"PR_DIR:  {PR_DIR}")
print(f"TXN_DIR: {TXN_DIR}")
print(f"Periodo: {START_DATE} a {END_DATE}")
print(f"Water year: {USE_WATER_YEAR}")
print(f"Umbral nieve: Tmean <= {T_SNOW} °C")
print("======================================================")

# Leer precipitación
pr = open_precipitation()

# Leer Tmax y Tmin desde carpeta TXN
tmax, tmin = open_temperature_txn()

# Asegurar lat/lon
pr = ensure_lat_lon(pr, "precipitación")
tmax = ensure_lat_lon(tmax, "Tmax")
tmin = ensure_lat_lon(tmin, "Tmin")

# Selección temporal
print("\nSeleccionando periodo...")
pr = pr.sel(time=slice(START_DATE, END_DATE))
tmax = tmax.sel(time=slice(START_DATE, END_DATE))
tmin = tmin.sel(time=slice(START_DATE, END_DATE))

# Alinear
print("Alineando pr, Tmax y Tmin...")
pr, tmax, tmin = xr.align(pr, tmax, tmin, join="inner")

print("\nDatos alineados:")
print("Tiempo inicial:", str(pr["time"].values[0]))
print("Tiempo final:  ", str(pr["time"].values[-1]))
print("Shape pr:", pr.shape)
print("Shape Tmax:", tmax.shape)
print("Shape Tmin:", tmin.shape)

# Calcular Tmean
print("\nCalculando Tmean = (Tmax + Tmin) / 2...")
tmean = (tmax + tmin) / 2.0
tmean.attrs["units"] = "degC"
tmean.attrs["long_name"] = "Mean daily temperature"

# Calcular snowfall diario
print("Calculando snowfall diario: pr cuando Tmean <= 2°C...")
snow_daily = pr.where(tmean <= T_SNOW, 0.0)
snow_daily.attrs["units"] = "mm/day"
snow_daily.attrs["long_name"] = "Daily snowfall estimated from precipitation and Tmean threshold"

# Agregar año o water year
print("Agrupando por año...")
pr, year_dim = add_year_coord(pr)
snow_daily, _ = add_year_coord(snow_daily)

# Acumulado anual
print("Calculando acumulados anuales...")
p_ann = pr.groupby(year_dim).sum("time", skipna=True)
sf_ann = snow_daily.groupby(year_dim).sum("time", skipna=True)

# Promedios anuales del periodo
print("Calculando promedio anual del periodo...")
p_ann_mean = p_ann.mean(year_dim, skipna=True)
sf_ann_mean = sf_ann.mean(year_dim, skipna=True)

p_ann_mean.attrs["long_name"] = "Mean annual precipitation"
p_ann_mean.attrs["units"] = "mm/year"

sf_ann_mean.attrs["long_name"] = "Mean annual snowfall"
sf_ann_mean.attrs["units"] = "mm/year"

# fsnow = Sf / P
print("Calculando fsnow = Sf / P...")
fsnow = sf_ann_mean / p_ann_mean
fsnow = fsnow.where(p_ann_mean >= MIN_P_ANNUAL)
fsnow = fsnow.clip(min=0.0, max=1.0)

fsnow.attrs["long_name"] = "Fraction of precipitation falling as snow"
fsnow.attrs["units"] = "1"
fsnow.attrs["formula"] = "fsnow = mean_annual_snowfall / mean_annual_precipitation"
fsnow.attrs["snowfall_definition"] = f"Sf = sum(pr when Tmean <= {T_SNOW} degC)"
fsnow.attrs["tmean_definition"] = "Tmean = (Tmax + Tmin) / 2"
fsnow.attrs["period"] = f"{START_DATE} to {END_DATE}"

# Máscara Chile
shape_path = find_chile_shape()

fsnow_masked, chile_gdf = mask_to_shape(fsnow, shape_path)
sf_ann_mean_masked, _ = mask_to_shape(sf_ann_mean, shape_path)
p_ann_mean_masked, _ = mask_to_shape(p_ann_mean, shape_path)

# Guardar NetCDF
print("\nGuardando NetCDF final...")

out_ds = xr.Dataset(
    {
        "fsnow": fsnow_masked.astype("float32"),
        "snowfall_annual_mean": sf_ann_mean_masked.astype("float32"),
        "precip_annual_mean": p_ann_mean_masked.astype("float32"),
    }
)

out_ds.attrs["description"] = "CR2MET v2.5 snowfall fraction calculated following Vasquez et al. 2025 approach"
out_ds.attrs["period"] = f"{START_DATE} to {END_DATE}"
out_ds.attrs["water_year"] = str(USE_WATER_YEAR)
out_ds.attrs["threshold_temperature_degC"] = T_SNOW
out_ds.attrs["formula"] = "fsnow = Sf/P; Sf = precipitation for days with Tmean <= 2 degC; Tmean=(Tmax+Tmin)/2"

encoding = {
    var: {
        "zlib": True,
        "complevel": 4,
        "dtype": "float32",
        "_FillValue": np.float32(-9999.0)
    }
    for var in out_ds.data_vars
}

out_ds.to_netcdf(OUT_NC, encoding=encoding)

print("NetCDF guardado en:")
print(OUT_NC)

# Guardar CSV resumen
save_summary_csv(fsnow_masked, sf_ann_mean_masked, p_ann_mean_masked)

# Generar mapa
plot_fsnow_paper_style(fsnow_masked, chile_gdf)

# Resumen final
print("\n======================================================")
print("PROCESO TERMINADO")
print("======================================================")
print(f"Periodo usado: {START_DATE} a {END_DATE}")
print(f"Tipo de año: {'water year abril-marzo' if USE_WATER_YEAR else 'año calendario'}")
print(f"Umbral nieve: Tmean <= {T_SNOW} °C")
print(f"Años usados: {len(p_ann[year_dim])}")
print("------------------------------------------------------")
print("fsnow mínimo:", float(fsnow_masked.min(skipna=True).compute()))
print("fsnow máximo:", float(fsnow_masked.max(skipna=True).compute()))
print("fsnow promedio espacial:", float(fsnow_masked.mean(skipna=True).compute()))
print("------------------------------------------------------")
print("Salidas:")
print(OUT_NC)
print(OUT_PNG)
print(OUT_PDF)
print(OUT_CSV)
print("======================================================")