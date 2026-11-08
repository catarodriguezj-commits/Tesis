# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:31:33 2026

@author: DESKTOP
"""

"""
Winter (JJA) precipitation share map for Chile: climatological winter
(June-July-August) accumulated precipitation divided by the climatological
annual accumulated precipitation, expressed as a percentage.

Expected input data (same as the annual precipitation script):
  - Provincias.shp            : Chile boundary shapefile
  - Monthly CR2MET files      : one .nc file per month, all in DATA_DIR,
                                 following the CR2MET naming pattern:
                                 CR2MET_pr_v2.5_day_{year}_{month:02d}_005deg.nc
                                 Each file has a daily "pr" variable in mm/day
                                 and a "cl_mask" variable marking continental Chile.

What the script does:
  1. Scans DATA_DIR for all monthly files matching the naming pattern
     (does NOT require complete years -- every available month is used,
     nothing is filled with zero).
  2. For each of the 12 calendar months, averages that month's total
     precipitation across all years that have data for that month
     (monthly climatology).
  3. Winter climatology  = sum of the June, July and August monthly means.
     Annual climatology  = sum of all 12 monthly means.
     Winter share (%)    = 100 * winter climatology / annual climatology.
  4. Plots the winter share map using the same visual style as the annual
     precipitation map: neighboring countries in gray with bold borders,
     colorbar placed on the ocean side (west) of the map, discrete color
     levels with extend="both".

NOTE on the shapefile: Provincias.shp is missing .shx, .dbf and .prj.
The .shx is reconstructed automatically (SHAPE_RESTORE_SHX) and EPSG:3857
is assumed for its CRS given the coordinate range.
"""

import os
import re
import glob

os.environ["SHAPE_RESTORE_SHX"] = "YES"

import numpy as np
import geopandas as gpd
import xarray as xr
import rioxarray
import requests
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable

# --------------------------------------------------------------------------
# 1. Inputs
# --------------------------------------------------------------------------
SHAPEFILE_PATH = r"C:\Users\DESKTOP\Desktop\Tesis\macrozona\Macrozonas_Chile.shp"
DATA_DIR = r"C:\Users\DESKTOP\Desktop\Tesis\Datos\CR2MET2\CR2MET_pr_v2.5\pr"
FILENAME_RE = re.compile(r"CR2MET_pr_v2[._]5_day_(\d{4})_(\d{2})_005deg\.nc$")

WORLD_BOUNDARIES_URL = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"

MAP_OUT = "winter_precipitation_share_chile.png"
WINTER_MONTHS = [6, 7, 8]

# Recorte/aspecto: mismos margenes usados en los otros mapas (elevacion,
# precipitacion y temperatura), para que todas las figuras compartan el
# mismo encuadre.
PAD_LON_WEST = 0.45
PAD_LON_EAST = 1.3
PAD_LAT_TOP = 1.1
PAD_LAT_BOT = 1.0

# Encuadre comun de todos los mapas [oeste, este, sur, norte] en grados.
# (Reemplaza el calculo con PAD_*, que cortaba Tierra del Fuego y las islas.)
MAP_EXTENT = [-80, -65, -57, -17]

# Tamanos de letra (mas grandes)
CBAR_TITLE_FONTSIZE = 30
CBAR_TICK_FONTSIZE = 26
AXIS_TICK_FONTSIZE = 26

# Fuente Times New Roman para todo el grafico (incluidos los subindices)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

# --------------------------------------------------------------------------
# 2. Discover monthly files and group by year
# --------------------------------------------------------------------------
files = glob.glob(os.path.join(DATA_DIR, "CR2MET_pr_v2*5_day_*_005deg.nc"))

files_by_year = {}
for f in files:
    m = FILENAME_RE.search(os.path.basename(f))
    if not m:
        continue
    year, month = int(m.group(1)), int(m.group(2))
    files_by_year.setdefault(year, {})[month] = f

# --------------------------------------------------------------------------
# 3. Load monthly totals (mm/month) for every available month, no zero-fill
# --------------------------------------------------------------------------
# NOTA: la mascara "cl_mask" se combina (OR) entre TODOS los archivos leidos,
# en lugar de tomarse solo del primero. Si un unico archivo viniera con la
# mascara incompleta/mal definida en alguna zona (p.ej. el sur), usar solo
# ese primer archivo dejaria esa zona enmascarada (NaN) para siempre. Con la
# union esto ya no puede pasar, siempre que AL MENOS un archivo tenga la
# mascara correcta ahi.
monthly_totals_raw = {}   # (year, month) -> DataArray SIN enmascarar todavia
cl_mask_accum = None

for year in sorted(files_by_year):
    for month in sorted(files_by_year[year]):
        path = files_by_year[year][month]
        with xr.open_dataset(path) as ds:
            this_mask = (ds["cl_mask"] == 1)
            cl_mask_accum = this_mask if cl_mask_accum is None else (cl_mask_accum | this_mask)
            total = ds["pr"].sum(dim="time", skipna=True).load()  # forzar lectura antes de cerrar el archivo
        monthly_totals_raw[(year, month)] = total

monthly_totals_map = {
    key: da.where(cl_mask_accum) for key, da in monthly_totals_raw.items()
}

if not monthly_totals_map:
    raise RuntimeError(
        "No se encontró ningún archivo mensual en DATA_DIR. "
        "Revisa la ruta y el patrón de nombres de los archivos."
    )

# --------------------------------------------------------------------------
# 4. Monthly climatology: each calendar month averaged over the years that
#    have data for it (no zero-fill for missing years/months).
# --------------------------------------------------------------------------
monthly_climatology = {}  # month (1-12) -> DataArray mean (lat, lon)

for month in range(1, 13):
    maps_this_month = [
        monthly_totals_map[(year, month)]
        for year in files_by_year
        if month in files_by_year[year]
    ]
    if maps_this_month:
        monthly_climatology[month] = sum(maps_this_month) / len(maps_this_month)

missing_months = [m for m in range(1, 13) if m not in monthly_climatology]
if missing_months:
    raise RuntimeError(
        f"No hay datos disponibles para el/los mes(es) {missing_months} en "
        "ningún año -> no se puede calcular la climatología anual completa."
    )

# --------------------------------------------------------------------------
# 5. Winter share (%) = 100 * winter climatology / annual climatology
# --------------------------------------------------------------------------
winter_climatology = sum(monthly_climatology[m] for m in WINTER_MONTHS)
annual_climatology = sum(monthly_climatology.values())

winter_share = 100 * winter_climatology / annual_climatology
winter_share = winter_share.rio.write_crs("EPSG:4326")
winter_share = winter_share.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

# --------------------------------------------------------------------------
# 6. Load shapefile, clip, and load neighboring countries
# --------------------------------------------------------------------------
chile = gpd.read_file(SHAPEFILE_PATH)
if chile.crs is None:
    chile = chile.set_crs("EPSG:3857")
chile_reproj = chile.to_crs(winter_share.rio.crs)

winter_share_clip = winter_share.rio.clip(
    chile_reproj.geometry.values, chile_reproj.crs, drop=True
)

lon_min = float(winter_share_clip.lon.min())
lon_max = float(winter_share_clip.lon.max())
lat_min = float(winter_share_clip.lat.min())
lat_max = float(winter_share_clip.lat.max())

# Mismas coordenadas/encuadre que los otros mapas: se calculan a partir del
# poligono continental mas grande de cada macrozona (sin islas), con el
# mismo margen (PAD_LON/PAD_LAT) y la misma correccion de aspecto real.
mainland_geoms = []
for geom in chile_reproj.geometry:
    polys = list(geom.geoms) if geom.geom_type == "MultiPolygon" else [geom]
    polys = sorted(polys, key=lambda p: p.area, reverse=True)
    mainland_geoms.append(polys[0])
mainland_bounds = np.array([g.bounds for g in mainland_geoms])
lon_min = mainland_bounds[:, 0].min()
lon_max = mainland_bounds[:, 2].max()
lat_min = mainland_bounds[:, 1].min()
lat_max = mainland_bounds[:, 3].max()

xlim = (MAP_EXTENT[0], MAP_EXTENT[1])
ylim = (MAP_EXTENT[2], MAP_EXTENT[3])
mean_lat = (ylim[0] + ylim[1]) / 2
map_aspect = 1 / np.cos(np.radians(mean_lat))

resp = requests.get(WORLD_BOUNDARIES_URL, timeout=30)
world = gpd.read_file(io.BytesIO(resp.content))
other_countries = world[world["name"] != "Chile"].to_crs(winter_share.rio.crs)

# --------------------------------------------------------------------------
# Guardar el campo para la figura compuesta (figura_1_compuesta.py)
# --------------------------------------------------------------------------
CACHE_DIR = r"C:\Users\DESKTOP\Desktop\Tesis\primera figura\datos_figura"
os.makedirs(CACHE_DIR, exist_ok=True)
winter_share_clip.drop_vars("spatial_ref", errors="ignore").rename("PJJA_Ptot").to_netcdf(
    os.path.join(CACHE_DIR, "c_PJJA_Ptot.nc"))
print("Campo guardado en:", os.path.join(CACHE_DIR, "c_PJJA_Ptot.nc"))

# --------------------------------------------------------------------------
# 7. Plot
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 12))

# Países vecinos en gris, con límites en negrita, al fondo del gráfico
other_countries.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=1.1, zorder=0)

# Escala de colores discreta (porcentaje 0-100%), paleta RdBu
# (rojo = poco aporte de invierno, azul = mucho)
LEVELS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

cmap = plt.get_cmap("RdBu", len(LEVELS) - 1)
norm = mcolors.BoundaryNorm(LEVELS, ncolors=cmap.N, clip=False)

img = winter_share_clip.plot(
    ax=ax, cmap=cmap, norm=norm, add_colorbar=False, zorder=1,
)
ax.set_title("")
chile_reproj.boundary.plot(ax=ax, color="black", linewidth=1.1, zorder=2)

# Barra de colores hacia el lado del océano (oeste = izquierda del mapa).
# extend="neither": sin flechas en ningún extremo (el % va de 0 a 100).
divider = make_axes_locatable(ax)
cax = divider.append_axes("left", size="5%", pad=0.8)
cbar = fig.colorbar(img, cax=cax, extend="neither", ticks=LEVELS)
cbar.ax.set_yticklabels([f"{t:.0f}" for t in LEVELS], fontsize=CBAR_TICK_FONTSIZE)
cbar.ax.yaxis.set_ticks_position("left")

# Titulo con la unidad, ubicado ARRIBA de la barra de colores
cbar.ax.set_title(r"c) P$_{\mathrm{JJA}}$/P$_{\mathrm{tot}}$ [%]",
                  fontsize=CBAR_TITLE_FONTSIZE, loc="left", pad=20)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect(map_aspect)
ax.set_xlabel("")
ax.set_ylabel("")
ax.tick_params(axis="both", labelsize=AXIS_TICK_FONTSIZE)
for spine in ax.spines.values():
    spine.set_visible(True)

plt.tight_layout()
plt.savefig(MAP_OUT, dpi=200)
plt.show()

years_sorted = sorted(files_by_year)
print(f"Listo: {MAP_OUT} generado (climatología mensual construida con datos de "
      f"{years_sorted[0]}-{years_sorted[-1]}, {len(monthly_totals_map)} meses en total).")