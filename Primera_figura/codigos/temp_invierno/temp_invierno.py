# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 17:07:51 2026

@author: DESKTOP
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:39:30 2026

@author: DESKTOP
"""

"""
Winter (JJA) mean temperature map for Chile: climatological winter
(June-July-August) average daily mean temperature ((tmin+tmax)/2),
with a diverging red/blue color scale centered at 0 degC
(above zero -> red, below zero -> blue).

Expected input data:
  - Provincias.shp        : Chile boundary shapefile
  - Monthly CR2MET files  : one .nc file per month, all in DATA_DIR,
                             following the CR2MET naming pattern:
                             CR2MET_tmin_tmax_v2.5_day_{year}_{month:02d}_005deg.nc
                             Each file has daily "tmin" and "tmax" variables
                             in Celsius, and a "cl_mask" variable marking
                             continental Chile.

What the script does:
  1. Scans DATA_DIR for all monthly files matching the naming pattern
     (does NOT require complete years -- every available month is used,
     nothing is filled with zero/NaN-filled data).
  2. For each month-file, computes the daily mean temperature as
     (tmin + tmax) / 2, then averages over days to get that month's mean
     temperature per grid cell.
  3. For each of the 12 calendar months, averages that month's mean
     temperature across all years that have data for that month
     (monthly climatology, simple mean -- no zero-fill for missing years).
  4. Winter climatology = simple average of the June, July and August
     monthly climatological means.
  5. Plots the winter mean temperature map using the same visual style as
     the other maps: neighboring countries in gray with bold borders,
     colorbar on the ocean side (west), discrete diverging color levels
     centered at 0 degC (red above zero, blue below zero) with
     extend="both".

NOTE on the shapefile: Macrozonas_Chile.shp is missing .shx, .dbf and .prj.
The .shx is reconstructed automatically (SHAPE_RESTORE_SHX) and EPSG:3857
is assumed for its CRS given the coordinate range.

NOTE on the map frame: the map extent (xlim/ylim) and aspect ratio are
computed the same way as in the elevation and precipitation scripts (same
PAD_LON/PAD_LAT margins around the mainland macrozone polygons), so all
three figures share the exact same framing.
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
DATA_DIR = r"C:\Users\DESKTOP\Desktop\Tesis\Datos\CR2MET2\CR2MET_txn_v2.5\txn"
FILENAME_RE = re.compile(r"CR2MET_tmin_tmax_v2[._]5_day_(\d{4})_(\d{2})_005deg\.nc$")

WORLD_BOUNDARIES_URL = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"

MAP_OUT = "winter_temperature_chile.png"
WINTER_MONTHS = [6, 7, 8]

# Recorte/aspecto: mismos margenes usados en los otros mapas (elevacion y
# precipitacion), para que las tres figuras queden con el mismo encuadre.
PAD_LON_WEST = 0.45
PAD_LON_EAST = 1.3
PAD_LAT_TOP = 1.1
PAD_LAT_BOT = 1.0

# Tamanos de letra (mismos que la figura c)
CBAR_TITLE_FONTSIZE = 30
CBAR_TICK_FONTSIZE = 26
AXIS_TICK_FONTSIZE = 26

# Fuente Times New Roman para todo el grafico (incluidos barra y subindice)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

# --------------------------------------------------------------------------
# 2. Discover monthly files and group by year
# --------------------------------------------------------------------------
files = glob.glob(os.path.join(DATA_DIR, "CR2MET_tmin_tmax_v2*5_day_*_005deg.nc"))

files_by_year = {}
for f in files:
    m = FILENAME_RE.search(os.path.basename(f))
    if not m:
        continue
    year, month = int(m.group(1)), int(m.group(2))
    files_by_year.setdefault(year, {})[month] = f

# --------------------------------------------------------------------------
# 3. Load monthly mean temperature for every available month, no zero-fill
# --------------------------------------------------------------------------
monthly_mean_map = {}  # (year, month) -> DataArray (lat, lon), monthly mean degC
cl_mask = None

for year in sorted(files_by_year):
    for month in sorted(files_by_year[year]):
        path = files_by_year[year][month]
        with xr.open_dataset(path) as ds:
            if cl_mask is None:
                cl_mask = ds["cl_mask"]
            tmean_daily = (ds["tmin"] + ds["tmax"]) / 2
            monthly_mean = tmean_daily.mean(dim="time", skipna=True).where(cl_mask == 1)
        monthly_mean_map[(year, month)] = monthly_mean

if not monthly_mean_map:
    raise RuntimeError(
        "No se encontró ningún archivo mensual en DATA_DIR. "
        "Revisa la ruta y el patrón de nombres de los archivos."
    )

# --------------------------------------------------------------------------
# 4. Monthly climatology: each calendar month averaged over the years that
#    have data for it (no zero-fill for missing years/months).
# --------------------------------------------------------------------------
monthly_climatology = {}  # month (1-12) -> DataArray mean (lat, lon)

for month in WINTER_MONTHS:
    maps_this_month = [
        monthly_mean_map[(year, month)]
        for year in files_by_year
        if month in files_by_year[year]
    ]
    if maps_this_month:
        monthly_climatology[month] = sum(maps_this_month) / len(maps_this_month)

missing_months = [m for m in WINTER_MONTHS if m not in monthly_climatology]
if missing_months:
    raise RuntimeError(
        f"No hay datos disponibles para el/los mes(es) {missing_months} en "
        "ningún año -> no se puede calcular la climatología de invierno."
    )

# --------------------------------------------------------------------------
# 5. Winter mean temperature = simple average of JJA monthly climatologies
# --------------------------------------------------------------------------
winter_temp = sum(monthly_climatology.values()) / len(monthly_climatology)
winter_temp = winter_temp.rio.write_crs("EPSG:4326")
winter_temp = winter_temp.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

# --------------------------------------------------------------------------
# 6. Load shapefile, clip, and load neighboring countries
# --------------------------------------------------------------------------
chile = gpd.read_file(SHAPEFILE_PATH)
if chile.crs is None:
    chile = chile.set_crs("EPSG:3857")
chile_reproj = chile.to_crs(winter_temp.rio.crs)

winter_temp_clip = winter_temp.rio.clip(
    chile_reproj.geometry.values, chile_reproj.crs, drop=True
)

lon_min = float(winter_temp_clip.lon.min())
lon_max = float(winter_temp_clip.lon.max())
lat_min = float(winter_temp_clip.lat.min())
lat_max = float(winter_temp_clip.lat.max())

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

xlim = (lon_min - PAD_LON_WEST, lon_max + PAD_LON_EAST)
ylim = (lat_min - PAD_LAT_BOT, lat_max + PAD_LAT_TOP)
mean_lat = (ylim[0] + ylim[1]) / 2
map_aspect = 1 / np.cos(np.radians(mean_lat))

resp = requests.get(WORLD_BOUNDARIES_URL, timeout=30)
world = gpd.read_file(io.BytesIO(resp.content))
other_countries = world[world["name"] != "Chile"].to_crs(winter_temp.rio.crs)

# --------------------------------------------------------------------------
# 7. Plot
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 12))

# Países vecinos en gris, con límites en negrita, al fondo del gráfico
other_countries.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=1.1, zorder=0)

# Escala de colores divergente centrada en 0 degC (rojo sobre cero, azul bajo
# cero), con pocos saltos y simetrica para que el blanco caiga justo en 0.
LEVELS = [-20, -15, -10, -5, 0, 5, 10, 15, 20]

cmap = plt.get_cmap("RdBu_r", len(LEVELS) - 1)
norm = mcolors.BoundaryNorm(LEVELS, ncolors=cmap.N, clip=False)

img = winter_temp_clip.plot(
    ax=ax, cmap=cmap, norm=norm, add_colorbar=False, zorder=1,
)
ax.set_title("")
chile_reproj.boundary.plot(ax=ax, color="black", linewidth=1.1, zorder=2)

# Barra de colores hacia el lado del océano (oeste = izquierda del mapa)
divider = make_axes_locatable(ax)
cax = divider.append_axes("left", size="5%", pad=0.8)
cbar = fig.colorbar(img, cax=cax, extend="both", ticks=LEVELS)
cbar.ax.set_yticklabels([f"{t:.0f}" for t in LEVELS], fontsize=CBAR_TICK_FONTSIZE)
cbar.ax.yaxis.set_ticks_position("left")

# Titulo con la unidad, ubicado ARRIBA de la barra de colores
cbar.ax.set_title(r"d) $\overline{\mathrm{T}}_{\mathrm{JJA}}$ [°C]",
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
print(f"Listo: {MAP_OUT} generado (climatología de invierno construida con datos de "
      f"{years_sorted[0]}-{years_sorted[-1]}, {len(monthly_mean_map)} meses en total).")