# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:55:33 2026

@author: DESKTOP
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 12:20:12 2026

@author: DESKTOP
"""

"""
Annual precipitation map for Chile (CR2MET) + macrozones, and CSV export
of domain-averaged annual precipitation per year.

Expected input data:
  - Provincias.shp            : Chile boundary shapefile
  - Monthly CR2MET files      : one .nc file per month, all in DATA_DIR,
                                 following the naming pattern used by CR2MET:
                                 CR2MET_pr_v2_5_day_{year}_{month:02d}_005deg.nc
                                 e.g. CR2MET_pr_v2_5_day_1960_01_005deg.nc
                                 Each file has a daily "pr" variable in mm/day
                                 and a "cl_mask" variable marking continental Chile.

What the script does:
  1. Scans DATA_DIR for all monthly files matching the naming pattern.
  2. For every year with a complete set of 12 months, sums daily precipitation
     within each month, then sums the 12 monthly totals to get the annual
     total precipitation (mm/year) at every grid cell.
  3. Averages the annual totals across all available years to get the
     climatological mean annual precipitation (this is what gets mapped).
  4. Computes the domain-averaged (spatial mean over continental Chile)
     annual total precipitation for EACH year and exports it to a CSV.
  5. Plots the climatological annual precipitation map with the same style
     used for the elevation map: Chile boundary, DGA macrozones (dashed
     lines + black labels), full compass rose, English axis labels.

Macrozones: DGA definition from Gateño et al. (2024), Climatic Change 177:87,
section 2.1 -- Far North (17-26S), Near North (26-32S), Central (32-37S),
South (37-44S), Austral (44-56S).

NOTE on the shapefile: Macrozonas_Chile.shp is missing .shx, .dbf and .prj.
The .shx is reconstructed automatically (SHAPE_RESTORE_SHX) and EPSG:3857
is assumed for its CRS given the coordinate range (see previous elevation
script for details).

NOTE on the map frame: the map extent (xlim/ylim) and aspect ratio are
computed the same way as in the elevation script (same PAD_LON/PAD_LAT
margins around the mainland macrozone polygons), so both figures share
the exact same framing.
"""

import os
import re
import glob

os.environ["SHAPE_RESTORE_SHX"] = "YES"

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import requests
import io
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Polygon, Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


# --------------------------------------------------------------------------
# 0. Compass rose helper (same as in the elevation script)
# --------------------------------------------------------------------------
def draw_compass_rose(parent_ax, width="18%", height="18%", loc="upper left"):
    axc = inset_axes(parent_ax, width=width, height=height, loc=loc, borderpad=1)
    axc.set_xlim(-1.3, 1.3)
    axc.set_ylim(-1.3, 1.3)
    axc.set_aspect("equal")
    axc.axis("off")

    def kite(angle_deg, length, half_width, color_left="black", color_right="white"):
        angle = np.radians(angle_deg)
        tip = np.array([length * np.cos(angle), length * np.sin(angle)])
        perp = np.array([-np.sin(angle), np.cos(angle)]) * half_width
        center = np.array([0, 0])
        axc.add_patch(Polygon([center, tip, perp], closed=True,
                               facecolor=color_left, edgecolor="black", linewidth=0.6, zorder=5))
        axc.add_patch(Polygon([center, tip, -perp], closed=True,
                               facecolor=color_right, edgecolor="black", linewidth=0.6, zorder=5))

    for ang in [90, 0, 270, 180]:
        kite(ang, length=1.0, half_width=0.16, color_left="black", color_right="white")
    for ang in [45, 315, 225, 135]:
        kite(ang, length=0.55, half_width=0.09, color_left="black", color_right="black")

    axc.add_patch(Circle((0, 0), 1.05, fill=False, edgecolor="black", linewidth=3.5, zorder=4))
    axc.add_patch(Circle((0, 0), 0.9, fill=False, edgecolor="black", linewidth=0.8, zorder=4))

    label_r = 1.28
    for ang, lab in zip([90, 0, 270, 180], ["N", "E", "S", "W"]):
        rad = np.radians(ang)
        axc.text(label_r * np.cos(rad), label_r * np.sin(rad), lab,
                  ha="center", va="center", fontsize=11, fontweight="bold", zorder=6)


# --------------------------------------------------------------------------
# 1. Inputs
# --------------------------------------------------------------------------
SHAPEFILE_PATH = r"C:\Users\DESKTOP\Desktop\Tesis\macrozona\Macrozonas_Chile.shp"
DATA_DIR = r"C:\Users\DESKTOP\Desktop\Tesis\Datos\CR2MET2\CR2MET_pr_v2.5\pr"
FILENAME_RE = re.compile(r"CR2MET_pr_v2[._]5_day_(\d{4})_(\d{2})_005deg\.nc$")

# Límites de otros países (para dibujarlos en gris de fondo)
WORLD_BOUNDARIES_URL = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"

CSV_OUT = "annual_precipitation_chile.csv"
MAP_OUT = "precipitation_chile_macrozonas.png"

# Recorte/aspecto: mismos margenes usados en el mapa de elevacion, para que
# ambas figuras queden con exactamente el mismo encuadre.
PAD_LON_WEST = 0.45
PAD_LON_EAST = 1.3
PAD_LAT_TOP = 1.1
PAD_LAT_BOT = 1.0

# Encuadre comun de todos los mapas [oeste, este, sur, norte] en grados.
# (Reemplaza el calculo con PAD_*, que cortaba Tierra del Fuego y las islas.)
MAP_EXTENT = [-80, -65, -57, -17]

# Tamanos de letra (grandes)
CBAR_TITLE_FONTSIZE = 26
CBAR_TICK_FONTSIZE = 22
AXIS_TICK_FONTSIZE = 22

# Fuente Times New Roman para todo el grafico (incluido el superindice -1)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

MACROZONAS = [
    ("Far North", -17, -26, "#d95f02"),
    ("Near North", -26, -32, "#e6ab02"),
    ("Central",   -32, -37, "#1b9e77"),
    ("South",     -37, -44, "#377eb8"),
    ("Austral",   -44, -56, "#66c2a5"),
]

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
# 3. Cargar los totales mensuales de precipitación (mm/mes) de TODOS los
#    meses disponibles, sin exigir años completos y sin rellenar meses
#    faltantes con cero.
# --------------------------------------------------------------------------
monthly_totals_map = {}   # (year, month) -> DataArray (lat, lon), total mensual mm
monthly_totals_mean = {}  # (year, month) -> float, promedio espacial sobre Chile

cl_mask = None  # máscara de Chile continental, se toma del primer archivo leído

for year in sorted(files_by_year):
    for month in sorted(files_by_year[year]):
        path = files_by_year[year][month]
        with xr.open_dataset(path) as ds:
            if cl_mask is None:
                cl_mask = ds["cl_mask"]
            total = ds["pr"].sum(dim="time", skipna=True).where(cl_mask == 1)
        monthly_totals_map[(year, month)] = total
        monthly_totals_mean[(year, month)] = float(total.mean(skipna=True))

if not monthly_totals_map:
    raise RuntimeError(
        "No se encontró ningún archivo mensual en DATA_DIR. "
        "Revisa la ruta y el patrón de nombres de los archivos."
    )

# --------------------------------------------------------------------------
# 4. Total anual por año, usando SOLO los meses disponibles para ese año
#    (sin rellenar con cero los meses faltantes). Se incluyen todos los
#    años, aunque estén incompletos.
# --------------------------------------------------------------------------
annual_mean_chile = {}   # year -> float, suma de los meses disponibles ese año
annual_n_months = {}     # year -> cantidad de meses disponibles ese año

for year in sorted(files_by_year):
    months_available = sorted(files_by_year[year])
    annual_mean_chile[year] = sum(
        monthly_totals_mean[(year, m)] for m in months_available
    )
    annual_n_months[year] = len(months_available)
    if len(months_available) < 12:
        print(f"Año {year}: {len(months_available)}/12 meses disponibles -> "
              f"se usa la suma de esos meses (sin rellenar con cero)")

# --------------------------------------------------------------------------
# 5. Export CSV: precipitación anual por año (con los meses disponibles)
# --------------------------------------------------------------------------
df = pd.DataFrame({
    "year": list(annual_mean_chile.keys()),
    "precipitation_mm": list(annual_mean_chile.values()),
    "n_months_available": [annual_n_months[y] for y in annual_mean_chile],
}).sort_values("year")
df.to_csv(CSV_OUT, index=False)
print(f"CSV exportado: {CSV_OUT} ({len(df)} años)")

# --------------------------------------------------------------------------
# 6. Climatología mensual: para cada uno de los 12 meses del año, se
#    promedia su total mensual SOLO entre los años que tienen ese mes
#    disponible (sin rellenar con cero). La climatología anual acumulada
#    es la suma de los 12 promedios mensuales.
# --------------------------------------------------------------------------
monthly_climatology = {}  # month (1-12) -> DataArray promedio (lat, lon)

for month in range(1, 13):
    maps_this_month = [
        monthly_totals_map[(year, month)]
        for year in files_by_year
        if month in files_by_year[year]
    ]
    if maps_this_month:
        monthly_climatology[month] = sum(maps_this_month) / len(maps_this_month)

climatology = sum(monthly_climatology.values())
climatology = climatology.rio.write_crs("EPSG:4326")
climatology = climatology.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

# --------------------------------------------------------------------------
# 7. Load shapefile and clip
# --------------------------------------------------------------------------
chile = gpd.read_file(SHAPEFILE_PATH)
if chile.crs is None:
    chile = chile.set_crs("EPSG:3857")
chile_reproj = chile.to_crs(climatology.rio.crs)

climatology_clip = climatology.rio.clip(
    chile_reproj.geometry.values, chile_reproj.crs, drop=True
)

# Mismas coordenadas/encuadre que el mapa de elevacion: se calculan a partir
# del poligono continental mas grande de cada macrozona (sin islas), con el
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

# Países vecinos (Argentina, Perú, Bolivia, etc.) para dibujar en gris de fondo
resp = requests.get(WORLD_BOUNDARIES_URL, timeout=30)
world = gpd.read_file(io.BytesIO(resp.content))
other_countries = world[world["name"] != "Chile"].to_crs(climatology.rio.crs)

# --------------------------------------------------------------------------
# Guardar el campo para la figura compuesta (figura_1_compuesta.py)
# --------------------------------------------------------------------------
CACHE_DIR = r"C:\Users\DESKTOP\Desktop\Tesis\primera figura\datos_figura"
os.makedirs(CACHE_DIR, exist_ok=True)
climatology_clip.drop_vars("spatial_ref", errors="ignore").rename("Ptot").to_netcdf(
    os.path.join(CACHE_DIR, "b_Ptot.nc"))
print("Campo guardado en:", os.path.join(CACHE_DIR, "b_Ptot.nc"))

# --------------------------------------------------------------------------
# 8. Plot
# --------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 12))

# Países vecinos en gris, con límites en negrita, al fondo del gráfico
other_countries.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=1.1, zorder=0)

# Escala de colores discreta con MAS clases en montos bajos (0-200 mm),
# para distinguir bien el norte arido, y clases mas anchas en montos altos.
LEVELS = [0, 5, 20, 50, 75, 100, 150, 200, 300, 500,
          750, 1000, 1500, 2000, 3000, 4000]

# Paleta NCL "prcp_1" (17 colores). Se escribe directamente en RGB para no
# depender de ningun paquete extra (es identica a colormaps.prcp_1).
#   - Color 1 (blanco): se reserva SOLO para P = 0.
#   - Colores 2 a 16: una por cada una de las 15 clases (0-5 ... 3000-4000).
#   - El color 17 original de prcp_1 es casi blanco (lino); no sirve para
#     el tramo > 4000 mm porque se confundiria con el 0, asi que la flecha
#     "max" usa un morado mas oscuro que el de la ultima clase.
PRCP_1_RGB = [
    (255, 255, 255),  # 0 mm (blanco)
    (170, 255, 255),  # 0-5
    ( 85, 160, 255),  # 5-20
    ( 29,   0, 255),  # 20-50
    (126, 229,  91),  # 50-75
    ( 78, 204,  67),  # 75-100
    ( 46, 178,  57),  # 100-150
    ( 30, 153,  61),  # 150-200
    (255, 255, 102),  # 200-300
    (255, 204, 102),  # 300-500
    (255, 136,  76),  # 500-750
    (255,  25,  25),  # 750-1000
    (204,  61,  61),  # 1000-1500
    (165,  49,  49),  # 1500-2000
    (237,   0, 237),  # 2000-3000
    (137, 103, 205),  # 3000-4000
]
COLOR_SOBRE_MAX = (85, 55, 140)  # > 4000 mm (morado oscuro)

n_clases = len(LEVELS) - 1
colores_clases = [np.array(c) / 255 for c in PRCP_1_RGB[1:]]
assert len(colores_clases) == n_clases, "Debe haber un color por clase"

cmap = mcolors.ListedColormap(colores_clases, name="prcp_1_discreta")
cmap.set_over(np.array(COLOR_SOBRE_MAX) / 255)   # > 4000 mm
cmap.set_under("white")                           # exactamente 0 mm
cmap.set_bad(alpha=0)                             # fuera de Chile: transparente

# El primer limite se corre a un valor infinitesimal: asi P = 0 cae bajo
# el primer limite ("under" = blanco) y cualquier P > 0 ya tiene color.
LIMITES = [1e-6] + LEVELS[1:]
norm = mcolors.BoundaryNorm(LIMITES, ncolors=n_clases)

img = climatology_clip.plot(
    ax=ax, cmap=cmap, norm=norm, add_colorbar=False, zorder=1,
)
ax.set_title("")
chile_reproj.boundary.plot(ax=ax, color="black", linewidth=1.1, zorder=2)

# Barra de colores hacia el lado del océano (oeste = izquierda del mapa)
# extend="max": solo flecha arriba (> 4000 mm); sin flecha en el 0.
# (Los pixeles con P = 0 igual se pintan blancos en el mapa.)
divider = make_axes_locatable(ax)
cax = divider.append_axes("left", size="5%", pad=0.8)
cbar = fig.colorbar(img, cax=cax, extend="max", ticks=LIMITES,
                    spacing="uniform")  # todas las clases del mismo alto
cbar.ax.set_yticklabels([f"{t:.0f}" for t in LEVELS], fontsize=CBAR_TICK_FONTSIZE)
cbar.ax.yaxis.set_ticks_position("left")

# Titulo con la unidad, ubicado ARRIBA de la barra de colores
cbar.ax.set_title(r"b) P_tot [mm yr$^{-1}$]", fontsize=CBAR_TITLE_FONTSIZE,
                  loc="left", pad=20)

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