# -*- coding: utf-8 -*-
"""
Figura 1 compuesta: 4 mapas lado a lado (a-d) + panel inferior (e-f).

    a) Elevation [m]          b) P_tot [mm yr^-1]
    c) P_JJA/P_tot [%]        d) T_JJA (con barra) [°C]
    e) SnowFall fraction      f) 0 °C isotherm height

Todo se dibuja aqui mismo (no se pegan imagenes), asi los paneles quedan
del mismo tamano, con el mismo encuadre y la misma tipografia.
Solo el panel a) lleva longitud y latitud.

ANTES DE CORRER ESTE SCRIPT, corre una vez (en cualquier orden):
  - prep_primera_figura.py                    -> b_Ptot.nc
  - prep_acum_invierno_divido_prep_anual.py   -> c_PJJA_Ptot.nc
  - temp_invierno.py                          -> d_TJJA.nc
  - H0_y_Snowfaction.py                       -> ef_SnowFraction_H0.pkl
Todos guardan sus resultados en CACHE_DIR. La elevacion se lee directo del
NetCDF de ETOPO (ELEV_NC).
"""

import io
import os

import numpy as np
import xarray as xr
import geopandas as gpd
import requests
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import matplotlib.patheffects as pe
from matplotlib.ticker import MultipleLocator, FuncFormatter
from matplotlib.patches import RegularPolygon
from shapely import contains_xy

# ==========================================================================
# 1. RUTAS
# ==========================================================================
TESIS = r"C:\Users\DESKTOP\Desktop\Tesis"
CACHE_DIR = os.path.join(TESIS, "primera figura", "datos_figura")

SHP_PATH = os.path.join(TESIS, "macrozona", "Macrozonas_Chile.shp")
ELEV_NC = os.path.join(TESIS, "ETOPO_land_Chile_Z_dZ_005deg_RESAMPLE_REGRID.nc")
PTOT_NC = os.path.join(CACHE_DIR, "b_Ptot.nc")
SHARE_NC = os.path.join(CACHE_DIR, "c_PJJA_Ptot.nc")
TJJA_NC = os.path.join(CACHE_DIR, "d_TJJA.nc")
EF_PKL = os.path.join(CACHE_DIR, "ef_SnowFraction_H0.pkl")

OUT_PNG = os.path.join(TESIS, "primera figura", "Figura1_compuesta.png")
OUT_PDF = OUT_PNG.replace(".png", ".pdf")

WORLD_URL = "https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json"

# ==========================================================================
# 2. ENCUADRE Y TAMANOS
# ==========================================================================
# [oeste, este, sur, norte] -- 2 grados menos de mar al oeste que antes (-80)
MAP_EXTENT = [-78, -65, -57, -17]
ASPECT = 1 / np.cos(np.radians((MAP_EXTENT[2] + MAP_EXTENT[3]) / 2))

# Geometria de cada columna (en pulgadas), de izquierda a derecha:
#   [numeros barra | barra | espacio | MAPA]
CB_LABEL_W = 0.80   # espacio para los numeros de la barra de colores
CB_W = 0.20         # ancho de la barra de colores
CB_GAP = 0.18       # espacio entre barra y mapa
MAP_W = 2.55        # ancho de cada mapa (el alto sale de la proporcion real)
COL_GAP = 0.10      # espacio extra entre columnas
LATLAB_W = 0.75     # espacio para las latitudes del panel a) (a su derecha)

MARGIN_L, MARGIN_R = 0.15, 0.15
MARGIN_TOP = 0.75       # espacio para los titulos a)-d)
LONLAB_H = 0.85         # espacio para las longitudes bajo los mapas
ROW_GAP = 0.15          # espacio entre los mapas y el panel e-f
MARGIN_BOT = 0.10

# Panel inferior e) - f)
EF_AX_H = 4.3           # alto de los graficos e) y f)
EF_TITLE_H = 0.55       # espacio para los titulos e) y f)
EF_XLAB_H = 0.95        # espacio para numeros y nombre del eje x
EF_YLAB_W = 1.05        # espacio a la izquierda de cada grafico (eje y)
EF_TWIN_W = 1.05        # espacio a la derecha de f) (segundo eje y)

# Tipografia
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"

TITLE_FS = 21        # titulos a) ... f)
# Titulos a)-d): alineados a la izquierda, arriba de su barra de colores.
# TITLE_OFFSET corre el titulo a la derecha (+) o izquierda (-), en pulgadas,
# respecto del borde izquierdo de los numeros de la barra.
TITLE_OFFSET = 0.15
CB_TICK_FS = 15      # numeros de las barras de colores
COORD_FS = 14        # lon/lat del panel a)
ZONE_FS = 10         # nombres de macrozonas en el panel a)
EF_TICK_FS = 15      # numeros de los ejes de e) y f)
EF_LABEL_FS = 15     # nombres de los ejes de e) y f)
EF_LEGEND_FS = 11    # leyendas de e) y f)
SF_YMAX = 0.7        # limite superior de e)
SHOW_ZONE_NAMES = True
SHOW_NORTH_ARROW = True

CHILE_LW = 0.7       # grosor del limite de Chile / macrozonas
NEIGH_LW = 0.8       # grosor del borde de los paises vecinos
DPI = 300

# ==========================================================================
# 3. DATOS COMUNES: macrozonas y paises vecinos
# ==========================================================================
macro = gpd.read_file(SHP_PATH)
if macro.crs is None:
    macro = macro.set_crs("EPSG:3857")
macro = macro.to_crs("EPSG:4326")

world = gpd.read_file(io.BytesIO(requests.get(WORLD_URL, timeout=30).content))
neighbors = world[world["name"] != "Chile"].to_crs("EPSG:4326")


def leer_campo(path):
    """Lee un campo (lat, lon) guardado por los scripts individuales."""
    if not os.path.exists(path):
        print(f"[!] No se encontró {path} -> el panel quedará vacío")
        return None
    da = xr.open_dataarray(path).squeeze()
    return da.sortby("lat")


def leer_elevacion(path):
    """Elevacion ETOPO enmascarada al poligono de Chile (como mapa_elevacion.py)."""
    if not os.path.exists(path):
        print(f"[!] No se encontró {path} -> el panel a) quedará vacío")
        return None
    ds = xr.open_dataset(path)
    z = ds["z"].astype(float)
    # Cada macrozona simplificada y con un margen de 0.1° (incluye la costa).
    # Se hace zona por zona: unir todo y luego hacer buffer usa demasiada memoria.
    lon2d, lat2d = np.meshgrid(ds["lon"].values, ds["lat"].values)
    dentro = np.zeros(lon2d.shape, dtype=bool)
    for g in macro.geometry:
        dentro |= contains_xy(g.simplify(0.01).buffer(0.1, quad_segs=2), lon2d, lat2d)
    z = z.where(dentro & (z > 0))
    return z.sortby("lat")


# ==========================================================================
# 4. ESCALAS DE COLORES (mismas de cada figura individual)
# ==========================================================================
# a) Elevacion
ELEV_COLORS = ["#2e7d32", "#c9d94a", "#d94a3d", "#c8967a", "#8a5a3b", "#4a2f22"]
elev_cmap = mcolors.LinearSegmentedColormap.from_list("elev", ELEV_COLORS, N=256)
elev_cmap.set_bad("none")
elev_norm = mcolors.Normalize(0, 5500)

# b) Precipitacion anual (paleta NCL prcp_1)
P_LEVELS = [0, 5, 20, 50, 75, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000, 4000]
P_RGB = [(170, 255, 255), (85, 160, 255), (29, 0, 255), (126, 229, 91), (78, 204, 67),
         (46, 178, 57), (30, 153, 61), (255, 255, 102), (255, 204, 102), (255, 136, 76),
         (255, 25, 25), (204, 61, 61), (165, 49, 49), (237, 0, 237), (137, 103, 205)]
p_cmap = mcolors.ListedColormap([np.array(c) / 255 for c in P_RGB])
p_cmap.set_over(np.array((85, 55, 140)) / 255)
p_cmap.set_under("white")
p_cmap.set_bad(alpha=0)
P_LIMITS = [1e-6] + P_LEVELS[1:]
p_norm = mcolors.BoundaryNorm(P_LIMITS, ncolors=p_cmap.N)

# c) Fraccion de invierno
S_LEVELS = list(range(0, 101, 10))
s_cmap = plt.get_cmap("RdBu", len(S_LEVELS) - 1)
s_norm = mcolors.BoundaryNorm(S_LEVELS, ncolors=s_cmap.N)

# d) Temperatura de invierno
T_LEVELS = [-20, -15, -10, -5, 0, 5, 10, 15, 20]
t_cmap = plt.get_cmap("RdBu_r", len(T_LEVELS) - 1)
t_norm = mcolors.BoundaryNorm(T_LEVELS, ncolors=t_cmap.N, clip=False)

PANELES = [
    dict(data=leer_elevacion(ELEV_NC), cmap=elev_cmap, norm=elev_norm,
         ticks=list(range(0, 5001, 1000)), ticklabels=None, extend="neither",
         title=r"a) Elevation [m]", coords=True),
    dict(data=leer_campo(PTOT_NC), cmap=p_cmap, norm=p_norm,
         ticks=P_LIMITS, ticklabels=[f"{t:.0f}" for t in P_LEVELS], extend="max",
         spacing="uniform", title=r"b) P$_{\mathrm{tot}}$ [mm yr$^{-1}$]", coords=False),
    dict(data=leer_campo(SHARE_NC), cmap=s_cmap, norm=s_norm,
         ticks=S_LEVELS, ticklabels=None, extend="neither",
         title=r"c) P$_{\mathrm{JJA}}$/P$_{\mathrm{tot}}$ [%]", coords=False),
    dict(data=leer_campo(TJJA_NC), cmap=t_cmap, norm=t_norm,
         ticks=T_LEVELS, ticklabels=None, extend="both",
         title=r"d) $\overline{\mathrm{T}}_{\mathrm{JJA}}$ [°C]", coords=False),
]

# ==========================================================================
# 5. GEOMETRIA DE LA FIGURA (todo en pulgadas -> fracciones de la figura)
# ==========================================================================
MAP_H = MAP_W * (MAP_EXTENT[3] - MAP_EXTENT[2]) / (MAP_EXTENT[1] - MAP_EXTENT[0]) * ASPECT
COL_W = CB_LABEL_W + CB_W + CB_GAP + MAP_W

# posicion x (pulgadas) de cada columna; tras el panel a) se deja espacio
# para sus latitudes
x_cols = []
x = MARGIN_L
for i in range(4):
    x_cols.append(x)
    x += COL_W + COL_GAP + (LATLAB_W if PANELES[i]["coords"] else 0)
FIG_W = x - COL_GAP + MARGIN_R

EF_H = EF_XLAB_H + EF_AX_H + EF_TITLE_H

FIG_H = MARGIN_TOP + MAP_H + LONLAB_H + ROW_GAP + EF_H + MARGIN_BOT
y_map = MARGIN_BOT + EF_H + ROW_GAP + LONLAB_H   # base de los mapas (pulgadas)

fig = plt.figure(figsize=(FIG_W, FIG_H))


def ax_in(x0, y0, w, h, **kw):
    """Crea un eje con coordenadas en pulgadas."""
    return fig.add_axes([x0 / FIG_W, y0 / FIG_H, w / FIG_W, h / FIG_H], **kw)


def lon_fmt(v, _):
    return f"{abs(v):.0f}°W" if v < 0 else f"{v:.0f}°E"


def lat_fmt(v, _):
    return f"{abs(v):.0f}°S" if v < 0 else f"{v:.0f}°N"


# ==========================================================================
# 6. MAPAS a) - d)
# ==========================================================================
for x0, P in zip(x_cols, PANELES):
    x_cb = x0 + CB_LABEL_W
    x_map = x_cb + CB_W + CB_GAP
    ax = ax_in(x_map, y_map, MAP_W, MAP_H)
    cax = ax_in(x_cb, y_map, CB_W, MAP_H)

    ax.set_facecolor("white")
    neighbors.plot(ax=ax, facecolor="lightgray", edgecolor="black", linewidth=NEIGH_LW, zorder=0)

    if P["data"] is not None:
        da = P["data"]
        mesh = ax.pcolormesh(da["lon"].values, da["lat"].values, da.values,
                             cmap=P["cmap"], norm=P["norm"], shading="nearest",
                             rasterized=True, zorder=1)
    else:
        mesh = plt.cm.ScalarMappable(norm=P["norm"], cmap=P["cmap"])
        ax.text(0.5, 0.5, "sin datos", transform=ax.transAxes, ha="center", color="red")

    macro.boundary.plot(ax=ax, color="black", linewidth=CHILE_LW, zorder=2)

    ax.set_xlim(MAP_EXTENT[0], MAP_EXTENT[1])
    ax.set_ylim(MAP_EXTENT[2], MAP_EXTENT[3])
    ax.set_aspect(ASPECT)
    ax.set_xlabel("")
    ax.set_ylabel("")
    for sp in ax.spines.values():
        sp.set_linewidth(0.8)

    # Graduaciones (marcas de lon/lat) en TODOS los mapas; numeros solo en a)
    ax.xaxis.set_major_locator(MultipleLocator(4))
    ax.yaxis.set_major_locator(MultipleLocator(5))
    ax.yaxis.tick_right()
    ax.tick_params(axis="both", direction="out", length=4, width=0.8)

    if P["coords"]:
        ax.xaxis.set_major_formatter(FuncFormatter(lon_fmt))
        ax.yaxis.set_major_formatter(FuncFormatter(lat_fmt))
        ax.tick_params(axis="both", labelsize=COORD_FS)
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")

        if SHOW_ZONE_NAMES and "Macrozone" in macro.columns:
            for _, row in macro.iterrows():
                g = row.geometry
                main = max(g.geoms, key=lambda p: p.area) if g.geom_type == "MultiPolygon" else g
                minx, miny, maxx, maxy = main.bounds
                ax.text(main.centroid.x, (miny + maxy) / 2, row["Macrozone"],
                        fontsize=ZONE_FS, fontweight="bold", ha="center", va="center",
                        color="0.1", zorder=6,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground="white")])

        if SHOW_NORTH_ARROW:
            cx, cy, r = MAP_EXTENT[1] - 1.1, MAP_EXTENT[3] - 1.6, 0.75
            ax.add_patch(RegularPolygon((cx, cy), numVertices=4, radius=r, orientation=0,
                                        facecolor="0.15", edgecolor="0.15", zorder=7))
            ax.add_patch(RegularPolygon((cx, cy), numVertices=4, radius=r * 0.45, orientation=0,
                                        facecolor="white", edgecolor="white", zorder=7.5))
            ax.text(cx, cy + r * ASPECT * 1.15, "N", fontsize=12, fontweight="bold",
                    ha="center", va="bottom", color="0.15", zorder=8)
    else:
        ax.tick_params(axis="both", labelbottom=False, labelleft=False,
                       labelright=False, labeltop=False)

    # Barra de colores a la izquierda, mismo alto que el mapa
    cb_kw = dict(cax=cax, extend=P["extend"], ticks=P["ticks"])
    if "spacing" in P:
        cb_kw["spacing"] = P["spacing"]
    cbar = fig.colorbar(mesh, **cb_kw)
    if P["ticklabels"] is not None:
        cbar.ax.set_yticklabels(P["ticklabels"])
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.tick_params(labelsize=CB_TICK_FS, length=3)
    cbar.outline.set_linewidth(0.8)

    # Titulo arriba de su barra de colores, alineado a la izquierda
    fig.text((x0 + TITLE_OFFSET) / FIG_W, (y_map + MAP_H + 0.22) / FIG_H, P["title"],
             fontsize=TITLE_FS, ha="left", va="bottom")

# ==========================================================================
# 7. PANEL INFERIOR e) - f)   (mismo estilo que H0_y_Snowfaction.py)
# ==========================================================================
MESES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PALETA = ["#DAA520", "#E8475F", "#C760C7", "#4F8FD9", "#8E5BC7",
          "#2CA02C", "#8C564B", "#17BECF"]
COLORES_H0 = ["#E8474B", "#3A78B5"]


def color_region(nombre, i):
    n = nombre.lower()
    claves = {"far north": 0, "norte grande": 0, "near north": 1, "norte chico": 1,
              "central": 2, "centro": 2, "south": 3, "sur": 3, "austral": 4}
    for k, c in claves.items():
        if k == n or (k in n and len(k) > 3):
            return PALETA[c]
    return PALETA[i % len(PALETA)]


# Dos graficos del mismo ancho que ocupan todo el ancho de la fila de mapas
EF_TOTAL_W = FIG_W - MARGIN_L - MARGIN_R
EF_W = (EF_TOTAL_W - 2 * EF_YLAB_W - EF_TWIN_W) / 2
y_ef = MARGIN_BOT + EF_XLAB_H
ax1 = ax_in(MARGIN_L + EF_YLAB_W, y_ef, EF_W, EF_AX_H)
ax2 = ax_in(MARGIN_L + 2 * EF_YLAB_W + EF_W, y_ef, EF_W, EF_AX_H)

if os.path.exists(EF_PKL):
    with open(EF_PKL, "rb") as f:
        R = pickle.load(f)

    # ---- e) Snowfall fraction ----
    x = np.arange(12)
    for i, r in enumerate(R["regiones"]):
        c = color_region(r, i)
        m, sd = R["sf_media"][r], R["sf_std"][r]
        ax1.fill_between(x, np.clip(m - sd, 0, 1), np.clip(m + sd, 0, 1),
                         color=c, alpha=0.15, linewidth=0)
        ax1.plot(x, m, "-", color=c, lw=2, label=r)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MESES)
    ax1.set_xlim(-0.4, 11.4)
    ax1.set_ylim(0, SF_YMAX)
    ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.legend(loc="upper right", fontsize=EF_LEGEND_FS, framealpha=0.9)

    # ---- f) H0 ----
    lat_s, cord = R["lat_s"], R["cordillera"]
    ax2.fill_between(lat_s, 0, cord, color="0.80", linewidth=0,
                     antialiased=False, zorder=1, label="Elevation")
    ax2.plot(lat_s, cord, color="0.65", lw=0.8, zorder=1)
    ymax_h0 = 0
    for (e, vals), c in zip(R["perfil"].items(), COLORES_H0):
        ok = np.isfinite(vals)
        ax2.plot(R["lat_h0"][ok], vals[ok], "-", color=c, lw=2.2, zorder=3, label=e)
        if ok.any():
            ymax_h0 = max(ymax_h0, np.nanmax(vals))
    ymax = np.ceil(max(ymax_h0, np.nanmax(cord)) * 1.08 / 1000) * 1000
    ax2.set_ylim(0, ymax)
    ax2.set_xlim(np.floor(lat_s.min()) - 1, np.ceil(lat_s.max()) + 1)
    ax2.legend(loc="upper right", fontsize=EF_LEGEND_FS, framealpha=0.9)
else:
    print(f"[!] No se encontró {EF_PKL} -> los paneles e) y f) quedarán vacíos")
    for a in (ax1, ax2):
        a.text(0.5, 0.5, "sin datos", transform=a.transAxes, ha="center", color="red")

ax1.set_xlabel("Month", fontsize=EF_LABEL_FS)
ax1.set_ylabel(r"Snowfall fraction [$S_f/P$]", fontsize=EF_LABEL_FS)
ax1.set_title("e) SnowFall fraction", loc="center", fontsize=TITLE_FS)
ax1.grid(True, alpha=0.35)
ax1.tick_params(labelsize=EF_TICK_FS)

ax2.set_axisbelow(True)
ax2.grid(True, alpha=0.35, zorder=0)
ax2.xaxis.set_major_locator(MultipleLocator(5))
ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°S"))
ax2.set_xlabel("Latitude", fontsize=EF_LABEL_FS)
ax2.set_ylabel("Elevation and H0 [m a.s.l.]", fontsize=EF_LABEL_FS)
ax2.set_title("f) 0 °C isotherm height", loc="center", fontsize=TITLE_FS)
ax2.tick_params(labelsize=EF_TICK_FS)
ax2b = ax2.twinx()
ax2b.set_ylim(ax2.get_ylim())
ax2b.set_ylabel("Elevation [m a.s.l.]", fontsize=EF_LABEL_FS)
ax2b.tick_params(labelsize=EF_TICK_FS)

# ==========================================================================
# 8. GUARDAR
# ==========================================================================
os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
fig.savefig(OUT_PNG, dpi=DPI, facecolor="white")
fig.savefig(OUT_PDF, dpi=DPI, facecolor="white")
print(f"Figura guardada en:\n  {OUT_PNG}\n  {OUT_PDF}")
print(f"Tamaño de la figura: {FIG_W:.1f} x {FIG_H:.1f} pulgadas")
plt.show()