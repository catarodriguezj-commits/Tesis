"""
Created on Fri Oct  9 14:56:01 2026

@author: DESKTOP
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 16:11:11 2026

@author: DESKTOP
"""

"""
Mapa de elevacion de Chile por macrozonas
------------------------------------------
Grafica el NetCDF de elevacion (ETOPO) recortado a Chile, con los limites de
las 5 macrozonas (Far North, Near North, Central, South, Austral) superpuestos,
el mar en blanco y los paises vecinos (Argentina, Bolivia, Peru) en gris, con
el limite internacional bien marcado y ejes con coordenadas (lon/lat).

Requisitos:
    pip install geopandas xarray netCDF4 matplotlib shapely pyproj

Archivos esperados (ajusta las rutas si es necesario):
    - NC_PATH:  NetCDF con variables \'lon\', \'lat\', \'z\' (elevacion en metros)
    - SHP_PATH: Shapefile de macrozonas de Chile
                (.shp + .shx + .dbf + .prj; si te faltan .shx/.prj el script
                reconstruye el indice y asume EPSG:3857 como respaldo)

Los paises vecinos (Argentina, Bolivia, Peru) vienen embebidos como GeoJSON
dentro de este mismo script (ver NEIGHBORS_GEOJSON mas abajo), asi que NO
necesitas ningun archivo ni carpeta adicional para el fondo gris.
"""

import os
import json

os.environ["SHAPE_RESTORE_SHX"] = "YES"  # reconstruye el .shx si falta

import numpy as np
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.patches import RegularPolygon
from matplotlib.ticker import MultipleLocator, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely import contains_xy
from shapely.geometry import MultiLineString
from shapely.ops import linemerge

# ----------------------------------------------------------------------
# 1. CONFIGURACION
# ----------------------------------------------------------------------
NC_PATH = r"C:\Users\DESKTOP\Desktop\Tesis\ETOPO_land_Chile_Z_dZ_005deg_RESAMPLE_REGRID.nc"
SHP_PATH = r"C:\Users\DESKTOP\Desktop\Tesis\macrozona\Macrozonas_Chile.shp"
OUT_PATH = r"C:\Users\DESKTOP\Desktop\Tesis\primera figura\codigos\elevacion\mapa_chile_elevacion.png"

EPSG_FALLBACK = 3857
ZONE_NAMES = ["Far North", "Near North", "Central", "South", "Austral"]

CMAP_NAME = "gist_earth"       # se usa solo si CUSTOM_COLORS esta vacio
CUSTOM_COLORS = [
    "#2e7d32",  # verde (elevacion baja)
    "#c9d94a",  # amarillo
    "#d94a3d",  # rojo
    "#c8967a",  # cafe claro
    "#8a5a3b",  # cafe medio
    "#4a2f22",  # cafe oscuro (elevacion alta)
]
VMIN, VMAX = 0, 5500

LAND_GREY = "#cfcfcf"          # color de los paises vecinos
BORDER_COLOR = "#8a8a8a"       # borde de los paises vecinos entre si
CHILE_BORDER_COLOR = "black"   # limite internacional de Chile (marcado)
CHILE_BORDER_WIDTH = 1.5

# limites entre macrozonas (bien marcados)
MACRO_LINE_COLOR = "black"
MACRO_LINE_WIDTH = 2.8        # grosor de la linea entre macrozonas
MACRO_HALO_WIDTH = 5.0        # halo blanco alrededor (0 = sin halo)
MACRO_TOL_DEG = 0.02          # tolerancia para detectar el borde comun entre zonas
MACRO_MIN_LEN_DEG = 0.3       # se descartan fragmentos mas cortos (evita puntitos)

# margen horizontal: mas angosto al oeste (mar) y mas ancho al este para
# dejar ver una franja clara de Argentina/Bolivia en gris
PAD_LON_WEST = 0.45
PAD_LON_EAST = 1.3
PAD_LAT_TOP = 1.1
PAD_LAT_BOT = 1.0

# Encuadre comun de todos los mapas [oeste, este, sur, norte] en grados.
# (Reemplaza el calculo con PAD_*, que cortaba Tierra del Fuego y las islas.)
MAP_EXTENT = [-80, -65, -57, -17]

FIGSIZE = (7.2, 11.5)
DPI = 200

LABEL_FONTSIZE = 18        # nombres de macrozona (Far North, etc.)
CBAR_LABEL_FONTSIZE = 28   # "Elevation (m)"
CBAR_TICK_FONTSIZE = 24    # numeros de la barra de color
COORD_FONTSIZE = 24        # numeros de longitud/latitud en los ejes

TITLE_TEXT = "a) Elevation [m]"   # titulo (con unidad) arriba de la barra de color
TITLE_FONT = {"family": "Times New Roman", "size": 30}
AXES_FONT = "Times New Roman"  # fuente de los ejes (lon/lat) y de la barra de color

MASK_BUFFER_DEG = 0.1  # margen (grados) al recortar la elevacion al poligono de Chile

# Paises vecinos (Argentina, Bolivia, Peru), version simplificada, embebidos
# aqui mismo como GeoJSON para que el script no dependa de ningun archivo
# externo adicional.
NEIGHBORS_GEOJSON = r'''
{
"type": "FeatureCollection",
"name": "neighbors_simplified",
"crs": { "type": "name", "properties": { "name": "urn:ogc:def:crs:OGC:1.3:CRS84" } },
"features": [
{ "type": "Feature", "properties": { "name": "Argentina" }, "geometry": { "type": "MultiPolygon", "coordinates": [ [ [ [ -68.634010227583232, -52.636370458874488 ], [ -67.75, -53.85 ], [ -66.45, -54.45 ], [ -65.05, -54.7 ], [ -65.5, -55.2 ], [ -66.45, -55.25 ], [ -66.95992, -54.89681 ], [ -68.63335, -54.8695 ], [ -68.634010227583232, -52.636370458874488 ] ] ], [ [ [ -57.625133429582959, -30.216294854454262 ], [ -58.142440355040762, -32.044503676076154 ], [ -58.132647671121447, -33.040566908502015 ], [ -58.349611172098875, -33.263188978815407 ], [ -58.495442064026548, -34.431489760070079 ], [ -57.225829637263658, -35.288026625307879 ], [ -57.36235877137878, -35.977390232081476 ], [ -56.737487352105447, -36.413125909166553 ], [ -56.78828528504836, -36.901571547189334 ], [ -57.749156867083457, -38.183870538079887 ], [ -59.231857062401893, -38.720220228837235 ], [ -61.237445237865643, -38.928424574541197 ], [ -62.335956997310127, -38.827707208004334 ], [ -62.125763108962936, -39.424104913084847 ], [ -62.330530971919494, -40.172586358400338 ], [ -62.145994432205214, -40.676896661136723 ], [ -62.745802781816984, -41.028761488612098 ], [ -63.77049475773255, -41.166789239263693 ], [ -64.732089809819726, -40.802677097335149 ], [ -65.118035244391578, -41.064314874028909 ], [ -64.978560553635816, -42.05800099056934 ], [ -64.303407965742494, -42.359016208669509 ], [ -63.755947842042389, -42.043686618824495 ], [ -63.458059048095876, -42.563138116222405 ], [ -64.378803880456331, -42.873558444999688 ], [ -65.181803961839748, -43.495380954767796 ], [ -65.328823411710133, -44.501366062193696 ], [ -65.565268927661606, -45.036785577169795 ], [ -66.509965786389344, -45.039627780945857 ], [ -67.29379391139247, -45.55189625425519 ], [ -67.580546434180079, -46.30177296324257 ], [ -66.597066413017288, -47.033924655953825 ], [ -65.64102657740149, -47.23613453551193 ], [ -65.985088263600787, -48.133289076531135 ], [ -67.166178961847692, -48.697337334996945 ], [ -67.816087612566434, -49.869668877970383 ], [ -68.728745083273211, -50.264218438518832 ], [ -69.138539191347775, -50.732510267947795 ], [ -68.81556148952356, -51.771104011594126 ], [ -68.149994879820383, -52.349983406127677 ], [ -69.49836218939609, -52.142760912637272 ], [ -71.914803839796377, -52.009022305865898 ], [ -72.329403856074066, -51.42595631287243 ], [ -72.309973517532342, -50.677009779666321 ], [ -72.975746832964688, -50.741450290734285 ], [ -73.328050910114527, -50.378785088909915 ], [ -73.415435757120093, -49.318436374712967 ], [ -72.648247443314943, -48.878618259476831 ], [ -72.331160854772008, -48.244238376661798 ], [ -72.447355312780275, -47.738532810253517 ], [ -71.917258470330239, -46.884838148791772 ], [ -71.552009446891276, -45.560732924177103 ], [ -71.659315558545359, -44.973688653341426 ], [ -71.222778896759763, -44.784242852559416 ], [ -71.329800788036223, -44.407521661151655 ], [ -71.793622606071935, -44.207172133156064 ], [ -71.464056159130507, -43.787611179378345 ], [ -71.915423956983886, -43.408564548517447 ], [ -72.14889807807856, -42.254888197601375 ], [ -71.746803758415496, -42.051386407235981 ], [ -71.915734015577627, -40.832339369470688 ], [ -71.413516608349056, -38.916022230791143 ], [ -70.814664272734689, -38.552995293940739 ], [ -71.11862504747549, -37.576827487947241 ], [ -71.12188066270987, -36.658123874662323 ], [ -70.364769253201644, -36.005088799789917 ], [ -70.388049485949125, -35.169687595359491 ], [ -69.817309129501524, -34.1935714657983 ], [ -69.814776984319224, -33.273886000299825 ], [ -70.074399380153594, -33.091209812148051 ], [ -70.53506893581951, -31.365010267870311 ], [ -69.919008348251936, -30.336339206668281 ], [ -70.013550381129917, -29.367922865518572 ], [ -69.656130337183171, -28.459141127233686 ], [ -69.001234910748252, -27.52121388113618 ], [ -68.295541551370434, -26.899339694935779 ], [ -68.594799770772681, -26.506908868111296 ], [ -68.386001146097357, -26.185016371365215 ], [ -68.41765296087614, -24.518554782816881 ], [ -67.328442959244171, -24.02530323659095 ], [ -66.9852339341777, -22.986348565362839 ], [ -67.106673550063604, -22.735924574476417 ], [ -66.273339402924847, -21.83231047942072 ], [ -64.964892137294612, -22.075861504812327 ], [ -64.377021043542257, -22.79809132252354 ], [ -63.986838141522476, -21.99364430103595 ], [ -62.846468471921561, -22.034985446869449 ], [ -60.846564704009914, -23.880712579038292 ], [ -60.02896603050403, -24.032796319273274 ], [ -58.807128465394982, -24.771459242453311 ], [ -57.777217169817938, -25.16233977630904 ], [ -57.633660040911131, -25.603656508081642 ], [ -58.618173590719749, -27.123718763947096 ], [ -57.609759690976141, -27.395898532828387 ], [ -56.486701626192996, -27.548499037386293 ], [ -55.695845506398157, -27.387837009390864 ], [ -54.788794928595053, -26.621785577096134 ], [ -54.625290696823576, -25.739255466415514 ], [ -54.13004960795439, -25.547639255477254 ], [ -53.628348965048744, -26.124865004177472 ], [ -53.648735317587892, -26.92347258881609 ], [ -55.162286342984572, -27.881915378533463 ], [ -56.290899624239081, -28.852760512000895 ], [ -57.625133429582959, -30.216294854454262 ] ] ] ] } },
{ "type": "Feature", "properties": { "name": "Bolivia" }, "geometry": { "type": "Polygon", "coordinates": [ [ [ -69.529678107364958, -10.951734307502194 ], [ -68.271253628193264, -11.014521172736821 ], [ -68.048192308205387, -10.712059014532485 ], [ -67.173801235610739, -10.306812432499612 ], [ -66.646908331962806, -9.931331475466862 ], [ -65.338435228116424, -9.761987806846392 ], [ -65.444837002205389, -10.511451104375432 ], [ -65.321898769783019, -10.895872084194679 ], [ -65.402281460213032, -11.566270440317155 ], [ -64.316352912031604, -12.461978041232193 ], [ -63.196498786050569, -12.627032565972435 ], [ -62.803060268796386, -13.000653171442686 ], [ -62.127080857986385, -13.198780612849724 ], [ -61.713204311760776, -13.489202162330052 ], [ -61.084121263255653, -13.479383640194598 ], [ -60.503304002511136, -13.775954685117659 ], [ -60.459198167550028, -14.354007256734555 ], [ -60.26432634137737, -14.645979099183641 ], [ -60.251148851142936, -15.07721892665932 ], [ -60.542965664295153, -15.093910414289596 ], [ -60.158389655179036, -16.258283786690086 ], [ -58.24121985536668, -16.299573256091293 ], [ -58.388058437724041, -16.877109063385276 ], [ -58.280804002502251, -17.271710300366017 ], [ -57.734558274961003, -17.552468357007768 ], [ -57.498371141170992, -18.174187513911292 ], [ -57.676008877174311, -18.961839694904029 ], [ -57.949997321185826, -19.400004164306822 ], [ -57.853801642474508, -19.969995212486189 ], [ -58.166392381408045, -20.176700941653678 ], [ -58.183471442280506, -19.868399346600363 ], [ -59.115042487206111, -19.356906019775401 ], [ -60.043564622626491, -19.342746677327426 ], [ -61.786326463453769, -19.633736667562964 ], [ -62.265961269770798, -20.513734633061276 ], [ -62.291179368729225, -21.051634616787393 ], [ -62.685057135657885, -22.249029229422387 ], [ -62.846468471921561, -22.034985446869449 ], [ -63.986838141522476, -21.99364430103595 ], [ -64.377021043542257, -22.79809132252354 ], [ -64.964892137294612, -22.075861504812327 ], [ -66.273339402924847, -21.83231047942072 ], [ -67.106673550063604, -22.735924574476417 ], [ -67.828179897722734, -22.872918796482175 ], [ -68.21991309271128, -21.494346612231865 ], [ -68.757167121033746, -20.372657972904463 ], [ -68.442225104430918, -19.405068454671429 ], [ -68.966818406841867, -18.981683444904107 ], [ -69.100246955019486, -18.260125420812678 ], [ -69.59042375352405, -17.580011895419332 ], [ -68.959635382753305, -16.50069793057127 ], [ -69.389764166934711, -15.660129082911652 ], [ -69.160346645774951, -15.323973890853019 ], [ -69.339534674747014, -14.953195489158832 ], [ -68.948886684836594, -14.453639418193283 ], [ -68.88007951523997, -12.899729099176653 ], [ -68.665079718689626, -12.561300144097173 ], [ -69.529678107364958, -10.951734307502194 ] ] ] } },
{ "type": "Feature", "properties": { "name": "Peru" }, "geometry": { "type": "Polygon", "coordinates": [ [ [ -69.893635219996625, -4.298186944194327 ], [ -70.794768846302304, -4.251264743673303 ], [ -70.928843349883579, -4.401591485210368 ], [ -71.748405727816547, -4.593982842633011 ], [ -72.891927659787257, -5.274561455916981 ], [ -72.9645072089412, -5.741251315944893 ], [ -73.21971126981461, -6.089188734566078 ], [ -73.120027431923603, -6.629930922068239 ], [ -73.724486660441642, -6.91859547285064 ], [ -73.723401455363501, -7.340998630404414 ], [ -73.987235480429661, -7.523829847853065 ], [ -73.571059332967067, -8.424446709835834 ], [ -73.015382656532552, -9.032833347208062 ], [ -73.226713426390162, -9.462212823121234 ], [ -72.563033006465645, -9.520193780152717 ], [ -72.18489071316985, -10.053597914269432 ], [ -71.302412278921537, -10.079436130415374 ], [ -70.481893886991173, -9.490118096558845 ], [ -70.548685675728407, -11.009146823778465 ], [ -70.093752204046893, -11.123971856331012 ], [ -69.529678107364958, -10.951734307502194 ], [ -68.665079718689626, -12.561300144097173 ], [ -68.88007951523997, -12.899729099176653 ], [ -68.948886684836594, -14.453639418193283 ], [ -69.339534674747014, -14.953195489158832 ], [ -69.160346645774951, -15.323973890853019 ], [ -69.389764166934711, -15.660129082911652 ], [ -68.959635382753305, -16.50069793057127 ], [ -69.858443569605868, -18.092693780187012 ], [ -70.372572394477714, -18.347975355708869 ], [ -71.375250210236928, -17.773798516513857 ], [ -71.462040778271131, -17.363487644116383 ], [ -73.444529588500416, -16.359362888252996 ], [ -75.237882656541444, -15.265682875227782 ], [ -76.009205084929945, -14.649286390850321 ], [ -76.423469204397747, -13.823186944232432 ], [ -76.25924150257417, -13.535039157772943 ], [ -77.106192389621839, -12.22271615972082 ], [ -78.092152879534638, -10.377712497604065 ], [ -79.036953091126946, -8.386567884965892 ], [ -79.445920376284846, -7.93083342858386 ], [ -79.760578172510051, -7.194340915560084 ], [ -80.537481655586078, -6.541667575713717 ], [ -81.249996304026425, -6.136834405139183 ], [ -80.926346808582437, -5.690556735866565 ], [ -81.410942552399462, -4.736764825055459 ], [ -81.099669562489368, -4.036394138203697 ], [ -80.302560594387216, -3.404856459164713 ], [ -80.184014858709673, -3.821161797708044 ], [ -80.469294603176948, -4.059286797708999 ], [ -80.442241990872162, -4.425724379090674 ], [ -80.02890804718561, -4.346090996928893 ], [ -79.624979214176179, -4.454198093283495 ], [ -79.205289069317729, -4.959128513207389 ], [ -78.639897223612337, -4.547784112164074 ], [ -78.450683966775642, -3.873096612161376 ], [ -77.837904832658609, -3.003020521663103 ], [ -76.635394253226721, -2.608677666843818 ], [ -75.544995693652041, -1.56160979574588 ], [ -75.233722703741947, -0.911416924649529 ], [ -75.373223232713855, -0.15203175212045 ], [ -75.106624518520078, -0.05720549886486 ], [ -74.441600511355972, -0.530820000819887 ], [ -74.122395189089062, -1.002832533373848 ], [ -73.6595035468346, -1.260491224781134 ], [ -73.070392218707241, -2.308954359550953 ], [ -72.325786505813653, -2.434218031426454 ], [ -71.774760708285399, -2.169789727388938 ], [ -71.413645799429787, -2.342802422702128 ], [ -70.813475714791963, -2.256864515800743 ], [ -70.047708502874855, -2.725156345229699 ], [ -70.692682054309714, -3.742872002785859 ], [ -70.394043952094989, -3.766591485207825 ], [ -69.893635219996625, -4.298186944194327 ] ] ] } }
]
}
'''

# ----------------------------------------------------------------------
# 2. MACROZONAS DE CHILE
# ----------------------------------------------------------------------
gdf = gpd.read_file(SHP_PATH)
if gdf.crs is None:
    gdf = gdf.set_crs(epsg=EPSG_FALLBACK)
gdf = gdf.to_crs(epsg=4326)
gdf = gdf.reset_index(drop=True)
gdf["zone"] = ZONE_NAMES[: len(gdf)]

# nos quedamos con el poligono continental mas grande de cada zona (sin
# islas oceanicas) para definir el recorte del mapa y los limites internos
mainland_geoms = []
for geom in gdf.geometry:
    polys = sorted(geom.geoms, key=lambda p: p.area, reverse=True)
    mainland_geoms.append(polys[0])
mainland = gpd.GeoSeries(mainland_geoms, crs=gdf.crs)
mainland_bounds = np.array([g.bounds for g in mainland_geoms])
lon_min, lon_max = mainland_bounds[:, 0].min(), mainland_bounds[:, 2].max()
lat_min, lat_max = mainland_bounds[:, 1].min(), mainland_bounds[:, 3].max()

xlim = (MAP_EXTENT[0], MAP_EXTENT[1])
ylim = (MAP_EXTENT[2], MAP_EXTENT[3])

mean_lat = (ylim[0] + ylim[1]) / 2
aspect = 1 / np.cos(np.radians(mean_lat))  # correccion de aspecto real

# contorno nacional disuelto (con TODOS los poligonos, incl. islas). Se
# simplifica un poco y se aplica un pequeno buffer (~medio pixel de la
# grilla) para que el recorte no deje huecos falsos por desajuste entre la
# grilla (0.22 grados) y el detalle del contorno (fiordos/islas angostas).
# Se usa este MISMO poligono (buffer incluido) tanto para enmascarar la
# elevacion como para dibujar el borde negro, de modo que nunca quede un
# pixel de color fuera de la linea del borde.
chile_full = gdf.geometry.union_all().simplify(0.02, preserve_topology=True)
chile_mask_poly = chile_full.buffer(MASK_BUFFER_DEG)

# ----------------------------------------------------------------------
# 3. GRILLA DE ELEVACION, recortada al poligono real de Chile
# ----------------------------------------------------------------------
# El NetCDF viene recortado a un rectangulo (lon/lat) alrededor de Chile,
# no al limite politico real -> trae terreno de Argentina/Bolivia/Peru con
# elevacion > 0 dentro de ese rectangulo. Para que SOLO Chile tenga color,
# enmascaramos celda por celda contra el poligono real del pais.
ds = xr.open_dataset(NC_PATH)
lon = ds["lon"].values
lat = ds["lat"].values
z = ds["z"].values.astype(float)

lon2d, lat2d = np.meshgrid(lon, lat)
inside_chile = contains_xy(chile_mask_poly, lon2d, lat2d)
z = np.where(inside_chile, z, np.nan)
z[z <= 0] = np.nan  # por si quedara algun resto de mar dentro del poligono

# ----------------------------------------------------------------------
# 4. PAISES VECINOS (Argentina, Bolivia, Peru en gris) -- embebidos arriba
# ----------------------------------------------------------------------
neighbors = gpd.GeoDataFrame.from_features(
    json.loads(NEIGHBORS_GEOJSON)["features"], crs="EPSG:4326"
)

# ----------------------------------------------------------------------
# 5. FIGURA
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=FIGSIZE)
fig.patch.set_facecolor("white")
ax.set_facecolor("white")  # el mar queda blanco

# paises vecinos rellenos en gris (Chile no se rellena: queda cubierto por
# la grilla de elevacion, que va encima)
neighbors.plot(ax=ax, facecolor=LAND_GREY, edgecolor=BORDER_COLOR, linewidth=0.6, zorder=2)

cmap = mcolors.LinearSegmentedColormap.from_list("elevation_custom", CUSTOM_COLORS, N=256) \
    if CUSTOM_COLORS else plt.get_cmap(CMAP_NAME).copy()
cmap.set_bad("none")  # fuera de Chile: transparente -> deja ver el blanco/gris de abajo
norm = mcolors.Normalize(vmin=VMIN, vmax=VMAX)
mesh = ax.pcolormesh(lon, lat, z, cmap=cmap, norm=norm, shading="nearest", zorder=3)

# limites internos entre macrozonas, bien negritos. Se dibuja SOLO el borde
# que comparten dos macrozonas vecinas (contorno exterior de una zona que
# toca a la otra). Asi no aparecen lagos, fiordos ni islas como puntos negros.
def _exteriores(geom):
    partes = geom.geoms if hasattr(geom, "geoms") else [geom]
    return MultiLineString([list(p.exterior.coords) for p in partes])

_segmentos = []
_zonas = list(gdf.geometry)
for i in range(len(_zonas)):
    for j in range(i + 1, len(_zonas)):
        comun = _exteriores(_zonas[i]).intersection(_zonas[j].buffer(MACRO_TOL_DEG))
        if comun.is_empty:
            continue
        comun = linemerge(comun) if comun.geom_type == "MultiLineString" else comun
        partes = comun.geoms if hasattr(comun, "geoms") else [comun]
        _segmentos += [g for g in partes
                       if g.geom_type in ("LineString", "MultiLineString")
                       and g.length >= MACRO_MIN_LEN_DEG]
macro_lines = gpd.GeoSeries(_segmentos, crs=gdf.crs)
halo = [pe.Stroke(linewidth=MACRO_HALO_WIDTH, foreground="white"), pe.Normal()] \
    if MACRO_HALO_WIDTH > 0 else None
macro_lines.plot(ax=ax, color=MACRO_LINE_COLOR, linewidth=MACRO_LINE_WIDTH,
                 capstyle="round", path_effects=halo, zorder=5.5)

# limite internacional de Chile, bien marcado -- mismo poligono (con buffer
# e islas) usado para enmascarar la elevacion, para que coincida exacto
gpd.GeoSeries([chile_mask_poly], crs=gdf.crs).boundary.plot(
    ax=ax, color=CHILE_BORDER_COLOR, linewidth=CHILE_BORDER_WIDTH, zorder=5
)

ax.set_xlim(*xlim)
ax.set_ylim(*ylim)
ax.set_aspect(aspect)
ax.set_xlabel("")  # geopandas reciente agrega 'Geodetic latitude/longitude'
ax.set_ylabel("")
for spine in ax.spines.values():
    spine.set_visible(True)
    spine.set_color("0.25")
    spine.set_linewidth(0.9)

# ----------------------------------------------------------------------
# 6. EJES CON COORDENADAS (longitud abajo, latitud a la derecha)
# ----------------------------------------------------------------------
def lon_fmt(x, pos):
    return f"{abs(x):.0f}°W"


def lat_fmt(y, pos):
    return f"{abs(y):.0f}°S"


ax.xaxis.set_major_locator(MultipleLocator(3))
ax.yaxis.set_major_locator(MultipleLocator(5))
ax.xaxis.set_major_formatter(FuncFormatter(lon_fmt))
ax.yaxis.set_major_formatter(FuncFormatter(lat_fmt))
ax.yaxis.tick_right()
ax.tick_params(axis="both", labelsize=COORD_FONTSIZE, length=4)
plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontfamily=AXES_FONT)
plt.setp(ax.get_yticklabels(), fontfamily=AXES_FONT)

# ----------------------------------------------------------------------
# 7. BARRA DE COLOR (texto grande)
# ----------------------------------------------------------------------
# Misma disposicion que los otros mapas: barra a lo alto del mapa, al lado
# del oceano (izquierda), con el titulo y la unidad ARRIBA de la barra.
divider = make_axes_locatable(ax)
cax = divider.append_axes("left", size="5%", pad=0.8)
cbar = fig.colorbar(mesh, cax=cax)
cbar.ax.yaxis.set_ticks_position("left")
cbar.ax.tick_params(labelsize=CBAR_TICK_FONTSIZE)
cbar.ax.set_title(TITLE_TEXT, loc="left", fontdict=TITLE_FONT, pad=20)
plt.setp(cbar.ax.get_yticklabels(), fontfamily=AXES_FONT)
cbar.outline.set_linewidth(0.6)

# ----------------------------------------------------------------------
# 8. ETIQUETAS DE MACROZONA (centradas sobre cada zona)
# ----------------------------------------------------------------------
for row_geom, (_, row) in zip(mainland_geoms, gdf.iterrows()):
    minx, miny, maxx, maxy = row_geom.bounds
    y_mid = (miny + maxy) / 2
    x_mid = row_geom.centroid.x
    ax.text(
        x_mid, y_mid, row["zone"],
        fontsize=LABEL_FONTSIZE, fontweight="bold", ha="center", va="center", zorder=6,
        color="0.1",
        path_effects=[pe.withStroke(linewidth=3, foreground="white")],
    )

# ----------------------------------------------------------------------
# 9. ROSA DE LOS VIENTOS
# ----------------------------------------------------------------------
cx, cy = xlim[1] - 0.95, ylim[1] - 1.05
size = 0.68
ax.add_patch(RegularPolygon((cx, cy), numVertices=4, radius=size, orientation=0,
                             facecolor="0.15", edgecolor="none", zorder=7))
ax.add_patch(RegularPolygon((cx, cy), numVertices=4, radius=size * 0.55, orientation=np.pi / 4,
                             facecolor="white", edgecolor="0.15", linewidth=0.6, zorder=8))
ax.text(cx, cy + size * 1.45, "N", fontsize=18, fontweight="bold", ha="center", va="center", zorder=8, color="0.15")

# ----------------------------------------------------------------------
# 10. GUARDAR
# ----------------------------------------------------------------------
# (el titulo ahora va arriba de la barra de color, ver seccion 7)

plt.tight_layout()
fig.savefig(OUT_PATH, dpi=DPI, facecolor="white")
print(f"Mapa guardado en: {OUT_PATH}")