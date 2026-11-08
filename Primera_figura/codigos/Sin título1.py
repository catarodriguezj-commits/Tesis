# -*- coding: utf-8 -*-
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle

# --- Rutas -----------------------------------------------------------------
shp = r"C:\Users\DESKTOP\Desktop\Tesis\shape_macrozona\Macrozonas_Chile.shp"
salida = r"C:\Users\DESKTOP\Desktop\Tesis\shape_macrozona\Macrozonas_Chile.pdf"

# --- Extension del mapa [oeste, este, sur, norte] --------------------------
#   Mas mar al oeste -> bajar el 1er valor | Mas mar al sur -> bajar el 3ero
EXTENT = [-94, -64, -60, -17.3]

# --- Datos -------------------------------------------------------------------
macro = gpd.read_file(shp).to_crs(4326)   # a grados lat/lon

colores = {'Far North': '#e6550d', 'Near North': '#fdae6b', 'Central': '#31a354',
           'South': '#3182bd', 'Austral': '#756bb1'}
orden = list(colores)

# --- Figura ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8.27, 11.69))   # A4 vertical
macro.plot(ax=ax, color=[colores[z] for z in macro['Macrozone']],
           edgecolor='k', linewidth=0.25)

ax.set_xlim(EXTENT[0], EXTENT[1])
ax.set_ylim(EXTENT[2], EXTENT[3])
ax.set_yticks(np.arange(-60, -15, 5))   # etiquetas cada 5°, desde -60

# Recuadro zona sin demarcar (Campo de Hielo Sur)
ax.add_patch(Rectangle((-73.8, -50.0), 1.4, 1.0, fill=False, ec='red', lw=1.2, ls='--'))
ax.annotate('Zona sin demarcar\n(Campo de Hielo Sur)', xy=(-72.4, -49.5), xytext=(-92, -53),
            fontsize=8, color='red', arrowprops=dict(arrowstyle='->', color='red', lw=0.8))

# Leyenda con superficie de cada macrozona
areas = macro.set_index('Macrozone').loc[orden, 'area_km']
leyenda = [Patch(fc=colores[z], ec='k',
                 label=f'{z}  ({areas[z]:,.0f} km²)'.replace(',', '.')) for z in orden]
ax.legend(handles=leyenda, title='Macrozona', loc='upper left', fontsize=9, title_fontsize=10)

ax.set_title('Macrozonas de Chile (DGA)', fontsize=14, pad=12)
ax.set_xlabel('Longitud (°)')
ax.set_ylabel('Latitud (°)')
ax.grid(ls=':', lw=0.4, alpha=0.6)
fig.text(0.5, 0.015,
         'Fuente: elaboración propia a partir de la División Regional de la BCN.\n'
         'Isla de Pascua (incluida en Central) queda fuera del área mostrada.\n'
         'Los límites no comprometen al Estado de Chile (DFL N° 83 de 1979).',
         ha='center', fontsize=7, color='0.3')
fig.subplots_adjust(left=0.1, right=0.95, top=0.94, bottom=0.1)

fig.savefig(salida)                                    # PDF vectorial
fig.savefig(salida.replace('.pdf', '.png'), dpi=300)   # y PNG
plt.show()