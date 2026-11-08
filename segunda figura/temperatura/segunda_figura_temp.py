# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 16:09:04 2026

@author: DESKTOP
"""

# -*- coding: utf-8 -*-
"""
Series de temperatura media anual por macrozona: CR2MET y todas las familias
de modelos (CMIP5, CMIP6, HighResMIP, RCMs), datos crudos (sin corregir ni
suavizar). Version de temperatura de series_pr_macrozonas.py.

Figura: una FILA por macrozona (Far North -> Austral) y una COLUMNA por
familia (CMIP5, CMIP6, HighResMIP, RCMs). En cada columna, la familia de esa
columna va en color y todas las demas en gris; CR2MET en negro. Solo se
muestran los anos de los periodos historico, futuro 1 y futuro 2, uno al lado
del otro y separados por lineas verticales punteadas.

Temperatura media diaria = (tasmax + tasmin) / 2 [°C], igual que en los
scripts delta_temp_*.py; la temperatura anual es el promedio de los dias del
ano.

Como funciona
  1. Para cada modelo lee sus archivos diarios de tasmax y tasmin (solo los
     anos de los periodos), calcula la temperatura media de cada ano y la
     promedia sobre cada macrozona. Lo mismo para CR2MET (tmin, tmax).
  2. Guarda los resultados en CSV (CACHE_DIR), uno por familia. La lectura de
     los datos se hace UNA sola vez: si el CSV ya existe se usa directamente
     (pon FORZAR_RECALCULO = True para rehacerlo).
  3. Dibuja la figura.
"""

import os
import re
import glob

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from shapely import contains_xy

# ==========================================================================
# 1. RUTAS
# ==========================================================================
TESIS = r"C:\Users\DESKTOP\Desktop\Tesis"
DATOS = os.path.join(TESIS, "Datos")

SHP_PATH = os.path.join(TESIS, "macrozona", "Macrozonas_Chile.shp")

CR2MET_DIR = os.path.join(DATOS, "CR2MET2", "CR2MET_txn_v2.5", "txn")

# CMIP5 y CMIP6: se buscan TODOS los archivos tasmax_day_*.nc y tasmin_day_*.nc
# dentro de estas carpetas (y sus subcarpetas), y se agrupan por modelo segun
# el nombre del archivo. Si tus archivos de temperatura estan en otra carpeta,
# cambiala aqui.
CMIP5_TEMP_ROOTS = [os.path.join(DATOS, "CMIP5")]
CMIP6_TEMP_ROOTS = [os.path.join(DATOS, "CMIP6")]

HIGHRES_TASMAX_ROOTS = [os.path.join(DATOS, "HighResMIP", "tasmax_hist"),
                        os.path.join(DATOS, "HighResMIP", "tasmax_fut")]
HIGHRES_TASMIN_ROOTS = [os.path.join(DATOS, "HighResMIP", "tasmin_hist"),
                        os.path.join(DATOS, "HighResMIP", "tasmin_fut")]
HIGHRES_MODELS = ["CNRM-CM6-1-HR", "EC-Earth3P-HR", "HadGEM3-GC31-HM", "HadGEM3-GC31-LL",
                  "HadGEM3-GC31-MM", "MPI-ESM1-2-HR", "MPI-ESM1-2-XR"]

RCMS_ROOT = os.path.join(DATOS, "RCMs")
RCM_MODELS = ["HadGem2-REGCM47", "HadGEM2-REMO2015", "mpi-esm-lr-REMO2015",
              "MPI-ESM-MR-REGCM47", "NorESM1-REGCM47", "NorESM1-REMO2015", "RegCM4_CL_CR2"]

# Carpeta de salida (CSV + figura)
CACHE_DIR = os.path.join(TESIS, "segunda figura", "temperatura")
OUT_PNG = os.path.join(CACHE_DIR, "series_temp_macrozonas.png")
OUT_PDF = OUT_PNG.replace(".png", ".pdf")

# WRF-SAAG: solo trae climatologias (un mapa por periodo), no series anuales.
# Se dibuja como una linea horizontal con el promedio de cada periodo:
#   historico WRF (2000-2020) -> sobre el panel "Hist." (ojo: no son los mismos anos)
#   futuro WRF (2060-2080)    -> sobre el panel "Fut. 2"
WRF_NC = os.path.join(TESIS, "WRF", "WRF_SAAG_PROYECCION_2060_2080_vs_2000_2020_005deg.nc")
MOSTRAR_WRF = True
WRF_PERIODOS = {"Hist.": "HIST", "Fut. 2": "FUT"}   # panel -> sufijo de la variable
WRF_LABEL = "WRF-SAAG (2000–2020, 2060–2080)"

FORZAR_RECALCULO = False   # True = vuelve a leer todos los datos aunque existan los CSV

# ==========================================================================
# 2. PERIODOS Y ESTILO
# ==========================================================================
PERIODOS = {                       # nombre corto (va sobre la 1a fila): (inicio, fin)
    "Hist.": (1980, 2000),
    "Fut. 1": (2030, 2050),
    "Fut. 2": (2060, 2080),
}
SOMBREAR_PERIODOS = False          # franja gris muy suave dentro de cada periodo
SEPARACION = 3                     # espacio (en "anos") entre un periodo y el siguiente

ANOS_PERIODOS = sorted({a for ini, fin in PERIODOS.values() for a in range(ini, fin + 1)})

ZONAS = ["Far North", "Near North", "Central", "South", "Austral"]
FAMILIAS = ["CMIP5", "CMIP6", "HighResMIP", "RCMs"]
COLOR_FAMILIA = {
    "CMIP5": "#E66101",
    "CMIP6": "#1F78B4",
    "HighResMIP": "#1A9850",
    "RCMs": "#984EA3",
}
COLOR_GRIS = "0.55"          # gris de las otras familias (0 = negro, 1 = blanco)
COLOR_OBS = "black"
COLOR_WRF = "#D7191C"       # rojo
LW_WRF = 3.0

LW_GRIS, LW_COLOR, LW_OBS = 0.6, 0.9, 1.6
ALPHA_GRIS, ALPHA_COLOR = 0.7, 0.85

FIGSIZE = (21, 17)
DPI = 300

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 24, 20, 17, 17   # antes 18, 15, 12, 13


# ==========================================================================
# 3. MASCARAS DE MACROZONAS (una por grilla distinta)
# ==========================================================================
macro = gpd.read_file(SHP_PATH)
if macro.crs is None:
    macro = macro.set_crs("EPSG:3857")
macro = macro.to_crs("EPSG:4326")
GEOMS = {row["Macrozone"]: row.geometry.simplify(0.01) for _, row in macro.iterrows()}

_cache_mascaras = {}


def mascaras_zonas(lat, lon):
    """Devuelve {zona: mascara booleana (lat, lon)} para la grilla dada."""
    clave = (len(lat), len(lon), round(float(lat[0]), 3), round(float(lon[0]), 3))
    if clave not in _cache_mascaras:
        lon2d, lat2d = np.meshgrid(lon, lat)
        _cache_mascaras[clave] = {z: contains_xy(GEOMS[z], lon2d, lat2d) for z in ZONAS}
    return _cache_mascaras[clave]


# ==========================================================================
# 4. LECTURA DE UN MODELO -> SERIE ANUAL POR MACROZONA
# ==========================================================================
def a_celsius(da, nombre):
    u = str(da.attrs.get("units", "")).lower().strip()
    if u in ("k", "kelvin", "degk", "deg_k"):
        return da - 273.15
    if u in ("c", "degc", "deg_c", "celsius", "degrees_celsius", "°c"):
        return da
    # sin unidades claras: se decide por la magnitud
    muestra = float(da.isel(time=slice(0, 30)).mean())
    if muestra > 100:
        return da - 273.15
    print(f"    [!] {nombre}: unidades '{u}' no reconocidas, se asume °C")
    return da


def abrir_variable(archivos, var, nombre):
    def _solo_var(ds):
        return ds[[var]]

    try:
        ds = xr.open_mfdataset(sorted(archivos), combine="nested", concat_dim="time",
                               chunks={"time": 365}, coords="minimal", compat="override",
                               data_vars="minimal", preprocess=_solo_var)
    except Exception as e:
        print(f"    [!] {nombre} ({var}): open_mfdataset falló ({type(e).__name__}); "
              f"se abre archivo por archivo")
        ds = xr.concat([_solo_var(xr.open_dataset(f, chunks={"time": 365})) for f in sorted(archivos)],
                       dim="time", coords="minimal", compat="override")
    t = ds.indexes["time"]
    if t.duplicated().any():
        print(f"    [!] {nombre} ({var}): {int(t.duplicated().sum())} días repetidos -> se eliminan")
        ds = ds.isel(time=~t.duplicated(keep="first"))
    ds = ds.sortby("time")
    da = ds[var]
    da = da.sel(time=da["time"].dt.year.isin(ANOS_PERIODOS))
    da = da.assign_coords(lat=np.round(da["lat"].values, 4), lon=np.round(da["lon"].values, 4))
    return a_celsius(da, f"{nombre} ({var})")


def serie_anual_modelo(archivos, nombre):
    """archivos = {"tasmax": [...], "tasmin": [...]}. Devuelve un DataFrame
    con columnas zona, year, valor (°C)."""
    if not archivos.get("tasmax") or not archivos.get("tasmin"):
        faltan = [v for v in ("tasmax", "tasmin") if not archivos.get(v)]
        print(f"    [!] {nombre}: sin archivos de {', '.join(faltan)}")
        return None

    tmax = abrir_variable(archivos["tasmax"], "tasmax", nombre)
    tmin = abrir_variable(archivos["tasmin"], "tasmin", nombre)
    if tmax.sizes["time"] == 0 or tmin.sizes["time"] == 0:
        print(f"    [!] {nombre}: sin datos dentro de los periodos")
        return None

    # solo los dias que existen en ambas variables
    tmax, tmin = xr.align(tmax, tmin, join="inner")
    tmean = ((tmax + tmin) / 2).chunk({"time": 365})

    n_dias = tmean["time"].dt.year.to_series().value_counts()
    anual = tmean.groupby("time.year").mean("time", skipna=True).compute()
    anos_ok = [a for a in anual["year"].values if n_dias.get(a, 0) >= 300]
    anual = anual.sel(year=anos_ok)

    masks = mascaras_zonas(anual["lat"].values, anual["lon"].values)
    filas = []
    vals = anual.values
    for z, m in masks.items():
        if m.sum() == 0:
            continue
        serie = np.nanmean(vals[:, m], axis=1)
        filas += [{"zona": z, "year": int(a), "valor": float(v)}
                  for a, v in zip(anual["year"].values, serie)]
    print(f"    {nombre}: {len(anos_ok)} años dentro de los periodos")
    return pd.DataFrame(filas)


# ==========================================================================
# 5. BUSQUEDA DE ARCHIVOS POR FAMILIA
# ==========================================================================
RE_MODELO = re.compile(r"(?P<var>tasmax|tasmin)_day_(?P<model>.+?)_(?:historical|rcp\d+|ssp\d+)", re.I)


def modelos_cmip(roots):
    modelos = {}
    for root in roots:
        for var in ("tasmax", "tasmin"):
            for f in glob.glob(os.path.join(root, "**", f"{var}_day_*.nc"), recursive=True):
                m = RE_MODELO.search(os.path.basename(f))
                if m:
                    modelos.setdefault(m.group("model"), {"tasmax": [], "tasmin": []})[var].append(f)
    # si un modelo tiene archivos regrillados a 0.05°, se usan solo esos
    for d in modelos.values():
        for var, fs in d.items():
            reg = [f for f in fs if "regrid" in os.path.basename(f).lower()]
            d[var] = reg or fs
    return modelos


def modelos_highres():
    modelos = {}
    for name in HIGHRES_MODELS:
        d = {"tasmax": [], "tasmin": []}
        for var, roots in (("tasmax", HIGHRES_TASMAX_ROOTS), ("tasmin", HIGHRES_TASMIN_ROOTS)):
            for root in roots:
                carpeta = os.path.join(root, name)
                d[var] += glob.glob(os.path.join(carpeta, f"{var}_day_*_regrid_005deg_conservative.nc")) or \
                    glob.glob(os.path.join(carpeta, f"{var}_*.nc"))
        modelos[name] = d
    return modelos


def modelos_rcm():
    modelos = {}
    for name in RCM_MODELS:
        d = {"tasmax": [], "tasmin": []}
        for f in glob.glob(os.path.join(RCMS_ROOT, name, "*.nc")):
            norm = os.path.basename(f).lower()
            for var in ("tasmax", "tasmin"):
                if var in norm:
                    d[var].append(f)
        modelos[name] = d
    return modelos


BUSCADORES = {
    "CMIP5": lambda: modelos_cmip(CMIP5_TEMP_ROOTS),
    "CMIP6": lambda: modelos_cmip(CMIP6_TEMP_ROOTS),
    "HighResMIP": modelos_highres,
    "RCMs": modelos_rcm,
}


def series_familia(familia):
    csv = os.path.join(CACHE_DIR, f"serie_{familia}.csv")
    if os.path.exists(csv) and not FORZAR_RECALCULO:
        print(f"{familia}: usando {csv}")
        return pd.read_csv(csv)
    print(f"{familia}: leyendo datos...")
    partes = []
    for nombre, archivos in sorted(BUSCADORES[familia]().items()):
        try:
            df = serie_anual_modelo(archivos, nombre)
        except Exception as e:
            print(f"    [!] {nombre}: error ({type(e).__name__}: {e}) -> se omite")
            df = None
        if df is not None and len(df):
            df["modelo"] = nombre
            partes.append(df)
    df = pd.concat(partes, ignore_index=True) if partes else \
        pd.DataFrame(columns=["zona", "year", "valor", "modelo"])
    df["familia"] = familia
    df.to_csv(csv, index=False)
    print(f"{familia}: {df['modelo'].nunique()} modelos guardados en {csv}")
    return df


def serie_cr2met():
    csv = os.path.join(CACHE_DIR, "serie_CR2MET.csv")
    if os.path.exists(csv) and not FORZAR_RECALCULO:
        print(f"CR2MET: usando {csv}")
        return pd.read_csv(csv)
    print("CR2MET: leyendo datos...")
    patron = re.compile(r"CR2MET_tmin_tmax_v2[._]5_day_(\d{4})_(\d{2})_005deg\.nc$")
    sumas = {}   # (ano, mes) -> ({zona: suma de temperaturas diarias}, n_dias)
    for f in sorted(glob.glob(os.path.join(CR2MET_DIR, "CR2MET_tmin_tmax_v2*5_day_*_005deg.nc"))):
        m = patron.search(os.path.basename(f))
        if not m or int(m.group(1)) not in ANOS_PERIODOS:
            continue
        with xr.open_dataset(f) as ds:
            tmean = ((ds["tmin"] + ds["tmax"]) / 2)
            n = tmean.sizes["time"]
            suma = tmean.sum("time", skipna=True).where(ds["cl_mask"] == 1).values
            masks = mascaras_zonas(ds["lat"].values, ds["lon"].values)
        sumas[(int(m.group(1)), int(m.group(2)))] = ({z: np.nanmean(suma[mk]) for z, mk in masks.items()}, n)
    filas = []
    for ano in sorted({a for a, _ in sumas}):
        if all((ano, mes) in sumas for mes in range(1, 13)):   # solo anos completos
            n_ano = sum(sumas[(ano, mes)][1] for mes in range(1, 13))
            for z in ZONAS:
                total = sum(sumas[(ano, mes)][0][z] for mes in range(1, 13))
                filas.append({"zona": z, "year": ano, "valor": total / n_ano})
    df = pd.DataFrame(filas)
    df.to_csv(csv, index=False)
    print(f"CR2MET: {df['year'].nunique()} años completos guardados en {csv}")
    return df


# ==========================================================================
# 5c. WRF-SAAG (climatologias por periodo)
# ==========================================================================
def valores_wrf():
    """{zona: {panel: temperatura media [°C]}} desde el NetCDF de WRF,
    con T = (T2MAX + T2MIN) / 2."""
    if not MOSTRAR_WRF or not os.path.exists(WRF_NC):
        if MOSTRAR_WRF:
            print(f"[!] No se encontró {WRF_NC} -> la figura sale sin WRF")
        return {}
    ds = xr.open_dataset(WRF_NC)
    masks = mascaras_zonas(ds["lat"].values, ds["lon"].values)
    res = {z: {} for z in ZONAS}
    for panel, suf in WRF_PERIODOS.items():
        campo = ((ds[f"T2MAX_{suf}"] + ds[f"T2MIN_{suf}"]) / 2).values.astype(float)
        if np.nanmedian(campo) > 100:          # Kelvin -> °C
            campo = campo - 273.15
        print(f"WRF T2 ({suf}): mediana {np.nanmedian(campo):.1f} °C")
        for z, m in masks.items():
            res[z][panel] = float(np.nanmean(campo[m]))
    ds.close()
    return res


# ==========================================================================
# 5b. CONTROL DE UNIDADES (por modelo y periodo)
# ==========================================================================
# Si algun modelo quedo en Kelvin (valores > 100), se pasa a °C.
def corregir_unidades(df, familia):
    if df.empty:
        return df
    df = df.copy()
    periodo = pd.Series("otro", index=df.index)
    for nombre, (ini, fin) in PERIODOS.items():
        periodo[(df["year"] >= ini) & (df["year"] <= fin)] = nombre
    for (modelo, per), g in df.groupby(["modelo", periodo]):
        med = g["valor"].median()
        if med > 100:
            df.loc[g.index, "valor"] -= 273.15
            print(f"    [!] {familia} / {modelo} ({per}): mediana {med:.1f} -> estaba en Kelvin, "
                  f"pasado a °C ({df.loc[g.index, 'valor'].median():.1f} °C)")
    return df


def resumen(series, obs):
    """Tabla de control: media historica por modelo (°C, todas las zonas)."""
    ini, fin = list(PERIODOS.values())[0]
    print("\nControl: temperatura media 1980-2000 por modelo [°C]")
    print(f"  {'CR2MET':<28s} {obs[(obs.year >= ini) & (obs.year <= fin)]['valor'].mean():8.1f}")
    for fam, df in series.items():
        h = df[(df.year >= ini) & (df.year <= fin)]
        for modelo, g in h.groupby("modelo"):
            print(f"  {fam + ' / ' + modelo:<28s} {g['valor'].mean():8.1f}")


# ==========================================================================
# 6. FIGURA
# ==========================================================================
# Eje X "comprimido": los tres periodos uno al lado del otro.
_OFFSETS, _x = {}, 0
for _nombre, (_ini, _fin) in PERIODOS.items():
    _OFFSETS[_nombre] = _x - _ini
    _x += (_fin - _ini) + SEPARACION
X_MAX = _x - SEPARACION


def x_eje(nombre_periodo, anos):
    return np.asarray(anos) + _OFFSETS[nombre_periodo]


WRF_VALORES = {}


def dibujar_wrf(ax, zona):
    """Linea horizontal con el promedio WRF de cada periodo."""
    for panel, val in WRF_VALORES.get(zona, {}).items():
        if panel in PERIODOS and np.isfinite(val):
            ini, fin = PERIODOS[panel]
            ax.hlines(val, x_eje(panel, ini), x_eje(panel, fin), color=COLOR_WRF,
                      lw=LW_WRF, zorder=9)


def dibujar_serie(ax, df, **kw):
    """Dibuja una serie por periodo (sin unir un periodo con el siguiente)."""
    for nombre, (ini, fin) in PERIODOS.items():
        g = df[(df["year"] >= ini) & (df["year"] <= fin)].sort_values("year")
        if len(g):
            ax.plot(x_eje(nombre, g["year"]), g["valor"], **kw)


def figura(series, obs):
    fig, axes = plt.subplots(len(ZONAS), len(FAMILIAS), figsize=FIGSIZE,
                             sharex=True, sharey="row")

    # Eje X: rango de anos centrado bajo cada periodo + marcas cada 5 anos
    ticks = [float(x_eje(n, (ini + fin) / 2)) for n, (ini, fin) in PERIODOS.items()]
    labels = [f"{ini}–{fin}" for ini, fin in PERIODOS.values()]
    minor = [float(x_eje(n, a)) for n, (ini, fin) in PERIODOS.items()
             for a in range(ini, fin + 1, 5)]

    for i, zona in enumerate(ZONAS):
        for j, fam_color in enumerate(FAMILIAS):
            ax = axes[i, j]

            # 1) todas las demas familias en gris (al fondo)
            for fam, df in series.items():
                if fam == fam_color:
                    continue
                for _, g in df[df["zona"] == zona].groupby("modelo"):
                    dibujar_serie(ax, g, color=COLOR_GRIS, lw=LW_GRIS, alpha=ALPHA_GRIS, zorder=1)

            # 2) la familia de esta columna en color
            df = series[fam_color]
            for _, g in df[df["zona"] == zona].groupby("modelo"):
                dibujar_serie(ax, g, color=COLOR_FAMILIA[fam_color], lw=LW_COLOR,
                              alpha=ALPHA_COLOR, zorder=2)

            # 3) CR2MET en negro y WRF
            dibujar_serie(ax, obs[obs["zona"] == zona], color=COLOR_OBS, lw=LW_OBS, zorder=4)
            dibujar_wrf(ax, zona)

            # 4) separacion entre periodos: linea vertical punteada
            for nombre, (ini, fin) in list(PERIODOS.items())[:-1]:
                ax.axvline(x_eje(nombre, fin) + SEPARACION / 2, color="0.25",
                           ls="--", lw=1.0, zorder=3)
            if SOMBREAR_PERIODOS:
                for nombre, (ini, fin) in PERIODOS.items():
                    ax.axvspan(*x_eje(nombre, [ini, fin]), color="0.5", alpha=0.07, lw=0, zorder=0)

            ax.set_xlim(-1, X_MAX + 1)
            ax.set_xticks(ticks)
            ax.set_xticklabels(labels)
            ax.set_xticks(minor, minor=True)
            ax.tick_params(axis="x", which="major", length=0)
            ax.grid(True, which="minor", axis="x", alpha=0.3, lw=0.5)
            ax.grid(True, which="major", axis="y", alpha=0.3, lw=0.5)
            ax.tick_params(axis="y", labelsize=TICK_FS)
            ax.tick_params(axis="x", labelsize=TICK_FS - 4)   # rangos de anos mas chicos para que no choquen
            if i == 0:
                ax.set_title(fam_color, fontsize=TITLE_FS, color=COLOR_FAMILIA[fam_color],
                             fontweight="bold", pad=26)
                for nombre, (ini, fin) in PERIODOS.items():
                    ax.text(x_eje(nombre, (ini + fin) / 2), 1.02, nombre,
                            transform=ax.get_xaxis_transform(),
                            ha="center", va="bottom", fontsize=TICK_FS, color="0.25")
            if j == 0:
                ax.set_ylabel(f"{zona}\nT [°C]", fontsize=LABEL_FS)
            if i == len(ZONAS) - 1:
                ax.set_xlabel("Year", fontsize=LABEL_FS)


    leyenda = [Line2D([], [], color=COLOR_OBS, lw=LW_OBS, label="CR2MET")] + \
              [Line2D([], [], color=COLOR_FAMILIA[f], lw=2, label=f) for f in FAMILIAS] + \
              [Line2D([], [], color=COLOR_GRIS, lw=2, label="Other families")]
    if WRF_VALORES:
        leyenda.append(Line2D([], [], color=COLOR_WRF, lw=LW_WRF, label=WRF_LABEL))
    fig.legend(handles=leyenda, loc="lower center", ncol=min(len(leyenda), 4), fontsize=LEGEND_FS,
               frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT_PNG, dpi=DPI)
    fig.savefig(OUT_PDF)
    print(f"\nFigura guardada en:\n  {OUT_PNG}\n  {OUT_PDF}")
    plt.show()


# ==========================================================================
# 7. MAIN
# ==========================================================================
if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    obs = serie_cr2met()
    series = {fam: corregir_unidades(series_familia(fam), fam) for fam in FAMILIAS}
    WRF_VALORES.update(valores_wrf())
    resumen(series, obs)
    if WRF_VALORES:
        print("\nWRF-SAAG por macrozona:")
        for z, d in WRF_VALORES.items():
            print(f"  {z:<12s} " + "   ".join(f"{k}: {v:8.1f}" for k, v in d.items()))
    figura(series, obs)