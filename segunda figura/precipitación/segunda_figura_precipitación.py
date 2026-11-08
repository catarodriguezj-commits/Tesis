# -*- coding: utf-8 -*-
"""
Series de precipitacion anual por macrozona: CR2MET y todas las familias de
modelos (CMIP5, CMIP6, HighResMIP, RCMs), datos crudos (sin corregir ni
suavizar).

Figura: una FILA por macrozona (Far North -> Austral) y una COLUMNA por
familia (CMIP5, CMIP6, HighResMIP, RCMs). En cada columna, la familia de esa
columna va en color y todas las demas en gris; CR2MET en negro. Solo se
muestran los anos de los periodos historico, futuro 1 y futuro 2, uno al lado
del otro y separados por lineas verticales punteadas.

Como funciona
  1. Para cada modelo lee TODOS sus archivos diarios (historico + futuro),
     calcula la precipitacion anual de cada ano [mm/ano] y la promedia sobre
     cada macrozona. Lo mismo para CR2MET.
  2. Guarda los resultados en CSV (CACHE_DIR), uno por familia. La lectura de
     los datos se hace UNA sola vez: si el CSV ya existe se usa directamente
     (pon FORZAR_RECALCULO = True para rehacerlo).
  3. Dibuja la figura.

Precipitacion anual: promedio de pr diaria del ano x 365.25 (x 86400 si
viene en kg m-2 s-1), igual que en los scripts delta_prep_*.py. Asi los
calendarios de 360 dias (HadGEM) quedan comparables.
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

CR2MET_DIR = os.path.join(DATOS, "CR2MET2", "CR2MET_pr_v2.5", "pr")

CMIP5_ROOTS = [os.path.join(DATOS, "CMIP5", "Pr_hist"), os.path.join(DATOS, "CMIP5", "Pr_futuro")]
CMIP6_ROOTS = [os.path.join(DATOS, "CMIP6", "Pr_hist"), os.path.join(DATOS, "CMIP6", "Pr_futuro")]

HIGHRES_ROOTS = [os.path.join(DATOS, "HighResMIP", "pr_historico"),
                 os.path.join(DATOS, "HighResMIP", "pr_futuro")]
HIGHRES_MODELS = ["CNRM-CM6-1-HR", "EC-Earth3P-HR", "HadGEM3-GC31-HM", "HadGEM3-GC31-LL",
                  "HadGEM3-GC31-MM", "MPI-ESM1-2-HR", "MPI-ESM1-2-XR"]

RCMS_ROOT = os.path.join(DATOS, "RCMs")
RCM_MODELS = ["HadGem2-REGCM47", "HadGEM2-REMO2015", "mpi-esm-lr-REMO2015",
              "MPI-ESM-MR-REGCM47", "NorESM1-REGCM47", "NorESM1-REMO2015", "RegCM4_CL_CR2"]

CACHE_DIR = os.path.join(TESIS, "series_pr_macrozonas")
OUT_PNG = os.path.join(CACHE_DIR, "series_pr_macrozonas.png")
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

FIGSIZE = (21, 17)          # figura de 4 columnas (FIGURA = "columnas")

# Tipo de figura:
#   "bandas"   -> UN panel por macrozona; cada familia es una banda de color
#                 (dispersion entre sus modelos) + su promedio multimodelo.
#   "columnas" -> la version anterior: 4 columnas, una familia en color y el
#                 resto en gris.
FIGURA = "bandas"
BANDA = "minmax"            # "minmax" = minimo-maximo de los modelos
                            # "p10p90" = percentiles 10-90
                            # "std"    = promedio +/- 1 desviacion estandar
ALPHA_BANDA = 0.22
LW_MEDIA = 1.8
FIGSIZE_BANDAS = (14, 18)
DPI = 300

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
TITLE_FS, LABEL_FS, TICK_FS, LEGEND_FS = 24, 20, 17, 17   # antes 18, 15, 12, 13

SECONDS_PER_DAY = 86400
DAYS_PER_YEAR = 365.25

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
def factor_unidades(da, nombre):
    """Factor para pasar el promedio diario a mm/ano."""
    u = str(da.attrs.get("units", "")).lower().strip()
    if u in ("mm/day", "mm d-1", "mm day-1", "mm/d", "mm"):
        return DAYS_PER_YEAR
    if u not in ("kg m-2 s-1", "kg/m2/s", "kg m^-2 s^-1", "mm s-1", "mm/s"):
        print(f"    [!] {nombre}: unidades '{u}' no reconocidas, se asume kg m-2 s-1")
    return SECONDS_PER_DAY * DAYS_PER_YEAR


def serie_anual_modelo(archivos, nombre):
    """Lee todos los archivos de un modelo y devuelve un DataFrame con
    columnas zona, year, pr (mm/ano)."""
    if not archivos:
        print(f"    [!] {nombre}: sin archivos")
        return None

    def _solo_pr(ds):
        return ds[["pr"]]

    try:
        ds = xr.open_mfdataset(sorted(archivos), combine="nested", concat_dim="time",
                               chunks={"time": 365}, coords="minimal", compat="override",
                               data_vars="minimal", preprocess=_solo_pr)
    except Exception as e:
        print(f"    [!] {nombre}: open_mfdataset falló ({type(e).__name__}); se abre archivo por archivo")
        ds = xr.concat([_solo_pr(xr.open_dataset(f, chunks={"time": 365})) for f in sorted(archivos)],
                       dim="time", coords="minimal", compat="override")

    # Fechas repetidas (archivos que se traslapan) -> se conserva la primera
    t = ds.indexes["time"]
    if t.duplicated().any():
        print(f"    [!] {nombre}: {int(t.duplicated().sum())} días repetidos -> se eliminan")
        ds = ds.isel(time=~t.duplicated(keep="first"))
    ds = ds.sortby("time")

    pr = ds["pr"]
    factor = factor_unidades(pr, nombre)

    # Solo los anos de los periodos historico, futuro 1 y futuro 2
    pr = pr.sel(time=pr["time"].dt.year.isin(ANOS_PERIODOS))
    if pr.sizes["time"] == 0:
        print(f"    [!] {nombre}: sin datos dentro de los periodos")
        return None
    pr = pr.chunk({"time": 365})

    # Campo anual (ano, lat, lon) en mm/ano; se descartan anos con < 300 dias
    n_dias = pr["time"].dt.year.to_series().value_counts()
    anual = (pr.groupby("time.year").mean("time", skipna=True) * factor).compute()
    anos_ok = [a for a in anual["year"].values if n_dias.get(a, 0) >= 300]
    anual = anual.sel(year=anos_ok)

    masks = mascaras_zonas(anual["lat"].values, anual["lon"].values)
    filas = []
    vals = anual.values
    for z, m in masks.items():
        if m.sum() == 0:
            continue
        serie = np.nanmean(vals[:, m], axis=1)
        filas += [{"zona": z, "year": int(a), "pr": float(v)}
                  for a, v in zip(anual["year"].values, serie)]
    ds.close()
    print(f"    {nombre}: {len(anos_ok)} años dentro de los periodos")
    return pd.DataFrame(filas)


# ==========================================================================
# 5. BUSQUEDA DE ARCHIVOS POR FAMILIA
# ==========================================================================
RE_CMIP5 = re.compile(r"pr_day_(?P<model>.+?)_(?:historical|rcp\d+)(?:_recortado)?_r\d+i\d+p\d+(?:_recortado)?_.*\.nc$", re.I)
RE_CMIP6 = re.compile(r"pr_day_(?P<model>.+?)_(?:historical|ssp\d+)(?:_recortado)?_r\d+i\d+p\d+f\d+(?:_recortado)?_.*\.nc$", re.I)


def modelos_cmip(roots, regex):
    modelos = {}
    for root in roots:
        for f in glob.glob(os.path.join(root, "**", "pr_day_*.nc"), recursive=True):
            m = regex.search(os.path.basename(f))
            if m:
                modelos.setdefault(m.group("model"), []).append(f)
    return modelos


def modelos_highres():
    modelos = {}
    for name in HIGHRES_MODELS:
        archivos = []
        for root in HIGHRES_ROOTS:
            d = os.path.join(root, name)
            fs = glob.glob(os.path.join(d, "pr_day_*_regrid_005deg_conservative.nc")) or \
                glob.glob(os.path.join(d, "*.nc"))
            archivos += fs
        modelos[name] = archivos
    return modelos


def modelos_rcm():
    modelos = {}
    for name in RCM_MODELS:
        fs = []
        for f in glob.glob(os.path.join(RCMS_ROOT, name, "*.nc")):
            norm = os.path.basename(f).lower().replace(".", "").replace("_", "").replace("-", "")
            if "pr" in norm and "tas" not in norm:
                fs.append(f)
        modelos[name] = fs
    return modelos


BUSCADORES = {
    "CMIP5": lambda: modelos_cmip(CMIP5_ROOTS, RE_CMIP5),
    "CMIP6": lambda: modelos_cmip(CMIP6_ROOTS, RE_CMIP6),
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
        pd.DataFrame(columns=["zona", "year", "pr", "modelo"])
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
    patron = re.compile(r"CR2MET_pr_v2[._]5_day_(\d{4})_(\d{2})_005deg\.nc$")
    totales = {}   # (ano, mes) -> {zona: mm}
    for f in sorted(glob.glob(os.path.join(CR2MET_DIR, "CR2MET_pr_v2*5_day_*_005deg.nc"))):
        m = patron.search(os.path.basename(f))
        if not m or int(m.group(1)) not in ANOS_PERIODOS:
            continue
        with xr.open_dataset(f) as ds:
            tot = ds["pr"].sum("time", skipna=True).where(ds["cl_mask"] == 1).values
            masks = mascaras_zonas(ds["lat"].values, ds["lon"].values)
        totales[(int(m.group(1)), int(m.group(2)))] = {z: np.nanmean(tot[mk]) for z, mk in masks.items()}
    filas = []
    for ano in sorted({a for a, _ in totales}):
        if all((ano, mes) in totales for mes in range(1, 13)):   # solo anos completos
            for z in ZONAS:
                filas.append({"zona": z, "year": ano,
                              "pr": sum(totales[(ano, mes)][z] for mes in range(1, 13))})
    df = pd.DataFrame(filas)
    df.to_csv(csv, index=False)
    print(f"CR2MET: {df['year'].nunique()} años completos guardados en {csv}")
    return df


# ==========================================================================
# 5c. WRF-SAAG (climatologias por periodo)
# ==========================================================================
def valores_wrf():
    """{zona: {panel: precipitacion media anual [mm/ano]}} desde el NetCDF de WRF."""
    if not MOSTRAR_WRF or not os.path.exists(WRF_NC):
        if MOSTRAR_WRF:
            print(f"[!] No se encontró {WRF_NC} -> la figura sale sin WRF")
        return {}
    ds = xr.open_dataset(WRF_NC)
    masks = mascaras_zonas(ds["lat"].values, ds["lon"].values)
    res = {z: {} for z in ZONAS}
    for panel, suf in WRF_PERIODOS.items():
        da = ds[f"PREC_{suf}"]
        campo = da.values.astype(float)
        u = str(da.attrs.get("units", "")).lower()
        med = np.nanmedian(campo)
        # unidades: se aceptan mm/ano, mm/dia o mm/mes
        if "day" in u or "d-1" in u or (not u and med < 30):
            campo = campo * DAYS_PER_YEAR
            nota = "mm/día -> x365.25"
        elif "month" in u or "mon" in u:
            campo = campo * 12
            nota = "mm/mes -> x12"
        else:
            nota = "mm/año"
        print(f"WRF PREC_{suf}: unidades '{u or 'sin unidades'}' (mediana {med:.2f}) -> {nota}")
        for z, m in masks.items():
            res[z][panel] = float(np.nanmean(campo[m]))
    ds.close()
    return res


# ==========================================================================
# 5b. CONTROL DE UNIDADES (por modelo)
# ==========================================================================
# Algunos archivos dicen "kg m-2 s-1" pero en realidad vienen en mm/dia (o al
# reves). Eso deja un modelo con valores ~86400 veces mas grandes que el resto
# y aplasta a todos los demas contra el cero. Aqui se revisa la magnitud de
# cada modelo y se corrige si es claramente imposible.
P_MAX_RAZONABLE = 20000   # mm/ano (promedio de la macrozona)
P_MIN_RAZONABLE = 1       # mm/ano


def corregir_unidades(df, familia):
    if df.empty:
        return df
    df = df.copy()
    # Se revisa por modelo Y por periodo, porque el historico y el futuro
    # vienen de archivos distintos que pueden tener unidades distintas.
    periodo = pd.Series("otro", index=df.index)
    for nombre, (ini, fin) in PERIODOS.items():
        periodo[(df["year"] >= ini) & (df["year"] <= fin)] = nombre
    for (modelo, per), g in df.groupby(["modelo", periodo]):
        med = g["pr"].median()
        factor = 1.0
        if med > P_MAX_RAZONABLE:
            factor = 1 / SECONDS_PER_DAY      # venia en mm/dia y se multiplico por 86400
        elif 0 < med < P_MIN_RAZONABLE:
            factor = SECONDS_PER_DAY          # venia en kg m-2 s-1 y no se multiplico
        if factor != 1.0:
            df.loc[g.index, "pr"] *= factor
            print(f"    [!] {familia} / {modelo} ({per}): mediana {med:,.1f} mm/año no es realista "
                  f"-> unidades corregidas (x{factor:g}); nueva mediana "
                  f"{df.loc[g.index, 'pr'].median():,.0f} mm/año")
    return df


def resumen(series, obs):
    """Tabla de control: mediana historica por modelo (mm/ano, todas las zonas)."""
    ini, fin = list(PERIODOS.values())[0]
    print("\nControl: precipitación media 1980-2000 por modelo [mm/año]")
    print(f"  {'CR2MET':<28s} {obs[(obs.year >= ini) & (obs.year <= fin)]['pr'].mean():8.0f}")
    for fam, df in series.items():
        h = df[(df.year >= ini) & (df.year <= fin)]
        for modelo, g in h.groupby("modelo"):
            print(f"  {fam + ' / ' + modelo:<28s} {g['pr'].mean():8.0f}")


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
            ax.plot(x_eje(nombre, g["year"]), g["pr"], **kw)


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
                ax.set_ylabel(f"{zona}\nP [mm yr$^{{-1}}$]", fontsize=LABEL_FS)
            if i == len(ZONAS) - 1:
                ax.set_xlabel("Year", fontsize=LABEL_FS)

        axes[i, 0].set_ylim(bottom=0)

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


def estadisticos(df_zona, ini, fin):
    """Media y limites de la banda de una familia, ano a ano, en un periodo."""
    d = df_zona[(df_zona["year"] >= ini) & (df_zona["year"] <= fin)]
    if d.empty:
        return None
    tabla = d.pivot_table(index="year", columns="modelo", values="pr")
    media = tabla.mean(axis=1)
    if BANDA == "p10p90":
        lo, hi = tabla.quantile(0.10, axis=1), tabla.quantile(0.90, axis=1)
    elif BANDA == "std":
        sd = tabla.std(axis=1).fillna(0)
        lo, hi = media - sd, media + sd
    else:
        lo, hi = tabla.min(axis=1), tabla.max(axis=1)
    return tabla.index.values, media.values, lo.values, hi.values, tabla.shape[1]


def figura_bandas(series, obs):
    fig, axes = plt.subplots(len(ZONAS), 1, figsize=FIGSIZE_BANDAS, sharex=True)

    ticks = [float(x_eje(n, (ini + fin) / 2)) for n, (ini, fin) in PERIODOS.items()]
    labels = [f"{ini}–{fin}" for ini, fin in PERIODOS.values()]
    minor = [float(x_eje(n, a)) for n, (ini, fin) in PERIODOS.items()
             for a in range(ini, fin + 1, 5)]

    for i, zona in enumerate(ZONAS):
        ax = axes[i]
        for k, fam in enumerate(FAMILIAS):
            df = series[fam]
            dz = df[df["zona"] == zona]
            for nombre, (ini, fin) in PERIODOS.items():
                st = estadisticos(dz, ini, fin)
                if st is None:
                    continue
                anos, media, lo, hi, n_mod = st
                x = x_eje(nombre, anos)
                c = COLOR_FAMILIA[fam]
                ax.fill_between(x, lo, hi, color=c, alpha=ALPHA_BANDA, lw=0, zorder=1 + k)
                ax.plot(x, media, color=c, lw=LW_MEDIA, zorder=5 + k)

        o = obs[obs["zona"] == zona]
        for nombre, (ini, fin) in PERIODOS.items():
            g = o[(o["year"] >= ini) & (o["year"] <= fin)].sort_values("year")
            if len(g):
                ax.plot(x_eje(nombre, g["year"]), g["pr"], color=COLOR_OBS, lw=LW_OBS, zorder=10)
        dibujar_wrf(ax, zona)

        for nombre, (ini, fin) in list(PERIODOS.items())[:-1]:
            ax.axvline(x_eje(nombre, fin) + SEPARACION / 2, color="0.25", ls="--", lw=1.0, zorder=0)
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
        ax.tick_params(labelsize=TICK_FS)
        ax.set_ylabel(f"{zona}\nP [mm yr$^{{-1}}$]", fontsize=LABEL_FS)
        if i == 0:
            for nombre, (ini, fin) in PERIODOS.items():
                ax.text(x_eje(nombre, (ini + fin) / 2), 1.03, nombre,
                        transform=ax.get_xaxis_transform(),
                        ha="center", va="bottom", fontsize=LABEL_FS, color="0.25")
    axes[-1].set_xlabel("Year", fontsize=LABEL_FS)

    texto_banda = {"minmax": "min–max", "p10p90": "P10–P90", "std": "±1 SD"}[BANDA]
    from matplotlib.patches import Patch
    leyenda = [Line2D([], [], color=COLOR_OBS, lw=LW_OBS, label="CR2MET")]
    for fam in FAMILIAS:
        leyenda.append((Patch(facecolor=COLOR_FAMILIA[fam], alpha=ALPHA_BANDA + 0.1),
                        Line2D([], [], color=COLOR_FAMILIA[fam], lw=LW_MEDIA)))
    etiquetas = ["CR2MET"] + list(FAMILIAS)
    if WRF_VALORES:
        leyenda.append(Line2D([], [], color=COLOR_WRF, lw=LW_WRF))
        etiquetas.append(WRF_LABEL)
    fig.legend(leyenda, etiquetas, loc="lower center", ncol=min(len(etiquetas), 4), fontsize=LEGEND_FS,
               frameon=False, bbox_to_anchor=(0.5, 0.0),
               title=f"Lines: multimodel mean   |   Bands: {texto_banda} of the models",
               title_fontsize=LEGEND_FS - 2)
    fig.tight_layout(rect=(0, 0.085, 1, 1))
    out_png = OUT_PNG.replace(".png", "_bandas.png")
    fig.savefig(out_png, dpi=DPI)
    fig.savefig(out_png.replace(".png", ".pdf"))
    print(f"\nFigura guardada en:\n  {out_png}\n  {out_png.replace('.png', '.pdf')}")
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
    if FIGURA == "bandas":
        figura_bandas(series, obs)
    else:
        figura(series, obs)