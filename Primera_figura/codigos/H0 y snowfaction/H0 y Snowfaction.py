# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 15:49:49 2026

@author: DESKTOP
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Oct  6 14:15:20 2026

@author: DESKTOP
"""

# -*- coding: utf-8 -*-
"""
Snowfall fraction (Sf/P) mensual por macrozona y altura de la isoterma 0 °C (H0)
por latitud, a partir de CR2MET v2.5 (diario, 0.05°), siguiendo la metodología de:

    Vásquez N A et al. (2025) Robust spatial changes in climate classes: insights
    from bias-corrected CMIP6 models across Chile. Environ. Res. Lett. 20 014061.

Metodología (sección 2.3, 2.4 y 2.6 del paper):
    - Tmean diaria = (Tmax + Tmin) / 2
    - Nieve (Sf) = precipitación de los días con Tmean <= 2 °C (Jennings et al. 2018)
    - Sf/P por mes, macrozona y año -> media y desviación estándar interanual
    - H0 (FLH) = altitud de la isoterma 0 °C estimada con una REGRESIÓN LINEAL
      LOCAL entre la elevación de cada celda y su Tmean diaria (gradiente
      topográfico), calculada para cada día de tormenta y cada banda de latitud:
          Tmean = a + b·z   ->   H0 = -a / b
      Día de tormenta: al menos 10 % del área de la banda con pr >= 5 mm/d.
      H0 final = promedio de H0 de los días de tormenta de cada estación.

Requisitos:
    pip install xarray netCDF4 numpy pandas matplotlib geopandas shapely
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import shapely
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator

# =============================================================================
# CONFIGURACIÓN
# =============================================================================
DIR_PR = r"C:\Users\DESKTOP\Desktop\Tesis\Datos\CR2MET2\CR2MET_pr_v2.5\pr"
DIR_TXN = r"C:\Users\DESKTOP\Desktop\Tesis\Datos\CR2MET2\CR2MET_txn_v2.5\txn"
PATRON_TXN = "CR2MET_tmin_tmax_v2.5_day_{yyyy}_{mm}_005deg.nc"

RUTA_DEM = r"C:\Users\DESKTOP\Desktop\Tesis\ETOPO_land_Chile_Z_dZ_005deg_RESAMPLE_REGRID.nc"
VAR_DEM = None            # None = autodetecta ('Z', 'z', 'elev', ...)

RUTA_SHP = r"C:\Users\DESKTOP\Desktop\Tesis\macrozona\Macrozonas_Chile.shp"
CAMPO_NOMBRE = None       # columna con el nombre de la macrozona (None = autodetecta)

AÑO_INICIO = None         # None = todo el período disponible
AÑO_FIN = None

# --- Paper ---
T_UMBRAL_NIEVE = 2.0      # °C  (Sf: Tmean <= 2 °C)
PR_TORMENTA = 5.0         # mm/d
FRAC_AREA_TORMENTA = 0.10 # fracción mínima del área con pr >= PR_TORMENTA
SOLO_TORMENTAS = True     # False = usa todos los días (no solo tormentas)

# Estaciones para H0 (meses). El paper usa Otoño-Invierno (AMJJAS);
# el verano se define como ONDJFM (Tabla 1 del paper).
ESTACIONES_H0 = {
    "Spring–Summer [ONDJFM]": [10, 11, 12, 1, 2, 3],
    "Fall–Winter [AMJJAS]":   [4, 5, 6, 7, 8, 9],
}
COLORES_H0 = ["#E8474B", "#3A78B5"]

BANDA_LAT = 0.5           # ancho (°) de la banda de latitud donde se ajusta la regresión
MIN_PIXELES = 30          # mínimo de celdas válidas para ajustar la regresión
MIN_DIAS = 5              # mínimo de días de tormenta para entregar H0 en una banda
H0_RANGO = (-500, 8000)   # se descartan regresiones fuera de este rango (m)

VENTANA_SUAVIZADO_DEM = 9 # filas (0.05° c/u) del máximo móvil del perfil de cordillera

TAM_TICKS = 20            # numeros de los ejes
TAM_LABELS = 15
TAM_TITULO = 20
TAM_LEYENDA = 12

FUENTE = "Times New Roman"  # fuente de toda la figura (titulos, ejes, leyenda)
SF_YMAX = 0.7              # limite superior del eje de Snowfall fraction

SALIDA_FIG = "SnowFraction_H0_CR2MET.png"
SALIDA_CSV_SF = "SnowFraction_mensual.csv"
SALIDA_CSV_H0 = "H0_por_latitud.csv"

MESES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
PALETA = ["#DAA520", "#E8475F", "#C760C7", "#4F8FD9", "#8E5BC7",
          "#2CA02C", "#8C564B", "#17BECF"]


# =============================================================================
# LECTURA
# =============================================================================
def listar_archivos():
    archivos = []
    regex = re.compile(r"_day_(\d{4})_(\d{2})_005deg\.nc$")
    for f_pr in sorted(glob.glob(os.path.join(DIR_PR, "CR2MET_pr_v2.5_day_*_005deg.nc"))):
        m = regex.search(os.path.basename(f_pr))
        if not m:
            continue
        yyyy, mm = int(m.group(1)), int(m.group(2))
        if AÑO_INICIO is not None and yyyy < AÑO_INICIO:
            continue
        if AÑO_FIN is not None and yyyy > AÑO_FIN:
            continue
        f_txn = os.path.join(DIR_TXN, PATRON_TXN.format(yyyy=yyyy, mm=f"{mm:02d}"))
        if not os.path.exists(f_txn):
            print(f"  [aviso] falta {os.path.basename(f_txn)}, se omite {yyyy}-{mm:02d}")
            continue
        archivos.append((yyyy, mm, f_pr, f_txn))
    if not archivos:
        raise FileNotFoundError("No se encontraron pares de archivos pr / tmin_tmax.")
    return archivos


def normalizar_coords(ds):
    ren = {}
    for c in list(ds.coords) + list(ds.dims):
        cl = str(c).lower()
        if cl in ("latitude", "y", "lat") and c != "lat":
            ren[c] = "lat"
        elif cl in ("longitude", "x", "lon") and c != "lon":
            ren[c] = "lon"
    return ds.rename(ren) if ren else ds


def buscar_variable(ds, candidatos):
    for v in ds.data_vars:
        if v.lower() in candidatos:
            return v
    raise KeyError(f"No encontré ninguna de {candidatos} en {list(ds.data_vars)}")


def abrir(ruta):
    return normalizar_coords(xr.open_dataset(ruta, decode_times=False))


def cargar_dem(lat, lon):
    ds = normalizar_coords(xr.open_dataset(RUTA_DEM, decode_times=False))
    if VAR_DEM is not None:
        v = VAR_DEM
    else:
        cands = [v for v in ds.data_vars if v.lower() in
                 ("z", "elev", "elevation", "topo", "dem", "altitude", "height", "orog")]
        v = cands[0] if cands else list(ds.data_vars)[0]
    print(f"  DEM: variable '{v}' de {os.path.basename(RUTA_DEM)}")
    dem = ds[v].squeeze(drop=True)
    dem = dem.where(dem > -1000).sortby("lat").sortby("lon")
    z = dem.interp(lat=xr.DataArray(lat, dims="lat"),
                   lon=xr.DataArray(lon, dims="lon"), method="nearest")
    ds.close()
    return z.transpose("lat", "lon").values.astype(float)


def mascaras_macrozonas(lat, lon):
    """Rasteriza el shapefile de macrozonas en la grilla CR2MET (centros de celda)."""
    gdf = gpd.read_file(RUTA_SHP)
    if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)

    campo = CAMPO_NOMBRE
    if campo is None:
        cols = [c for c in gdf.columns if c != "geometry" and
                (gdf[c].dtype == object or pd.api.types.is_string_dtype(gdf[c]))]
        if not cols:
            raise KeyError(f"Define CAMPO_NOMBRE. Columnas disponibles: {list(gdf.columns)}")
        campo = cols[0]
    print(f"  Macrozonas: campo '{campo}' -> {list(gdf[campo])}")

    gdf = gdf.dissolve(by=campo, as_index=False)
    gdf["ycent"] = gdf.geometry.representative_point().y
    gdf = gdf.sort_values("ycent", ascending=False)       # de norte a sur

    LON, LAT = np.meshgrid(lon, lat)
    mascaras = {}
    for _, fila in gdf.iterrows():
        geom = fila.geometry
        shapely.prepare(geom)
        mascaras[str(fila[campo])] = shapely.contains_xy(geom, LON, LAT)
    return mascaras


# =============================================================================
# H0: regresión lineal local T = a + b·z
# =============================================================================
def h0_regresion(T, P, z):
    """
    T, P : (dias, n_pixeles)   z : (n_pixeles,)
    Devuelve H0 (dias,) y un booleano de días válidos/tormenta.
    """
    valido = np.isfinite(T) & np.isfinite(z)[None, :]
    n = valido.sum(axis=1)
    zt = np.where(valido, z[None, :], np.nan)
    Tt = np.where(valido, T, np.nan)
    with np.errstate(all="ignore"):
        zbar = np.nanmean(zt, axis=1)
        tbar = np.nanmean(Tt, axis=1)
        dz = zt - zbar[:, None]
        cov = np.nansum(dz * (Tt - tbar[:, None]), axis=1)
        var = np.nansum(dz ** 2, axis=1)
        b = cov / var
        a = tbar - b * zbar
        h0 = -a / b
    ok = (n >= MIN_PIXELES) & (var > 0) & (b < 0) & np.isfinite(h0)
    ok &= (h0 > H0_RANGO[0]) & (h0 < H0_RANGO[1])

    if SOLO_TORMENTAS:
        Pv = np.where(np.isfinite(P), P, np.nan)
        n_p = np.isfinite(Pv).sum(axis=1)
        with np.errstate(all="ignore"):
            frac = (Pv >= PR_TORMENTA).sum(axis=1) / n_p
        ok &= (n_p > 0) & (frac >= FRAC_AREA_TORMENTA)
    return h0, ok


# =============================================================================
# CÁLCULO
# =============================================================================
def calcular():
    archivos = listar_archivos()
    años = sorted({a for a, *_ in archivos})
    idx_año = {a: i for i, a in enumerate(años)}
    print(f"Archivos: {len(archivos)}  |  Período: {años[0]}–{años[-1]}")

    ds0 = abrir(archivos[0][2])
    lat = ds0["lat"].values.astype(float)
    lon = ds0["lon"].values.astype(float)
    ds0.close()
    ny, nx = lat.size, lon.size

    z = cargar_dem(lat, lon)
    z_flat = z.ravel()
    mascaras = mascaras_macrozonas(lat, lon)
    regiones = list(mascaras)

    # Bandas de latitud para la regresión de H0
    bordes = np.arange(np.floor(lat.min() / BANDA_LAT) * BANDA_LAT,
                       np.ceil(lat.max() / BANDA_LAT) * BANDA_LAT + BANDA_LAT, BANDA_LAT)
    LAT2D = np.repeat(lat[:, None], nx, axis=1).ravel()
    bandas = []
    for i in range(bordes.size - 1):
        idx = np.where((LAT2D >= bordes[i]) & (LAT2D < bordes[i + 1]) & np.isfinite(z_flat))[0]
        if idx.size >= MIN_PIXELES:
            bandas.append((0.5 * (bordes[i] + bordes[i + 1]), idx))
    centros = np.array([c for c, _ in bandas])

    mes_a_est = {m: e for e, ms in ESTACIONES_H0.items() for m in ms}
    sumH = {e: np.zeros(len(bandas)) for e in ESTACIONES_H0}
    sumH2 = {e: np.zeros(len(bandas)) for e in ESTACIONES_H0}
    nH = {e: np.zeros(len(bandas)) for e in ESTACIONES_H0}

    P_tot = {r: np.full((len(años), 12), np.nan) for r in regiones}
    P_nie = {r: np.full((len(años), 12), np.nan) for r in regiones}
    mask_tierra = np.zeros((ny, nx), bool)

    for k, (yyyy, mm, f_pr, f_txn) in enumerate(archivos, 1):
        print(f"  [{k}/{len(archivos)}] {yyyy}-{mm:02d}", end="\r")
        with abrir(f_pr) as dpr, abrir(f_txn) as dtx:
            v_pr = buscar_variable(dpr, ("pr", "prcp", "precip", "precipitation"))
            v_tn = buscar_variable(dtx, ("tmin", "tn", "tasmin"))
            v_tx = buscar_variable(dtx, ("tmax", "tx", "tasmax"))
            pr = dpr[v_pr].transpose(..., "lat", "lon").values.astype(float)
            tn = dtx[v_tn].transpose(..., "lat", "lon").values.astype(float)
            tx = dtx[v_tx].transpose(..., "lat", "lon").values.astype(float)

        if pr.shape[-2:] != (ny, nx):
            raise ValueError(f"Grilla distinta en {f_pr}")
        nd = min(pr.shape[0], tn.shape[0], tx.shape[0])
        pr, tn, tx = pr[:nd], tn[:nd], tx[:nd]
        pr[pr < 0] = np.nan
        tm = 0.5 * (tn + tx)
        mask_tierra |= np.isfinite(tm).any(axis=0)

        # ---- Snowfall fraction ----
        nieve = np.where(tm <= T_UMBRAL_NIEVE, pr, 0.0)
        nieve[~np.isfinite(pr) | ~np.isfinite(tm)] = np.nan
        ia, im = idx_año[yyyy], mm - 1
        for r, msk in mascaras.items():
            P_tot[r][ia, im] = np.nansum(pr[:, msk])
            P_nie[r][ia, im] = np.nansum(nieve[:, msk])

        # ---- H0 por regresión local ----
        if mm in mes_a_est:
            e = mes_a_est[mm]
            T2 = tm.reshape(nd, -1)
            P2 = pr.reshape(nd, -1)
            for j, (_, idx) in enumerate(bandas):
                h0, ok = h0_regresion(T2[:, idx], P2[:, idx], z_flat[idx])
                if ok.any():
                    sumH[e][j] += h0[ok].sum()
                    sumH2[e][j] += (h0[ok] ** 2).sum()
                    nH[e][j] += ok.sum()
    print()

    # ---- Resultados Sf/P ----
    sf_media, sf_std, filas = {}, {}, []
    for r in regiones:
        with np.errstate(all="ignore"):
            sf = P_nie[r] / P_tot[r]
        sf[~(P_tot[r] > 0)] = np.nan
        with np.errstate(all="ignore"):
            sf_media[r] = np.nanmean(sf, axis=0)
            sf_std[r] = np.nanstd(sf, axis=0)
        for m in range(12):
            filas.append({"macrozona": r, "mes": m + 1,
                          "SfP_media": sf_media[r][m], "SfP_std": sf_std[r][m]})
    pd.DataFrame(filas).to_csv(SALIDA_CSV_SF, index=False)

    # ---- Resultados H0 ----
    lat_s_band = -centros
    orden_b = np.argsort(lat_s_band)
    lat_s_band = lat_s_band[orden_b]
    perfil, tabla = {}, {"lat_S": lat_s_band}
    for e in ESTACIONES_H0:
        n_ = nH[e][orden_b]
        with np.errstate(all="ignore"):
            med = sumH[e][orden_b] / n_
            std = np.sqrt(np.maximum(sumH2[e][orden_b] / n_ - med ** 2, 0))
        med[n_ < MIN_DIAS] = np.nan
        tabla[f"H0_{e}_m"] = med
        tabla[f"H0_{e}_std_m"] = std
        tabla[f"n_dias_{e}"] = n_.astype(int)
        perfil[e] = med
    pd.DataFrame(tabla).to_csv(SALIDA_CSV_H0, index=False)

    # ---- Perfil de cordillera (gris), sin huecos ----
    z_tierra = np.where(mask_tierra, z, np.nan)
    with np.errstate(all="ignore"):
        cordillera = np.nanmax(z_tierra, axis=1)
    lat_s = -lat
    orden = np.argsort(lat_s)
    lat_s, cordillera = lat_s[orden], cordillera[orden]
    cordillera = rellenar_nan_1d(lat_s, cordillera)
    cordillera = pd.Series(cordillera).rolling(VENTANA_SUAVIZADO_DEM, center=True,
                                               min_periods=1).max().values
    cordillera = np.clip(cordillera, 0, None)

    return regiones, sf_media, sf_std, lat_s, cordillera, lat_s_band, perfil


def rellenar_nan_1d(x, y):
    y = np.asarray(y, float).copy()
    ok = np.isfinite(y)
    if ok.sum() >= 2:
        y[~ok] = np.interp(x[~ok], x[ok], y[ok])
    return y


# =============================================================================
# GRÁFICOS
# =============================================================================
def color_region(nombre, i):
    n = nombre.lower()
    claves = {"far north": 0, "norte grande": 0, "near north": 1, "norte chico": 1,
              "central": 2, "centro": 2, "south": 3, "sur": 3, "austral": 4}
    for k, c in claves.items():
        if k == n or (k in n and len(k) > 3):
            return PALETA[c]
    return PALETA[i % len(PALETA)]


def graficar(regiones, sf_media, sf_std, lat_s, cordillera, lat_h0, perfil):
    plt.rcParams.update({
        "font.family": FUENTE,
        "mathtext.fontset": "custom",       # S_f/P tambien en Times New Roman
        "mathtext.rm": FUENTE,
        "mathtext.it": f"{FUENTE}:italic",
        "mathtext.bf": f"{FUENTE}:bold",
    })
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))

    # ---- d) Snowfall fraction ----
    x = np.arange(12)
    for i, r in enumerate(regiones):
        c = color_region(r, i)
        m, s = sf_media[r], sf_std[r]
        ax1.fill_between(x, np.clip(m - s, 0, 1), np.clip(m + s, 0, 1),
                         color=c, alpha=0.15, linewidth=0)
        ax1.plot(x, m, "-", color=c, lw=2, label=r)
    ax1.set_xticks(x)
    ax1.set_xticklabels(MESES)
    ax1.set_xlim(-0.4, 11.4)
    ax1.set_ylim(0, SF_YMAX)
    ax1.yaxis.set_major_locator(MultipleLocator(0.1))
    ax1.set_xlabel("Month", fontsize=TAM_LABELS)
    ax1.set_ylabel(r"Snowfall fraction [$S_f/P$]", fontsize=TAM_LABELS)
    ax1.set_title("e) SnowFall fraction", loc="left", fontsize=TAM_TITULO)
    ax1.legend(loc="upper right", fontsize=TAM_LEYENDA, framealpha=0.9)
    ax1.grid(True, alpha=0.35)
    ax1.tick_params(labelsize=TAM_TICKS)

    # ---- e) H0 ----
    ax2.set_axisbelow(True)
    ax2.grid(True, alpha=0.35, zorder=0)
    ax2.fill_between(lat_s, 0, cordillera, color="0.80", linewidth=0,
                     antialiased=False, zorder=1, label="Elevation")
    ax2.plot(lat_s, cordillera, color="0.65", lw=0.8, zorder=1)
    ymax_h0 = 0
    for (e, vals), c in zip(perfil.items(), COLORES_H0):
        ok = np.isfinite(vals)
        ax2.plot(lat_h0[ok], vals[ok], "-", color=c, lw=2.2, zorder=3, label=e)
        if ok.any():
            ymax_h0 = max(ymax_h0, np.nanmax(vals))

    ymax = np.ceil(max(ymax_h0, np.nanmax(cordillera)) * 1.08 / 1000) * 1000
    ax2.set_ylim(0, ymax)
    ax2.set_xlim(np.floor(lat_s.min()) - 1, np.ceil(lat_s.max()) + 1)
    ax2.xaxis.set_major_locator(MultipleLocator(5))
    ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}°S"))
    ax2.set_xlabel("Latitude", fontsize=TAM_LABELS)
    ax2.set_ylabel("Elevation and H0 [m a.s.l.]", fontsize=TAM_LABELS)
    ax2.set_title("f) 0 °C isotherm height", loc="left", fontsize=TAM_TITULO)
    ax2.legend(loc="upper right", fontsize=TAM_LEYENDA, framealpha=0.9)
    ax2.tick_params(labelsize=TAM_TICKS)

    ax2b = ax2.twinx()
    ax2b.set_ylim(ax2.get_ylim())
    ax2b.set_ylabel("Elevation [m a.s.l.]", fontsize=TAM_LABELS)
    ax2b.tick_params(labelsize=TAM_TICKS)

    fig.tight_layout()
    ruta_fig = os.path.abspath(SALIDA_FIG)
    try:
        fig.savefig(ruta_fig, dpi=300, bbox_inches="tight")
    except OSError as err:
        # El PNG anterior suele estar abierto/bloqueado (visor de imágenes, OneDrive)
        base, ext = os.path.splitext(ruta_fig)
        ruta_fig = f"{base}_{pd.Timestamp.now():%Y%m%d_%H%M%S}{ext}"
        print(f"  [aviso] no se pudo escribir {SALIDA_FIG} ({err}); "
              f"se guarda como {os.path.basename(ruta_fig)}")
        fig.savefig(ruta_fig, dpi=300, bbox_inches="tight")
    print(f"Figura guardada en: {ruta_fig}")
    plt.show()


if __name__ == "__main__":
    resultados = calcular()

    # Guardar resultados para la figura compuesta (figura_1_compuesta.py)
    import pickle
    CACHE_DIR = r"C:\Users\DESKTOP\Desktop\Tesis\primera figura\datos_figura"
    os.makedirs(CACHE_DIR, exist_ok=True)
    claves = ["regiones", "sf_media", "sf_std", "lat_s", "cordillera", "lat_h0", "perfil"]
    ruta_pkl = os.path.join(CACHE_DIR, "ef_SnowFraction_H0.pkl")
    with open(ruta_pkl, "wb") as f:
        pickle.dump(dict(zip(claves, resultados)), f)
    print(f"Resultados guardados en: {ruta_pkl}")

    graficar(*resultados)