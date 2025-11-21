
"""
Script: procesar_calidad_aire.py  (v7: filtro PROVINCIA==28 para contaminantes)
Autor: (ChatGPT)

Resumen:
  - Lee CSVs de contaminantes (sep=';'), agrega a media diaria por contaminante
    (promediando 24h y entre estaciones), fusiona múltiples años en UNA columna por contaminante,
    y **filtra por PROVINCIA==28 (Comunidad de Madrid) si la columna existe**.
  - Imputa huecos por contaminante (interpolación lineal hasta 'limit' días + media de la columna).
  - Calcula media de los **5 días previos estrictos** (rolling(window=5, min_periods=5).mean().shift(1)).
  - Lee Excels de meteorología (detectados por contenido: 'Fecha' + Temp/Humedad Media),
    consolida años y crea columnas:
       * 'temperatura' = media 5 días previos estrictos (shift 1) de Temp Media (ºC)
       * 'humedad'     = media 5 días previos estrictos (shift 1) de Humedad Media (%)
  - Lee Excels de ingresos (detectados por contenido: 'Comunidad Autónoma' + 'Fecha simplificada'/'Fecha'),
    filtra CA==13 (Madrid) y agrega **conteos diarios** como 'ingresos'.
  - Exporta un Excel final con:
       [todas las columnas de contaminantes], 'temperatura', 'humedad', 'ingresos'
    Contiene SOLO días con **todas** las variables disponibles.

Uso:
  python procesar_calidad_aire.py --input_dir /ruta/carpeta --output /ruta/salida.xlsx --limit 3

Requisitos: pandas, numpy, openpyxl
"""

from __future__ import annotations
import argparse
from pathlib import Path
import re
import pandas as pd
import numpy as np

HOUR_COLS = [f"H{h:02d}" for h in range(1, 25)]  # H01..H24


# ---------- Utilidades ----------

def pollutant_from_filename(path: Path) -> str:
    """Nombre del contaminante: todo antes del primer '_' en el nombre de archivo; fallback = primeras letras/dígitos."""
    name = path.stem
    if "_" in name:
        return name.split("_", 1)[0]
    m = re.match(r"([A-Za-z0-9]+)", name)
    return m.group(1) if m else name


def read_csv_semicolon(path: Path) -> pd.DataFrame:
    """Lee CSV con separador ';' probando varias codificaciones comunes."""
    last_err = None
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return pd.read_csv(path, sep=";", encoding=enc, low_memory=False)
        except Exception as e:
            last_err = e
    raise RuntimeError(f"No se pudo leer {path} (sep=';'). Último error: {last_err}")


# ---------- Contaminantes ----------

def process_single_pollutant_csv(path: Path) -> pd.Series:
    """
    Procesa un CSV de contaminante -> Serie diaria con nombre = contaminante.
    - Si existe 'PROVINCIA', filtra por 28 (Madrid).
    - Calcula media por fila (24h) ignorando NaNs y luego media entre estaciones por fecha.
    """
    pollutant = pollutant_from_filename(path)
    df = read_csv_semicolon(path)
    df.columns = [str(c).strip() for c in df.columns]

    # --- Filtro por provincia Madrid (código 28) si la columna existe ---
    if column_name in df.columns:
        df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
        df = df[df[column_name] == 28]
        if df.empty:
            raise ValueError(f"{path.name}: sin filas con PROVINCIA==28 (Madrid)")

    needed = {"ANNO", "MES", "DIA"}
    if not needed.issubset(set(df.columns)):
        raise ValueError(f"{path.name}: faltan columnas {needed - set(df.columns)}")

    # Fecha
    for c in ("ANNO", "MES", "DIA"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    df["fecha"] = pd.to_datetime(
        df[["ANNO", "MES", "DIA"]].rename(columns={"ANNO": "year", "MES": "month", "DIA": "day"}),
        errors="coerce",
    )

    # Horas
    hour_cols_present = [c for c in HOUR_COLS if c in df.columns]
    if not hour_cols_present:
        raise ValueError(f"{path.name}: no hay columnas horarias H01..H24")

    for c in hour_cols_present:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Media por fila (24h) y después media entre estaciones por fecha
    df["row_hour_mean"] = np.nanmean(df[hour_cols_present].to_numpy(dtype=float), axis=1)
    daily = df.groupby("fecha", dropna=True)["row_hour_mean"].mean().sort_index()
    daily.name = pollutant
    return daily


def consolidate_pollutants(files: list[Path]) -> pd.DataFrame:
    """
    Devuelve DF ancho con UNA columna por contaminante (fusionando múltiples años; solapes -> media).
    Índice diario continuo entre min y max fecha.
    """
    series_by_pollutant: dict[str, pd.Series] = {}
    for p in files:
        try:
            s = process_single_pollutant_csv(p)
        except Exception as e:
            print(f"[AVISO] Se omite {p.name}: {e}")
            continue
        pol = s.name
        if pol not in series_by_pollutant:
            series_by_pollutant[pol] = s
        else:
            combined = pd.concat([series_by_pollutant[pol], s], axis=1)
            merged = combined.mean(axis=1, skipna=True)
            merged.name = pol
            series_by_pollutant[pol] = merged

    if not series_by_pollutant:
        raise RuntimeError("No se pudo procesar ningún archivo CSV de contaminantes.")

    df = pd.DataFrame(series_by_pollutant).sort_index()
    start, end = df.index.min(), df.index.max()
    full_index = pd.date_range(start=start, end=end, freq="D")
    df = df.reindex(full_index)
    df.index.name = "fecha"
    return df


def impute_and_make_rolling_5d(df: pd.DataFrame, interpolate_limit: int = 3) -> pd.DataFrame:
    """
    Imputa por columna (interpolación lineal hasta 'interpolate_limit' días + media de la columna)
    y luego calcula rolling estricto de 5 días previos (t-1..t-5): shift(1) tras mean().
    """
    df_imp = df.copy()
    for col in df_imp.columns:
        df_imp[col] = df_imp[col].interpolate(method="linear", limit_direction="both", limit=interpolate_limit)
        df_imp[col] = df_imp[col].fillna(df_imp[col].mean(skipna=True))
    df_roll = df_imp.rolling(window=5, min_periods=5).mean().shift(1)  # t-1..t-5
    return df_roll


# ---------- Meteorología ----------

def read_meteo_file(path: Path) -> pd.DataFrame:
    """
    Intenta leer un Excel con:
      - Columna de fecha ('Fecha' o 'Date')
      - 'Temp Media (ºC)' y 'Humedad Media (%)' (o variantes aproximadas que contengan 'temp'/'humedad' y 'media')
    Devuelve DF diario indexado por fecha con columnas ['Temp Media (ºC)', 'Humedad Media (%)'].
    """
    try:
        xls = pd.ExcelFile(path)
        df = xls.parse(xls.sheet_names[0])
    except Exception as e:
        raise RuntimeError(f"No se pudo leer Excel meteo {path}: {e}")

    df.columns = [str(c).strip() for c in df.columns]

    # Detectar columna de fecha
    date_col = None
    for cand in df.columns:
        if cand.lower() in ("fecha", "date"):
            date_col = cand
            break
    if date_col is None:
        raise ValueError(f"{path.name}: no se encontró columna 'Fecha'")

    df["fecha"] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=["fecha"]).copy()

    # Detectar columnas de temperatura/humedad media
    temp_candidates = [c for c in df.columns if c.lower() == "temp media (ºc)".lower()]
    hum_candidates  = [c for c in df.columns if c.lower() == "humedad media (%)".lower()]

    if not temp_candidates:
        temp_candidates = [c for c in df.columns if "temp" in c.lower() and "media" in c.lower()]
    if not hum_candidates:
        hum_candidates = [c for c in df.columns if "humedad" in c.lower() and "media" in c.lower()]

    if not temp_candidates or not hum_candidates:
        raise ValueError(f"{path.name}: no se localizaron columnas de temperatura/humedad media")

    temp_col = temp_candidates[0]
    hum_col  = hum_candidates[0]

    df[temp_col] = pd.to_numeric(df[temp_col], errors="coerce")
    df[hum_col]  = pd.to_numeric(df[hum_col],  errors="coerce")

    daily = df.groupby("fecha")[[temp_col, hum_col]].mean().sort_index()
    daily.columns = ["Temp Media (ºC)", "Humedad Media (%)"]
    return daily


def consolidate_meteo(files: list[Path]) -> pd.DataFrame | None:
    """
    Consolida múltiples Excels de meteo (detec. por contenido). Devuelve DF diario reindexado continuo.
    """
    meteo_list = []
    for p in files:
        try:
            dfi = read_meteo_file(p)
            meteo_list.append(dfi)
        except Exception:
            # Si no parece meteo, lo ignoramos
            continue
    if not meteo_list:
        return None
    dfm = pd.concat(meteo_list, axis=0)
    dfm = dfm.groupby(dfm.index).mean().sort_index()
    start, end = dfm.index.min(), dfm.index.max()
    full_index = pd.date_range(start=start, end=end, freq="D")
    dfm = dfm.reindex(full_index)
    dfm.index.name = "fecha"
    return dfm


def meteo_rolling_5d(dfm: pd.DataFrame) -> pd.DataFrame:
    """Crea 'temperatura' y 'humedad' como medias de 5 días previos estrictos (shift 1)."""
    if dfm is None or dfm.empty:
        return pd.DataFrame()
    feats = pd.DataFrame(index=dfm.index)
    feats["temperatura"] = dfm["Temp Media (ºC)"].rolling(window=5, min_periods=5).mean().shift(1)
    feats["humedad"]     = dfm["Humedad Media (%)"].rolling(window=5, min_periods=5).mean().shift(1)
    return feats


# ---------- Ingresos Madrid (CA=13) ----------

def read_ingresos_file(path: Path) -> pd.Series:
    """
    Intenta leer un Excel con columnas 'Comunidad Autónoma' y una de: 'Fecha simplificada'/'Fecha'/'Date'.
    Filtra CA==13 (Madrid) y devuelve Serie diaria 'ingresos' (conteos).
    """
    try:
        xls = pd.ExcelFile(path)
        # Intento rápido: leer sólo columnas clave si existen
        try:
            df = xls.parse(xls.sheet_names[0], usecols=["Comunidad Autónoma", "Fecha simplificada"])
        except Exception:
            df = xls.parse(xls.sheet_names[0])
    except Exception as e:
        raise RuntimeError(f"No se pudo leer Excel ingresos {path}: {e}")

    df.columns = [str(c).strip() for c in df.columns]

    if "Comunidad Autónoma" not in df.columns:
        raise ValueError(f"{path.name}: no se encontró columna 'Comunidad Autónoma'")

    # Detectar columna de fecha
    date_col = None
    for cand in df.columns:
        if cand.lower().strip() in ("fecha simplificada", "fecha", "date"):
            date_col = cand
            break
    if date_col is None:
        raise ValueError(f"{path.name}: no se encontró columna de fecha ('Fecha simplificada'/'Fecha')")

    # Filtrar Madrid
    df = df[pd.to_numeric(df["Comunidad Autónoma"], errors="coerce") == 13].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col])
    if df.empty:
        raise ValueError(f"{path.name}: sin registros válidos para CA=13")

    counts = df.groupby(df[date_col].dt.normalize()).size()
    counts.index.name = "fecha"
    counts.name = "ingresos"
    return counts


def consolidate_ingresos(files: list[Path]) -> pd.Series | None:
    """
    Intenta consolidar ingresos a partir de TODOS los Excels, detectando por contenido.
    """
    series = []
    for p in files:
        try:
            s = read_ingresos_file(p)
            series.append(s)
        except Exception:
            # Si no parece dataset de ingresos, ignorar
            continue
    if not series:
        return None
    s_all = pd.concat(series, axis=0)
    s_all = s_all.groupby(s_all.index).sum().sort_index()
    return s_all


# ---------- Orquestación ----------

def run(input_dir: str, output_path: str, interpolate_limit: int = 3) -> pd.DataFrame:
    input_dir = Path(input_dir)
    output_path = Path(output_path)

    pollutant_files = sorted([p for p in input_dir.glob("*.csv")])
    excel_files     = sorted([p for p in input_dir.glob("*.xls*")])

    if not pollutant_files:
        raise RuntimeError(f"No se encontraron CSVs de contaminantes en {input_dir}")

    # Contaminantes (rolling 5d prev, estricto)
    df_poll = consolidate_pollutants(pollutant_files)
    df_poll_roll = impute_and_make_rolling_5d(df_poll, interpolate_limit=interpolate_limit)

    # Meteo (detec. por contenido)
    df_meteo = consolidate_meteo(excel_files)
    df_meteo_feats = meteo_rolling_5d(df_meteo) if df_meteo is not None else pd.DataFrame(index=df_poll_roll.index)

    # Ingresos (detec. por contenido)
    s_ing = consolidate_ingresos(excel_files)

    # Combinar
    df_final = df_poll_roll.join(df_meteo_feats, how="outer")

    if s_ing is not None:
        df_final = df_final.join(s_ing, how="left")

    # ---- Filtrado estricto: conservar SOLO días con TODO disponible ----
    # (todas las columnas de contaminantes con lag 5d, + temperatura, + humedad, + ingresos)
    required_cols = list(df_poll_roll.columns)
    if "temperatura" in df_final.columns:
        required_cols.append("temperatura")
    if "humedad" in df_final.columns:
        required_cols.append("humedad")
    if "ingresos" in df_final.columns:
        required_cols.append("ingresos")

    if required_cols:
        df_final = df_final.dropna(subset=required_cols, how="any")

    # Orden de columnas: contaminantes, meteo, ingresos al final
    cols_pollutants = [c for c in df_final.columns if c not in ("temperatura", "humedad", "ingresos")]
    ordered_cols = cols_pollutants + [c for c in ["temperatura", "humedad"] if c in df_final.columns] + (["ingresos"] if "ingresos" in df_final.columns else [])
    df_final = df_final[ordered_cols]

    # Exportar
    out_df = df_final.reset_index().rename(columns={"index": "fecha"})
    out_df.to_excel(output_path, index=False)
    print(f"Excel guardado en: {output_path}")
    return df_final


def main():
    parser = argparse.ArgumentParser(description="Contaminantes (lag 5d estricto, PROVINCIA=28) + Meteo (lag 5d estricto) + Ingresos Madrid (CA=13) -> Excel")
    parser.add_argument("--input_dir", required=True, help="Directorio con CSVs de contaminantes y Excels (meteo/ingresos).")
    parser.add_argument("--output", required=True, help="Ruta del Excel de salida (.xlsx).")
    parser.add_argument("--limit", type=int, default=3, help="Máximo tamaño de hueco para interpolar por contaminante (por defecto 3).")
    args = parser.parse_args()
    run(args.input_dir, args.output, interpolate_limit=args.limit)


if __name__ == "__main__":
    main()
