"""Censo local, selección de vencimientos y diagnóstico de los vencimientos BTC15M."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from .exportar import PANEL_PATH, PANEL_SHA256, sha256, verified_parquet
from .datos import StudyData
from .protocolo import BAR_LATENCY, HORIZON_MINUTES, MAX_SPREAD, MINUTE, PERIODS, SECOND, VENUE_SERIES, utc_ns

MARKETS = Path("data/lake/kalshi/pm_markets/symbol=KXBTC15M@kalshi")
MARKET_HASHES = {
    "2026-06-01": "aa55f9a2e6bf11890a03961ce765779e6951e18e6f393c8d011cf506a9fc2a35",
    "2026-07-01": "6dc0cab54f098fc24b49f0141b16f60783fca2d91e6a1ebe7e3481c0a2f31036",
    "2026-08-01": "03f3c532396acdcdb4a95eddf434a4c6e96c514ce7015a9d7aab95efd34273d4",
}
HORIZONS = (1.0, 5.0, 7.5)
FILES = {"archivo_oficial.parquet", "horizontes_panel.parquet"}
STATUS = ("usado", "excluido_por_filtro", "oficial_sin_fila", "sin_registro_local")


def _window() -> tuple[int, int]:
    return utc_ns(PERIODS[0].start), utc_ns(PERIODS[-1].stop)


def _read_dev(root: Path, relative: Path, digest: str, clock: str, columns: list[str]) -> pd.DataFrame:
    source = verified_parquet(root, relative, digest)
    i = source.schema_arrow.get_field_index(clock)
    if i < 0:
        raise ValueError(f"Falta el reloj {clock} en {relative}")
    for n in range(source.metadata.num_row_groups):
        stats = source.metadata.row_group(n).column(i).statistics
        if stats is None or not stats.has_min_max or stats.max >= _window()[1]:
            raise ValueError(f"El pie Parquet no acredita una partición íntegramente DEV en {relative}")
    return source.read(columns=columns).to_pandas()


def export(original: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Ya existe {output}")
    start, stop = _window()
    official, provenance = [], []
    market_columns = ["market_id", "series_id", "ts_open", "ts_close", "ts_expiry",
                      "result", "status", "settle_source", "comparison"]
    for month, digest in MARKET_HASHES.items():
        path = MARKETS / f"date={month}" / "part-0.parquet"
        official.append(_read_dev(original, path, digest, "ts_close", market_columns))
        provenance.append({"ruta": str(path), "sha256": digest, "columnas": market_columns})
    market = pd.concat(official, ignore_index=True)
    market = market[market["ts_close"].between(start, stop, inclusive="left")].copy()
    valid = (
        market["series_id"].eq(VENUE_SERIES) & market["ts_close"].eq(market["ts_expiry"])
        & market["ts_close"].sub(market["ts_open"]).eq(15 * MINUTE)
        & market["ts_close"].mod(15 * MINUTE).eq(0) & market["status"].eq("finalized")
        & market["settle_source"].eq("CF Benchmarks") & market["comparison"].eq(">=")
        & market["result"].isin(["yes", "no"])
    )
    if not valid.all() or market["market_id"].duplicated().any() or market["ts_close"].duplicated().any():
        raise ValueError("Hay contratos del archivo con identidad, intervalo o liquidación inesperados")
    market["yes_oficial"] = market["result"].eq("yes").astype("int8")
    market = market.rename(columns={"market_id": "ticker", "ts_open": "open_ns", "ts_close": "expiry_ns"})
    market = market[["ticker", "open_ns", "expiry_ns", "yes_oficial", "settle_source", "comparison"]]

    cols = ["contract_id", "series", "split", "ts_expiry", "ts_decision", "tau_min", "b0_feed",
            "b0_source", "active", "bid", "ask", "p_b0", "book_age_s"]
    horizon = _read_dev(original, PANEL_PATH, PANEL_SHA256, "ts_expiry", cols)
    if not horizon["split"].eq("DEV").all():
        raise ValueError("El diagnóstico usa un panel íntegramente DEV")
    horizon = horizon[
        horizon["series"].eq(VENUE_SERIES) & horizon["ts_expiry"].between(start, stop, inclusive="left")
        & horizon["tau_min"].isin(HORIZONS)
    ].copy()
    if (horizon.duplicated(["contract_id", "tau_min"]).any()
            or not horizon["ts_expiry"].sub(horizon["ts_decision"]).eq(horizon["tau_min"] * MINUTE).all()):
        raise ValueError("Hay decisiones duplicadas o desalineadas en el panel de horizontes")
    horizon = horizon.drop(columns=["series", "split"]).rename(
        columns={"contract_id": "ticker", "ts_expiry": "expiry_ns", "ts_decision": "decision_ns"},
    )
    provenance.append({"ruta": str(PANEL_PATH), "sha256": PANEL_SHA256, "columnas": cols})

    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".auditoria-", dir=output.parent) as directory:
        pending = Path(directory) / "auditoria"
        pending.mkdir()
        market.sort_values("expiry_ns").to_parquet(pending / "archivo_oficial.parquet", index=False, compression="zstd")
        horizon.sort_values(["expiry_ns", "tau_min"]).to_parquet(pending / "horizontes_panel.parquet",
                                                               index=False, compression="zstd")
        metadata = {
            "ventana_utc": [PERIODS[0].start, PERIODS[-1].stop], "fuentes": provenance,
            "uso": "Auditoría descriptiva del archivo Kalshi y de los horarios del panel. Las etiquetas del archivo sólo "
                   "se usan para cotejar resultados y medir la composición de las filas usadas y ausentes.",
            "artefactos": {path.name: sha256(path) for path in sorted(pending.iterdir())},
        }
        (pending / "procedencia.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        pending.rename(output)
    print(f"Auditoría exportada con {len(market):,} contratos oficiales y {len(horizon):,} filas de horizontes")


def load_audit(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = json.loads((path / "procedencia.json").read_text(encoding="utf-8"))
    if set(metadata["artefactos"]) != FILES:
        raise ValueError("El manifiesto no cubre los dos recortes de auditoría")
    for name, digest in metadata["artefactos"].items():
        if sha256(path / name) != digest:
            raise ValueError(f"Huella de auditoría inesperada en {name}")
    return pd.read_parquet(path / "archivo_oficial.parquet"), pd.read_parquet(path / "horizontes_panel.parquet")


def population(data: StudyData, official: pd.DataFrame) -> pd.DataFrame:
    """Cruza grilla teórica, archivo oficial y candidatos sin intervenir en el ajuste."""
    start, stop = _window()
    when = np.arange(start, stop, 15 * MINUTE, dtype=np.int64)
    table = pd.DataFrame({"expiry_ns": when})
    table = table.merge(official, on="expiry_ns", how="left", validate="one_to_one")
    observed = data.observations[["expiry_ns", "ticker", "yes", "status", "btc_last_close_ns"]].rename(
        columns={"ticker": "ticker_panel", "yes": "yes_panel", "status": "filtro_panel"},
    )
    observed["btc_last_close_ns"] = observed["btc_last_close_ns"].astype("Int64")
    table = table.merge(observed, on="expiry_ns", how="left", validate="one_to_one")
    in_panel = table["ticker_panel"].notna()
    if in_panel.sum() != len(observed) or not (
        table.loc[in_panel, "ticker"].eq(table.loc[in_panel, "ticker_panel"])
        & table.loc[in_panel, "yes_oficial"].eq(table.loc[in_panel, "yes_panel"])
    ).all():
        raise ValueError("El archivo oficial y el panel tienen claves o etiquetas diferentes")
    table["estado"] = np.select(
        [table["ticker"].isna(), ~in_panel, table["filtro_panel"].ne("usado")],
        [STATUS[3], STATUS[2], STATUS[1]], default=STATUS[0],
    )
    utc = pd.to_datetime(table["expiry_ns"], unit="ns", utc=True)
    table["dia_utc"], table["hora_utc"] = utc.dt.floor("D"), utc.dt.hour
    table["etapa"] = ""
    for period in PERIODS:
        mask = table["expiry_ns"].between(utc_ns(period.start), utc_ns(period.stop), inclusive="left")
        table.loc[mask, "etapa"] = period.key

    # Aplicamos a todo el calendario el margen de publicación y el corte de T−5 menos un segundo.
    closes = data.bars["ts_close"].to_numpy(dtype=np.int64)
    price = data.bars["close"].to_numpy(dtype=np.float64)
    decision = when - HORIZON_MINUTES * MINUTE
    last = np.searchsorted(closes + BAR_LATENCY, decision - SECOND, side="right") - 1
    safe = last.clip(30)
    good = (last >= 30) & (closes[safe] - closes[safe - 30] == 30 * MINUTE)
    good &= (decision - SECOND - closes[safe] - BAR_LATENCY < MINUTE)
    realized = np.full(len(table), np.nan)
    indices = last[good, None] - np.arange(30, -1, -1)
    realized[good] = 10_000 * np.sqrt(np.square(np.diff(np.log(price[indices]), axis=1)).sum(axis=1))
    used = table["estado"].eq("usado")
    if not good[used].all() or not np.array_equal(closes[last[used]], table.loc[used, "btc_last_close_ns"].to_numpy(dtype=np.int64)):
        raise ValueError("El reloj spot para las filas usadas no coincide con el de la muestra")
    table["volatilidad_30m_pb"] = realized
    return table.drop(columns=["btc_last_close_ns"])


def selection_table(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period in PERIODS:
        stage = table[table["etapa"].eq(period.key)]
        counts = stage["estado"].value_counts().reindex(STATUS, fill_value=0)
        rows.append({"etapa": period.key, "calendario": len(stage), "archivo_oficial": int(stage["ticker"].notna().sum()),
                     "candidatos_panel": int(stage["ticker_panel"].notna().sum()), **counts.to_dict()})
    return pd.DataFrame(rows)


def compare_population(table: pd.DataFrame) -> pd.DataFrame:
    official = table[table["ticker"].notna()].copy()
    official["seleccion"] = np.where(official["estado"].eq("usado"), "usados", "otros_del_archivo")
    return official.groupby(["etapa", "seleccion"], as_index=False).agg(
        vencimientos=("ticker", "size"), dias=("dia_utc", "nunique"), frecuencia_yes=("yes_oficial", "mean"),
        con_volatilidad=("volatilidad_30m_pb", "count"), volatilidad_mediana_pb=("volatilidad_30m_pb", "median"),
        volatilidad_p10_pb=("volatilidad_30m_pb", lambda r: r.quantile(.10)),
        volatilidad_p90_pb=("volatilidad_30m_pb", lambda r: r.quantile(.90)),
    )


def compare_horizons(rows: pd.DataFrame, table: pd.DataFrame) -> pd.DataFrame:
    keys = ["ticker", "expiry_ns"]
    eligible_archive = table.loc[table["ticker"].notna(), keys]
    aligned = rows.merge(eligible_archive, on=keys, how="inner", validate="many_to_one")
    if len(aligned) != len(rows):
        raise ValueError("Hay horizontes del panel que no coinciden con el archivo oficial")
    result = []
    for stage in PERIODS:
        in_archive = table[table["etapa"].eq(stage.key) & table["ticker"].notna()]
        in_window = aligned[aligned["expiry_ns"].between(utc_ns(stage.start), utc_ns(stage.stop), inclusive="left")]
        for horizon in HORIZONS:
            same_time = in_window[in_window["tau_min"].eq(horizon)]
            feed = same_time[same_time["b0_feed"].eq(stage.feed)]
            good = feed[
                feed["b0_source"].eq("mid") & feed["active"].fillna(False)
                & feed["bid"].gt(0) & feed["ask"].lt(1) & feed["bid"].le(feed["ask"])
                & feed["ask"].sub(feed["bid"]).le(MAX_SPREAD + 1e-12)
                & np.isclose(feed["p_b0"], (feed["bid"] + feed["ask"]) / 2, rtol=0, atol=1e-12)
            ]
            result.append({
                "etapa": stage.key, "minutos_antes": horizon, "filas_panel": len(same_time),
                "filas_fuente_etapa": len(feed), "dos_puntas_elegibles": len(good), "archivo_oficial": len(in_archive),
                "edad_libro_mediana_s": good["book_age_s"].median(),
                "spread_mediano_cent": (100 * good["ask"].sub(good["bid"])).median(),
            })
    return pd.DataFrame(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origen", required=True, type=Path)
    parser.add_argument("--salida", default=Path("data/auditoria"), type=Path)
    args = parser.parse_args()
    export(args.origen.resolve(), args.salida.resolve())


if __name__ == "__main__":
    main()
