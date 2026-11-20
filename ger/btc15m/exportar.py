"""Transfiere sólo las columnas y fechas de origen necesarias para este proyecto."""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .protocolo import HORIZON_MINUTES, MINUTE, PERIODS, VENUE_SERIES, utc_ns

PANEL_PATH = Path("fleet/c1/streams/validation/out/panel_b/v3.2/panel_b_dev_v3.2.parquet")
PANEL_SHA256 = "c5ce7a844e1446daf92d11fd406e6cb3e346667529acdfe5d2ca8aebfdbdcd04"
BARS_ROOT = Path("data/lake/binance/bars_1m/symbol=BTC-USDT-SPOT@binance")
MONTH_HASHES = {
    "2026-06-01": "3a204116cd60cb1bd4f79298836629cdbac643af0c3eec8bbf6acd0af25ff0f6",
    "2026-07-01": "8b62a6e2f042f2ae2c768bba170a51b10b35511d69251752ff0b4bc45a3ed68c",
    "2026-08-01": "be7aae505c2f99805f096e62d879f6bd75a7a692917dfc120d1b0ae65b22e81c",
}

MARKET_COLUMNS = {
    "contract_id": "ticker",
    "ts_window_start": "open_ns",
    "ts_expiry": "expiry_ns",
    "ts_decision": "decision_ns",
    "ts_b0_obs": "quote_ns",
    "ts_info": "info_ns",
    "label_settlement_ts": "settled_ns",
    "y": "yes",
    "p_b0": "p_market",
    "bid": "yes_bid",
    "ask": "yes_ask",
    "active": "active",
    "book_age_s": "book_age_s",
    "b0_source": "quote_kind",
    "b0_feed": "quote_feed",
}
BARS_COLUMNS = ("ts_open", "ts_close", "open", "high", "low", "close", "volume", "taker_buy_volume")


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def verified_parquet(root: Path, relative: Path, digest: str) -> pq.ParquetFile:
    path = root / relative
    if sha256(path) != digest:
        raise ValueError(f"Huella de origen inesperada en {relative}")
    return pq.ParquetFile(path)


def export(root: Path, output: Path) -> None:
    start = utc_ns(PERIODS[0].start)
    stop = utc_ns(PERIODS[-1].stop)
    source_panel = verified_parquet(root, PANEL_PATH, PANEL_SHA256)
    expiry_col = source_panel.schema_arrow.get_field_index("ts_expiry")
    for group in range(source_panel.metadata.num_row_groups):
        stats = source_panel.metadata.row_group(group).column(expiry_col).statistics
        if stats is None or not stats.has_min_max or stats.max >= stop:
            raise ValueError("El panel debe acreditar que todos los vencimientos pertenecen a DEV")

    input_cols = [*MARKET_COLUMNS, "series", "split", "kind", "tau_min"]
    panel = source_panel.read(columns=input_cols).to_pandas()
    if set(panel["split"]) != {"DEV"}:
        raise ValueError("La exportación requiere un panel íntegramente DEV")
    candidate = (
        panel["series"].eq(VENUE_SERIES)
        & panel["kind"].eq("tau")
        & panel["tau_min"].eq(HORIZON_MINUTES)
        & panel["ts_expiry"].ge(start)
        & panel["ts_expiry"].lt(stop)
    )
    markets = (
        panel.loc[candidate, list(MARKET_COLUMNS)]
        .rename(columns=MARKET_COLUMNS)
        .sort_values("expiry_ns", ignore_index=True)
    )
    if markets.empty or markets["ticker"].duplicated().any() or markets["expiry_ns"].duplicated().any():
        raise ValueError("Se espera una decisión por ticker y por vencimiento")

    read_sources = [{"archivo": str(PANEL_PATH), "sha256": PANEL_SHA256, "tipo": "panel Kalshi DEV v3.2"}]
    bar_parts = []
    for month, digest in MONTH_HASHES.items():
        relative = BARS_ROOT / f"date={month}" / "part-0.parquet"
        part = verified_parquet(root, relative, digest).read(columns=BARS_COLUMNS).to_pandas()
        bar_parts.append(part)
        read_sources.append({"archivo": str(relative), "sha256": digest, "tipo": "Binance spot BTCUSDT 1m"})
    bars = pd.concat(bar_parts, ignore_index=True).sort_values("ts_open")
    bars = bars.loc[bars["ts_open"].ge(start - 60 * MINUTE) & bars["ts_close"].lt(stop)].reset_index(drop=True)

    output.mkdir(parents=True, exist_ok=True)
    targets = {"kalshi_btc15m.parquet": markets, "binance_btcusdt_1m.parquet": bars}
    manifest = output / "procedencia.json"
    if manifest.exists() or any((output / name).exists() for name in targets):
        raise FileExistsError(f"Ya hay una exportación en {output}")
    artifacts = []
    for name, frame in targets.items():
        path = output / name
        frame.to_parquet(path, index=False, compression="zstd")
        artifacts.append({"archivo": name, "filas": len(frame), "bytes": path.stat().st_size, "sha256": sha256(path)})

    provenance = {
        "alcance_utc": [PERIODS[0].start, PERIODS[-1].stop],
        "alcance": "BTC15M a T−5 min en Kalshi DEV; Binance spot 1m con calentamiento de una hora",
        "atribucion": [
            {"fuente": "Kalshi API pública", "uso": "resultados oficiales y velas para transferencia", "licencia": "términos del proveedor"},
            {"fuente": "narutowindy/kalshiBTC15min", "uso": "observaciones de libro 1 Hz usadas por el panel v3.2", "licencia_registrada": "Apache-2.0"},
            {"fuente": "Binance Data Vision", "uso": "velas spot BTCUSDT 1m", "licencia": "términos del proveedor"},
        ],
        "fuentes_locales": read_sources,
        "artefactos": artifacts,
    }
    manifest.write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Exportados {len(markets):,} mercados y {len(bars):,} velas a {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origen", required=True, type=Path, help="checkout original con el panel DEV y el lago")
    parser.add_argument("--salida", default=Path("data"), type=Path)
    args = parser.parse_args()
    export(args.origen.resolve(), args.salida.resolve())


if __name__ == "__main__":
    main()
