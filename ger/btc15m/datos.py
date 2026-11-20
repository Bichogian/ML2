"""Ventanas causales, controles de integridad y particiones cronológicas."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import logit

from .exportar import sha256
from .protocolo import BAR_LATENCY, EPS, HORIZON_MINUTES, MAX_SPREAD, MINUTE, PERIODS, SECOND, WINDOW_MINUTES, utc_ns


@dataclass
class Sample:
    frame: pd.DataFrame
    sequence: np.ndarray
    context: np.ndarray

    @property
    def y(self) -> np.ndarray:
        return self.frame["yes"].to_numpy(dtype=np.int64)

    @property
    def market(self) -> np.ndarray:
        return self.frame["p_market"].to_numpy(dtype=np.float64)


@dataclass
class StudyData:
    splits: dict[str, Sample]
    observations: pd.DataFrame
    bars: pd.DataFrame

    def coverage(self) -> pd.DataFrame:
        rows = []
        for period in PERIODS:
            rows_in_period = self.observations[self.observations["split"].eq(period.key)]
            selected = self.splits[period.key].frame
            rows.append({
                "etapa": period.label,
                "desde_utc": period.start,
                "hasta_exclusivo_utc": period.stop,
                "vencimientos_teoricos": 96 * len(period.days()),
                "candidatos": len(rows_in_period),
                "usados": len(selected),
                "dias_con_datos": selected["day"].nunique(),
                "dias_calendario": len(period.days()),
            })
        return pd.DataFrame(rows)


def load_data(path: Path) -> StudyData:
    receipt = json.loads((path / "procedencia.json").read_text(encoding="utf-8"))
    expected = {"kalshi_btc15m.parquet", "binance_btcusdt_1m.parquet"}
    observed = {entry["archivo"] for entry in receipt["artefactos"]}
    if observed != expected:
        raise ValueError("El manifiesto debe cubrir ambos archivos de datos")
    for entry in receipt["artefactos"]:
        if sha256(path / entry["archivo"]) != entry["sha256"]:
            raise ValueError(f"Huella de datos inesperada en {entry['archivo']}")
    return prepare(pd.read_parquet(path / "kalshi_btc15m.parquet"), pd.read_parquet(path / "binance_btcusdt_1m.parquet"))


def prepare(markets: pd.DataFrame, bars: pd.DataFrame, periods=PERIODS) -> StudyData:
    """Por fila, usa 30 velas cerradas y el cierre previo necesario para el primer retorno."""
    frame = markets.sort_values("expiry_ns", ignore_index=True).copy()
    bars = bars.sort_values("ts_close", ignore_index=True).copy()
    expiry = frame["expiry_ns"].to_numpy(dtype=np.int64)
    opening = frame["open_ns"].to_numpy(dtype=np.int64)
    decision = frame["decision_ns"].to_numpy(dtype=np.int64)
    quote = frame["quote_ns"].to_numpy(dtype=np.int64)
    cutoff = frame["info_ns"].to_numpy(dtype=np.int64)
    if (len(frame) == 0 or frame["ticker"].duplicated().any() or len(np.unique(expiry)) != len(expiry)
            or np.any(expiry % (15 * MINUTE) != 0) or np.any(expiry - opening != 15 * MINUTE)
            or np.any(expiry - decision != HORIZON_MINUTES * MINUTE)
            or np.any(quote > decision) or np.any(cutoff != np.minimum(quote, decision - SECOND))):
        raise ValueError("Identidad, horizonte o causalidad inválidos en el panel Kalshi")

    closes = bars["ts_close"].to_numpy(dtype=np.int64)
    ohlc = bars[["open", "high", "low", "close"]].to_numpy(dtype=np.float64)
    volume = bars["volume"].to_numpy(dtype=np.float64)
    buys = bars["taker_buy_volume"].to_numpy(dtype=np.float64)
    if (len(bars) <= WINDOW_MINUTES or np.any(np.diff(closes) <= 0) or np.any(closes % MINUTE != 0)
            or np.any(closes - bars["ts_open"].to_numpy(dtype=np.int64) != MINUTE)
            or not np.isfinite(ohlc).all() or np.any(ohlc <= 0)
            or np.any(ohlc[:, 1] < ohlc[:, 2]) or not np.isfinite(volume).all() or not np.isfinite(buys).all()
            or np.any(volume < 0) or np.any(buys < 0) or np.any(buys - volume > 1e-8)):
        raise ValueError("Velas Binance con tiempos, OHLC o volúmenes inválidos")

    available = closes + BAR_LATENCY
    last = np.searchsorted(available, cutoff, side="right") - 1
    base = last - WINDOW_MINUTES
    open_idx = np.searchsorted(closes, opening, side="right") - 1
    # Los índices seguros sólo permiten vectorizar la comprobación de filas rechazadas.
    safe_last = last.clip(0, len(bars) - 1)
    safe_base = base.clip(0, len(bars) - 1)
    safe_open = open_idx.clip(0, len(bars) - 1)
    enough_history = (
        (base >= 0)
        & (cutoff - available[safe_last] >= 0)
        & (cutoff - available[safe_last] < MINUTE)
        & (closes[safe_last] - closes[safe_base] == WINDOW_MINUTES * MINUTE)
        & (closes[safe_open] == opening)
        & (open_idx >= base)
        & (open_idx <= last)
    )

    p_market = frame["p_market"].to_numpy(dtype=np.float64)
    bid = frame["yes_bid"].to_numpy(dtype=np.float64)
    ask = frame["yes_ask"].to_numpy(dtype=np.float64)
    quoted = (
        frame["quote_kind"].eq("mid").to_numpy()
        & frame["active"].fillna(False).to_numpy(dtype=bool)
        & (bid > 0) & (ask < 1) & (bid <= ask)
        & np.isclose(p_market, (bid + ask) / 2, rtol=0, atol=1e-12)
        & (ask - bid <= MAX_SPREAD + 1e-12)
    )
    valid_label = frame["yes"].isin([0, 1]).to_numpy()
    frame["day"] = pd.to_datetime(expiry, unit="ns", utc=True).floor("D")
    frame["split"] = "fuera"
    frame["status"] = "fuera"
    frame["btc_first_close_ns"] = closes[safe_base]
    frame["btc_last_close_ns"] = closes[safe_last]

    log_close = np.log(ohlc[:, 3])
    buy_fraction = np.divide(buys, volume, out=np.full_like(volume, 0.5), where=volume > 0)
    features = np.column_stack((
        np.diff(log_close, prepend=np.nan) * 1e4,
        np.log(ohlc[:, 1] / ohlc[:, 2]) * 1e4,
        np.log1p(volume),
        buy_fraction,
    ))
    splits = {}
    for period in periods:
        start, stop = utc_ns(period.start), utc_ns(period.stop)
        inside = (expiry >= start) & (expiry < stop)
        right_feed = (frame["quote_feed"].notna() if period.feed is None else frame["quote_feed"].eq(period.feed)).to_numpy()
        left_guard = closes[safe_base] >= start
        if period.key in ("train", "val"):
            settled = frame["settled_ns"].ge(frame["expiry_ns"]) & frame["settled_ns"].lt(stop)
            valid_close = settled.to_numpy()
        else:
            valid_close = frame["settled_ns"].ge(frame["expiry_ns"]).to_numpy()
        status = np.select(
            [~right_feed, ~quoted, ~valid_label | ~valid_close, ~enough_history, ~left_guard],
            ["fuente", "cotizacion", "etiqueta_o_liquidacion", "ventana_incompleta", "purga_inicio"],
            default="usado",
        )
        frame.loc[inside, "split"] = period.key
        frame.loc[inside, "status"] = status[inside]
        used = np.flatnonzero(inside & (status == "usado"))
        indices = base[used, None] + np.arange(1, WINDOW_MINUTES + 1)
        sequence = features[indices]
        context = np.column_stack((
            logit(np.clip(p_market[used], EPS, 1 - EPS)),
            (ask[used] - bid[used]) * 100,
            (log_close[last[used]] - log_close[open_idx[used]]) * 1e4,
        ))
        if not len(used) or not np.isfinite(sequence).all() or not np.isfinite(context).all():
            raise ValueError(f"Partición sin datos válidos: {period.label}")
        splits[period.key] = Sample(frame.iloc[used].reset_index(drop=True), sequence, context)

    if frame["split"].eq("fuera").any():
        raise ValueError("La muestra contiene vencimientos fuera de las fechas autorizadas")
    return StudyData(splits, frame, bars)
