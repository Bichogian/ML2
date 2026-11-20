"""Fechas y reglas del experimento, fijadas antes del entrenamiento."""

from dataclasses import dataclass

import pandas as pd

SECOND = 10**9
MINUTE = 60 * SECOND
VENUE_SERIES = "KXBTC15M"
HORIZON_MINUTES = 5
WINDOW_MINUTES = 30
BAR_LATENCY = SECOND
MAX_SPREAD = 0.05
EPS = 1e-4

USD_UNITS = 10_000  # unidades monetarias por dólar
TRADE_EDGE = 0.02  # USD por contrato después de comisiones
ADVERSE_TICKS = 100  # un centavo en unidades de 1/10.000 USD
SEEDS = (11, 41, 71)
BOOTSTRAP_SEED = 2315
BOOTSTRAP_DRAWS = 4000

SEQUENCE_NAMES = ("retorno_1m_pb", "rango_1m_pb", "log_volumen", "fraccion_compra")
STATIC_NAMES = ("logit_mercado", "spread_centavos", "retorno_desde_apertura_pb")


@dataclass(frozen=True)
class Period:
    key: str
    label: str
    start: str
    stop: str
    feed: str | None

    def days(self) -> pd.DatetimeIndex:
        return pd.date_range(self.start, self.stop, inclusive="left", tz="UTC")


FAST_FEED = "kalshi15m_hf_1hz"
CANDLE_FEED = "pm_bars_1m_p1"
PERIODS = (
    Period("train", "Entrenamiento", "2026-06-21", "2026-07-25", FAST_FEED),
    Period("val", "Validación", "2026-07-25", "2026-08-01", FAST_FEED),
    Period("test", "Prueba principal", "2026-08-01", "2026-08-22", FAST_FEED),
    Period("transfer", "Transferencia", "2026-08-22", "2026-09-01", CANDLE_FEED),
)


def utc_ns(day: str) -> int:
    return pd.Timestamp(day, tz="UTC").value
