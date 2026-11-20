"""Evaluación mensual de una LSTM BTC15M sobre el panel DEV ampliado."""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .datos import Sample, prepare
from .evaluacion import paired_log_loss, replay, scores
from .exportar import sha256
from .modelos import fit_models
from .protocolo import Period, utc_ns

HISTORY = Period("historico", "Histórico DEV", "2026-01-01", "2026-09-01", None)
TEST_MONTHS = ("2026-05-01", "2026-06-01", "2026-07-01", "2026-08-01")
SOURCES = {
    "pm_bars_1m_p3": "Velas Kalshi P3",
    "kalshi15m_hf_1hz": "Captura Kalshi 1 Hz",
    "pm_bars_1m_p1": "Velas Kalshi P1",
}


def load_history(directory: Path) -> Sample:
    manifest = json.loads((directory / "procedencia.json").read_text(encoding="utf-8"))
    names = {"kalshi_btc15m.parquet", "binance_btcusdt_1m.parquet"}
    recorded = {entry["archivo"]: entry["sha256"] for entry in manifest["artefactos"]}
    if set(recorded) != names or any(sha256(directory / name) != digest for name, digest in recorded.items()):
        raise ValueError("Los archivos históricos no concuerdan con el manifiesto")
    markets = pd.read_parquet(directory / "kalshi_btc15m.parquet")
    if not set(markets["quote_feed"].dropna()).issubset(SOURCES):
        raise ValueError("Fuente de cotización inesperada en el panel")
    return prepare(markets, pd.read_parquet(directory / "binance_btcusdt_1m.parquet"), (HISTORY,)).splits[HISTORY.key]


def slice_sample(sample: Sample, start: str, end: str, *, settled_before_end: bool = False) -> Sample:
    frame = sample.frame
    lower, upper = utc_ns(start), utc_ns(end)
    eligible = frame["expiry_ns"].between(lower, upper, inclusive="left") & frame["btc_first_close_ns"].ge(lower)
    if settled_before_end:
        eligible &= frame["settled_ns"].lt(upper)
    mask = eligible.to_numpy()
    if frame.loc[mask, "yes"].nunique() != 2:
        raise ValueError(f"Se necesitan ambos resultados en la ventana {start} a {end}")
    return Sample(frame.loc[mask].reset_index(drop=True), sample.sequence[mask], sample.context[mask])


def coverage_table(history: Sample) -> pd.DataFrame:
    frame = history.frame
    counts = frame.assign(mes=frame["day"].dt.strftime("%Y-%m")).groupby("mes").agg(
        contratos=("ticker", "size"), dias=("day", "nunique"))
    by_source = pd.crosstab(frame["day"].dt.strftime("%Y-%m"), frame["quote_feed"])
    return counts.join(by_source.reindex(columns=list(SOURCES), fill_value=0)).rename(columns=SOURCES)


def demo_history() -> Sample:
    """Datos ficticios para ejecutar el flujo sin el panel histórico."""
    rng = np.random.default_rng(1415)
    days = pd.date_range(HISTORY.start, HISTORY.stop, tz="UTC", inclusive="left")
    dates = days.repeat(4) + pd.to_timedelta(np.tile([5, 10, 15, 20], len(days)), unit="h")
    signal = rng.normal(size=len(dates))
    true_probability = 1 / (1 + np.exp(-signal))
    yes = (rng.uniform(size=len(dates)) < true_probability).astype(int)
    mid = np.rint(100 * true_probability.clip(.03, .97)) / 100
    timestamps = dates.astype("int64")
    frame = pd.DataFrame({
        "ticker": [f"DEMO-{i:04d}" for i in range(len(dates))],
        "day": dates.floor("D"), "expiry_ns": timestamps,
        "btc_first_close_ns": timestamps - 36 * 60 * 10**9, "settled_ns": timestamps + 10**9,
        "p_market": mid, "yes_bid": mid - .01, "yes_ask": mid + .01, "yes": yes,
        "quote_feed": "kalshi15m_hf_1hz",
    })
    sequence = rng.normal(size=(len(dates), 30, 4))
    context = np.column_stack((signal, np.full(len(dates), 2.), rng.normal(size=len(dates))))
    return Sample(frame, sequence, context)


@dataclass
class Evaluation:
    folds: pd.DataFrame
    selection: pd.DataFrame
    metrics: pd.DataFrame
    pooled: pd.DataFrame
    by_source: pd.DataFrame
    contrasts: pd.DataFrame
    paper: pd.DataFrame
    predictions: pd.DataFrame


def evaluate(history: Sample) -> Evaluation:
    folds, selections, metrics, test_sources, papers = [], [], [], [], []
    frames, forecasts, sequences, contexts = [], [], [], []
    for month in TEST_MONTHS:
        start = pd.Timestamp(month)
        validation = (start - pd.offsets.MonthBegin()).date().isoformat()
        end = (start + pd.offsets.MonthBegin()).date().isoformat()
        period = Period("test", month[:7], month, end, None)
        train = slice_sample(history, HISTORY.start, validation, settled_before_end=True)
        val = slice_sample(history, validation, month, settled_before_end=True)
        test = slice_sample(history, month, end)
        models, _, selected = fit_models(train, val)
        predictions = models.predict(test)
        folds.append({"mes_prueba": month[:7], "inicio_validacion": validation, "fin_test_exclusivo": end,
                      "n_train": len(train.frame), "dias_train": train.frame["day"].nunique(),
                      "n_val": len(val.frame), "dias_val": val.frame["day"].nunique(),
                      "n_test": len(test.frame), "dias_test": test.frame["day"].nunique()})
        selections.extend({**item, "mes_prueba": month[:7], "l2_logistica": selected["l2_logistica_elegida"]}
                          for item in selected["lstm"])
        metrics.append(scores(test, predictions).assign(mes_prueba=month[:7]))
        frame = test.frame.assign(mes_prueba=month[:7])
        paper, _, _ = replay(test, predictions, period)
        papers.append(paper.assign(mes_prueba=month[:7]))
        frames.append(frame)
        forecasts.append(predictions)
        sequences.append(test.sequence)
        contexts.append(test.context)

    actual = Sample(pd.concat(frames, ignore_index=True), np.concatenate(sequences), np.concatenate(contexts))
    prediction = pd.concat(forecasts, ignore_index=True)
    for source, group in actual.frame.groupby("quote_feed", sort=False):
        indices = group.index.to_numpy()
        subset = Sample(group.reset_index(drop=True), actual.sequence[indices], actual.context[indices])
        test_sources.append(scores(subset, prediction.iloc[indices].reset_index(drop=True)).assign(fuente=SOURCES[source]))
    contrasts = paired_log_loss(actual, prediction, [("lstm", "mercado"), ("lstm", "mercado_calibrado"), ("lstm", "logistica")])
    return Evaluation(pd.DataFrame(folds), pd.DataFrame(selections), pd.concat(metrics, ignore_index=True),
                      scores(actual, prediction), pd.concat(test_sources, ignore_index=True), contrasts,
                      pd.concat(papers, ignore_index=True), pd.concat([actual.frame, prediction], axis=1))
