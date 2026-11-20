"""Métricas pareadas, calibración y replay paper en unidades monetarias enteras."""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .datos import Sample
from .protocolo import ADVERSE_TICKS, BOOTSTRAP_DRAWS, BOOTSTRAP_SEED, EPS, TRADE_EDGE, USD_UNITS, Period


def _resample(days: int) -> np.ndarray:
    return np.random.default_rng(BOOTSTRAP_SEED).integers(0, days, size=(BOOTSTRAP_DRAWS, days))


def binary_loss(y: np.ndarray, probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(probabilities, EPS, 1 - EPS)
    return -np.where(y == 1, np.log(p), np.log1p(-p))


def scores(sample: Sample, predictions: pd.DataFrame) -> pd.DataFrame:
    """El intervalo de Δ usa días compartidos entre todos los modelos."""
    y = sample.y[:, None]
    names = list(predictions)
    probabilities = predictions.to_numpy(dtype=np.float64)
    loss = binary_loss(y, probabilities)
    deltas = loss - loss[:, [names.index("mercado")]]
    deltas_linear = loss - loss[:, [names.index("logistica")]]
    _, day_index = np.unique(sample.frame["day"].astype("int64"), return_inverse=True)
    counts = np.bincount(day_index)
    draws = _resample(len(counts))
    totals = counts[draws].sum(axis=1)
    output = []
    for i, name in enumerate(names):
        day_delta = np.bincount(day_index, weights=deltas[:, i])
        day_delta_linear = np.bincount(day_index, weights=deltas_linear[:, i])
        low, high = np.quantile(day_delta[draws].sum(axis=1) / totals, [0.025, 0.975])
        low_linear, high_linear = np.quantile(day_delta_linear[draws].sum(axis=1) / totals, [0.025, 0.975])
        output.append({
            "modelo": name,
            "n": len(probabilities),
            "dias": len(counts),
            "log_loss": float(loss[:, i].mean()),
            "brier": float(np.square(probabilities[:, i] - y[:, 0]).mean()),
            "auc": float(roc_auc_score(y[:, 0], probabilities[:, i])),
            "delta_mercado": float(deltas[:, i].mean()),
            "ic95_mercado_inf": low,
            "ic95_mercado_sup": high,
            "delta_logistica": float(deltas_linear[:, i].mean()),
            "ic95_logistica_inf": low_linear,
            "ic95_logistica_sup": high_linear,
        })
    return pd.DataFrame(output)


def paired_log_loss(sample: Sample, predictions: pd.DataFrame, comparisons: list[tuple[str, str]]) -> pd.DataFrame:
    """Contrastes definidos por el diseño y remuestreados sobre los mismos días."""
    _, day_index = np.unique(sample.frame["day"].astype("int64"), return_inverse=True)
    counts = np.bincount(day_index)
    draws = _resample(len(counts))
    totals = counts[draws].sum(axis=1)
    loss = pd.DataFrame(binary_loss(sample.y[:, None], predictions.to_numpy()), columns=predictions.columns)
    comparisons_out = []
    for model, control in comparisons:
        diff = (loss[model] - loss[control]).to_numpy()
        per_day = np.bincount(day_index, weights=diff)
        lo, hi = np.quantile(per_day[draws].sum(axis=1) / totals, [0.025, 0.975])
        comparisons_out.append({"modelo": model, "control": control, "delta": diff.mean(),
                                "ic95_inf": lo, "ic95_sup": hi, "n": len(diff), "dias": len(counts)})
    return pd.DataFrame(comparisons_out)


def reliability(sample: Sample, predictions: pd.DataFrame) -> pd.DataFrame:
    output = []
    for name in predictions:
        p = predictions[name].to_numpy(dtype=np.float64)
        bins = np.minimum(9, np.floor(10 * p).astype(int))
        table = pd.DataFrame({"decil": bins, "p": p, "y": sample.y})
        aggregated = table.groupby("decil", as_index=False).agg(
            n=("y", "size"), predicho=("p", "mean"), observado=("y", "mean"),
        )
        output.append(aggregated.assign(modelo=name))
    return pd.concat(output, ignore_index=True)


def fee_units(price: np.ndarray) -> np.ndarray:
    """Escenario taker de un contrato, 0,07 × p × (1−p), redondeado al centavo superior."""
    cent = USD_UNITS // 100
    by_square = USD_UNITS**2
    price = np.asarray(price, dtype=np.int64)
    if np.any((price < 0) | (price > USD_UNITS)):
        raise ValueError("El precio del contrato debe estar entre cero y un dólar")
    fee_cents = (7 * price * (USD_UNITS - price) + by_square - 1) // by_square
    return fee_cents * cent


def replay(
    sample: Sample, predictions: pd.DataFrame, period: Period, *, adverse_ticks: int = ADVERSE_TICKS,
    trade_edge: float = TRADE_EDGE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Una intención a lo sumo por BTC15M, liquidada con la etiqueta oficial."""
    frame = sample.frame
    price_yes = frame["yes_ask"].to_numpy(dtype=np.float64)
    price_no = 1 - frame["yes_bid"].to_numpy(dtype=np.float64)
    prices_float = np.column_stack((price_yes, price_no)) * USD_UNITS
    prices = np.rint(prices_float).astype(np.int64)
    if not np.allclose(prices_float, prices, rtol=0, atol=1e-7):
        raise ValueError("La resolución monetaria de las puntas excede 1/10.000 USD")
    fees = fee_units(prices)
    cost = (prices + fees) / USD_UNITS
    rows = np.arange(len(frame))
    days = period.days()
    draws = _resample(len(days))
    ledgers, daily_ledgers, summaries = [], [], []

    for name in predictions:
        p = predictions[name].to_numpy(dtype=np.float64)
        edge = np.column_stack((p, 1 - p)) - cost
        preferred = edge.argmax(axis=1)
        surplus = edge[rows, preferred]
        selected = surplus >= trade_edge - 1e-12
        wins = (preferred == 0) == (sample.y == 1)
        price = prices[rows, preferred]
        fees_paid = np.where(selected, fees[rows, preferred], 0)
        gross = np.where(selected, wins * USD_UNITS - price, 0)
        pnl = gross - fees_paid
        adverse = np.minimum(price + adverse_ticks, USD_UNITS)
        stress_pnl = np.where(selected, wins * USD_UNITS - adverse - fee_units(adverse), 0)
        if np.any(stress_pnl > pnl):
            raise AssertionError("El estrés de precio debe mantener o reducir el neto de cada intención")

        ledger = frame[["ticker", "day", "expiry_ns", "yes"]].copy()
        ledger["modelo"] = name
        ledger["p_yes"] = p
        ledger["senal"] = np.where(selected, np.where(preferred == 0, "YES", "NO"), "ABSTENCION")
        ledger["excedente_usd"] = surplus
        ledger["entrada_usd"] = np.where(selected, price / USD_UNITS, np.nan)
        ledger["comision_usd"] = fees_paid / USD_UNITS
        ledger["bruto_usd"] = gross / USD_UNITS
        ledger["neto_usd"] = pnl / USD_UNITS
        ledger["neto_estres_usd"] = stress_pnl / USD_UNITS
        ledgers.append(ledger)

        day_values = pd.DataFrame({"net_units": pnl, "stress_units": stress_pnl, "intenciones": selected.astype(int)})
        daily = day_values.groupby(frame["day"]).sum().reindex(days, fill_value=0)
        daily["neto_usd"] = daily.pop("net_units") / USD_UNITS
        daily["neto_estres_usd"] = daily.pop("stress_units") / USD_UNITS
        daily["acumulado_usd"] = daily["neto_usd"].cumsum()
        daily["acumulado_estres_usd"] = daily["neto_estres_usd"].cumsum()
        daily_ledgers.append(daily.rename_axis("day").reset_index().assign(modelo=name))

        running = np.cumsum(np.r_[0, pnl])
        daily_net = daily["neto_usd"].to_numpy()
        ci = np.quantile(daily_net[draws].sum(axis=1), [0.025, 0.975])
        count = int(selected.sum())
        summaries.append({
            "modelo": name,
            "n_oportunidades": len(frame),
            "intenciones": count,
            "intenciones_yes": int((selected & (preferred == 0)).sum()),
            "intenciones_no": int((selected & (preferred == 1)).sum()),
            "dias_con_intenciones": int(daily["intenciones"].gt(0).sum()),
            "acierto": float(wins[selected].mean()) if count else np.nan,
            "bruto_usd": float(gross.sum() / USD_UNITS),
            "comision_usd": float(fees_paid.sum() / USD_UNITS),
            "neto_usd": float(pnl.sum() / USD_UNITS),
            "neto_estres_usd": float(stress_pnl.sum() / USD_UNITS),
            "drawdown_usd": float((np.maximum.accumulate(running) - running).max() / USD_UNITS),
            "ic95_neto_inf": ci[0],
            "ic95_neto_sup": ci[1],
        })

    return pd.DataFrame(summaries), pd.concat(ledgers, ignore_index=True), pd.concat(daily_ledgers, ignore_index=True)
