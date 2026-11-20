"""Figuras de la evaluación cronológica."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .graficos import COLORS, _style
from .modelos import MODEL_LABELS
from .walkforward import SOURCES, Evaluation


def coverage_plot(coverage: pd.DataFrame, folds: pd.DataFrame):
    fig, (source, periods) = plt.subplots(1, 2, figsize=(11.3, 4.1), layout="constrained")
    start = np.zeros(len(coverage))
    palette = ("#9db1c6", "#254d9a", "#8866a3")
    for column, color in zip(SOURCES.values(), palette, strict=True):
        source.bar(coverage.index, coverage[column], bottom=start, label=column, color=color, zorder=3)
        start += coverage[column].to_numpy()
    source.set(title="Panel elegible por mes y fuente", ylabel="Contratos", xlabel="Mes UTC")
    source.tick_params(axis="x", labelrotation=35)
    source.legend(loc="upper left", frameon=False, fontsize=8)

    x = np.arange(len(folds))
    for offset, column, label, color in [
        (-.24, "dias_train", "Entrenamiento", "#254d9a"),
        (0, "dias_val", "Validación", "#ba7b30"),
        (.24, "dias_test", "Prueba", "#087b66"),
    ]:
        bars = periods.bar(x + offset, folds[column], width=.21, label=label, color=color, zorder=3)
        periods.bar_label(bars, fontsize=8, padding=2)
    periods.set_xticks(x, folds["mes_prueba"])
    periods.set(title="Días efectivamente observados por corte", ylabel="Días", xlabel="Mes de prueba",
                ylim=(0, folds["dias_train"].max() * 1.27))
    periods.legend(loc="upper left", frameon=False, fontsize=8)
    for axis in (source, periods):
        _style(axis)
    return fig


def comparison_plot(result: Evaluation):
    fig, (monthly, paired) = plt.subplots(1, 2, figsize=(11.3, 4.1), layout="constrained")
    for model in ("mercado_calibrado", "logistica", "lstm"):
        rows = result.metrics.query("modelo == @model")
        monthly.plot(rows["mes_prueba"], 1000 * rows["delta_mercado"], marker="o",
                     label=MODEL_LABELS[model].split(" · ")[0], color=COLORS[model])
    for axis in (monthly, paired):
        axis.axhline(0, color="#657181", linestyle="--", linewidth=1)
        _style(axis)
    monthly.set(title="Diferencia frente a Kalshi por mes", ylabel="Δ log loss [10⁻³ nats]", xlabel="Mes de prueba")
    monthly.legend(frameon=False, fontsize=8)
    contrast = result.contrasts
    x = np.arange(len(contrast))
    paired.vlines(x, 1000 * contrast["ic95_inf"], 1000 * contrast["ic95_sup"], color=COLORS["lstm"], linewidth=2)
    paired.scatter(x, 1000 * contrast["delta"], s=43, color=COLORS["lstm"], zorder=4)
    paired.set_xticks(x, ["Kalshi", "Kalshi recalibrado", "Logística"])
    paired.set(title="LSTM frente a cada control · IC 95 %", ylabel="Δ log loss [10⁻³ nats]", xlabel="Control")
    return fig
