"""Figuras del estudio calculadas a partir de las mismas tablas publicadas."""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .datos import StudyData
from .modelos import MODEL_LABELS
from .protocolo import PERIODS

COLORS = {"mercado": "#595f6a", "mercado_calibrado": "#a05909", "logistica": "#087b66",
          "lstm_binance": "#7466a3", "lstm": "#254d9a"}
PERIOD_COLORS = {"train": "#59758e", "val": "#c07929", "test": "#254d9a", "transfer": "#6b568c"}


def _style(axis) -> None:
    axis.set_axisbelow(True)
    axis.grid(axis="y", color="#e1e6ed", linewidth=0.8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def coverage_plot(data: StudyData) -> Figure:
    fig, axis = plt.subplots(figsize=(10.6, 3.6), layout="constrained")
    axis.axhline(96, linewidth=1, color="#747a84", linestyle="--", label="96 vencimientos por día UTC")
    for period in PERIODS:
        days = period.days()
        used = data.splits[period.key].frame.groupby("day").size().reindex(days, fill_value=0)
        axis.bar(days, used, width=0.86, color=PERIOD_COLORS[period.key], label=period.label, zorder=3)
        if period.key != "train":
            axis.axvline(days[0] - pd.Timedelta(hours=12), color="#a5adba", linewidth=0.85)
    axis.set(title="Cobertura utilizable por día", ylabel="Vencimientos", xlabel="Fecha UTC", ylim=(0, 108))
    axis.xaxis.set_major_locator(mdates.DayLocator(interval=7))
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    axis.legend(ncol=3, fontsize=8.2, frameon=False, loc="upper left", bbox_to_anchor=(0, -0.16))
    _style(axis)
    return fig


def training_plot(history: pd.DataFrame, selection: dict, baseline_loss: float) -> Figure:
    fig, (train_axis, val_axis) = plt.subplots(1, 2, figsize=(10.6, 3.7), layout="constrained")
    palette = ("#254d9a", "#087b66", "#a05909")
    for choice, color in zip(selection["lstm"], palette, strict=True):
        seed = choice["semilla"]
        rows = history[history["semilla"].eq(seed)]
        train_axis.plot(rows["epoca"], rows["train_log_loss"], color=color, label=f"Semilla {seed}")
        val_axis.plot(rows["epoca"], rows["val_log_loss"], color=color)
        val_axis.scatter(choice["epoca_elegida"], choice["val_log_loss"], s=45,
                         edgecolor="white", linewidth=0.7, color=color, zorder=4)
    val_axis.axhline(baseline_loss, color=COLORS["mercado"], linestyle="--", linewidth=1, label="Mercado en validación")
    train_axis.set_title("Ajuste sobre entrenamiento")
    val_axis.set_title("Validación y épocas elegidas")
    for axis in (train_axis, val_axis):
        axis.set(xlabel="Época", ylabel="Log loss [nats]")
        axis.legend(frameon=False, fontsize=8)
        _style(axis)
    return fig


def probability_plot(score_table: pd.DataFrame, reliability_table: pd.DataFrame) -> Figure:
    fig, (calibration, differences) = plt.subplots(1, 2, figsize=(10.6, 4.1), layout="constrained")
    calibration.plot([0, 1], [0, 1], color="#9cabb8", linewidth=1, linestyle="--", label="Calibración ideal")
    for name, label in MODEL_LABELS.items():
        rows = reliability_table[reliability_table["modelo"].eq(name)]
        color = COLORS[name]
        calibration.plot(rows["predicho"], rows["observado"], color=color, linewidth=1, label=label)
        calibration.scatter(rows["predicho"], rows["observado"], s=10 + 0.14 * rows["n"], color=color, zorder=3)
    calibration.set(xlim=(-0.02, 1.02), ylim=(-0.02, 1.02), xlabel="Probabilidad pronosticada",
                    ylabel="Frecuencia observada de YES", title="Calibración en prueba")
    calibration.legend(loc="upper left", fontsize=7.5, frameon=False)

    comparison = score_table.loc[score_table["modelo"].ne("mercado")].reset_index(drop=True)
    differences.axvline(0, color="#747a84", linestyle="--", linewidth=1)
    for row in comparison.itertuples():
        differences.hlines(row.Index, row.ic95_mercado_inf, row.ic95_mercado_sup, color=COLORS[row.modelo], linewidth=2)
        differences.scatter(row.delta_mercado, row.Index, color=COLORS[row.modelo], s=40, zorder=3)
    differences.set_yticks(np.arange(len(comparison)), [MODEL_LABELS[name] for name in comparison["modelo"]])
    differences.set(xlabel="Δ log loss frente al mercado [nats]", title="Diferencia pareada e IC 95 %\nPor debajo de cero favorece al modelo")
    differences.invert_yaxis()
    _style(calibration)
    _style(differences)
    return fig


def paper_plot(daily: pd.DataFrame) -> Figure:
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 3.8), sharey=True, layout="constrained")
    settings = (("acumulado_usd", "Compra a la punta observada"),
                ("acumulado_estres_usd", "Un centavo adicional por señal"))
    activity = daily.groupby("modelo")["intenciones"].sum()
    for axis, (column, title) in zip(axes, settings, strict=True):
        if activity.eq(0).any():
            axis.axhline(0, color=COLORS["mercado"], linestyle="--", label="Sin compras")
        for name, label in MODEL_LABELS.items():
            if activity[name] == 0:
                continue
            series = daily.loc[daily["modelo"].eq(name)].sort_values("day")
            start = series["day"].iloc[0] - pd.Timedelta(days=1)
            axis.step([start, *series["day"]], [0, *series[column]], where="post",
                      color=COLORS[name], linewidth=1.7, label=label)
        axis.set(title=title, xlabel="Fecha de vencimiento UTC")
        axis.xaxis.set_major_locator(mdates.DayLocator(interval=5))
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
        _style(axis)
    axes[0].set_ylabel("Neto paper acumulado [USD]")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=8, loc="outside lower center", ncol=4)
    return fig
