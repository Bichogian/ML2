"""Gráficos de selección y representatividad de los datos."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .graficos import _style

BLUE = "#174a85"
SLATE = "#687484"


def _number(value: float, digits: int = 2, *, signed=False) -> str:
    if abs(value) < 0.5 * 10**(-digits):
        value = 0.0
    format_code = f"+.{digits}f" if signed and value else f".{digits}f"
    return format(value, format_code).replace(".", ",").replace("-", "−")


def seleccion_plot(census: pd.DataFrame, horizons: pd.DataFrame):
    fig, (funnel, times) = plt.subplots(1, 2, figsize=(10.8, 4.0), layout="constrained",
                                      gridspec_kw={"width_ratios": [1, 1.12]})
    stage = census.set_index("etapa").loc["test"]
    steps = ("calendario", "archivo_oficial", "candidatos_panel", "usado")
    values = stage.loc[list(steps)].to_numpy(dtype=int)
    positions = np.arange(len(steps))
    funnel.barh(positions, values, color=["#dbe4ed", "#a9bbcf", "#7090b6", BLUE], height=.61)
    for pos, n in zip(positions, values, strict=True):
        funnel.text(n + 24, pos, f"{n:,}".replace(",", "."), va="center", fontsize=9)
    funnel.set_yticks(positions, ["Calendario teórico", "Archivo oficial local", "Candidatos en el panel", "Usados"])
    funnel.invert_yaxis()
    funnel.set(xlim=(0, max(values) * 1.17), xlabel="Vencimientos", title="Selección en la prueba principal")
    _style(funnel)

    pivot = horizons[horizons["etapa"].isin(["test", "transfer"])]
    hours = sorted(pivot["minutos_antes"].unique())
    x = np.arange(len(hours))
    for key, offset, color, label in (
        ("test", -.20, BLUE, "Prueba · 1 Hz"), ("transfer", .20, "#75618a", "Transferencia · 1 min"),
    ):
        rows = pivot[pivot["etapa"].eq(key)].set_index("minutos_antes").loc[hours]
        height = 100 * rows["dos_puntas_elegibles"] / rows["archivo_oficial"]
        times.bar(x + offset, height, width=.35, color=color, label=label, zorder=3)
        for pos, rate, n in zip(x + offset, height, rows["dos_puntas_elegibles"], strict=True):
            times.text(pos, rate + 2, f"{n:,}".replace(",", "."), ha="center",
                       fontsize=8.4, color=color)
    times.set_xticks(x, [f"T−{format(h, 'g').replace('.', ',')} min" for h in hours])
    times.set(ylim=(0, 114), ylabel="Vencimientos del archivo [%]", title="Dos puntas elegibles por horizonte")
    times.legend(frameon=False, fontsize=8.2, loc="upper left")
    _style(times)
    return fig


def poblacion_plot(population: pd.DataFrame, comparison: pd.DataFrame):
    test = population[population["etapa"].eq("test") & population["ticker"].notna()].copy()
    test["usado"] = test["estado"].eq("usado")
    hourly = test.groupby("hora_utc")["usado"].mean().reindex(range(24)) * 100
    groups = comparison[comparison["etapa"].eq("test")].set_index("seleccion").loc[["usados", "otros_del_archivo"]]
    fig, (coverage, volatility) = plt.subplots(1, 2, figsize=(10.7, 3.95), layout="constrained",
                                                gridspec_kw={"width_ratios": [1.2, 1]})
    coverage.bar(hourly.index, hourly, width=.8, color="#7090b6", zorder=3)
    overall = 100 * test["usado"].mean()
    coverage.axhline(overall, color=SLATE, linestyle="--", linewidth=1.2,
                    label=f"Promedio del archivo · {_number(overall, 1)} %", zorder=4)
    coverage.set(ylim=(0, 100), ylabel="Contratos del archivo usados [%]",
                 xlabel="Hora de vencimiento UTC", title="Cobertura por hora en prueba")
    coverage.set_xticks(range(0, 24, 3))
    coverage.legend(frameon=False, fontsize=8.3, loc="upper left")
    _style(coverage)

    for row, y, color in zip(groups.itertuples(), (0, 1), (BLUE, SLATE), strict=True):
        volatility.hlines(y, row.volatilidad_p10_pb, row.volatilidad_p90_pb, color=color, linewidth=2.8)
        volatility.scatter(row.volatilidad_mediana_pb, y, color=color, s=53, zorder=4)
        volatility.annotate(_number(row.volatilidad_mediana_pb, 1), (row.volatilidad_mediana_pb, y),
                            textcoords="offset points", xytext=(0, 9), ha="center", color=color, fontsize=9)
    counts = [f"{n:,}".replace(",", ".") for n in groups["vencimientos"]]
    volatility.set_yticks([0, 1], [f"Usados\n(n={counts[0]})", f"Otros del archivo\n(n={counts[1]})"], fontsize=9)
    volatility.set_ylim(1.48, -.48)
    volatility.set(xlim=(0, groups["volatilidad_p90_pb"].max() * 1.15),
                   xlabel="Volatilidad realizada en 30 min [pb]", title="Mediana y percentiles 10–90")
    volatility.grid(axis="x", color="#e1e6ed", linewidth=.8)
    _style(volatility)
    return fig
