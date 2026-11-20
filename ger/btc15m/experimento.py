"""Reproduce el entrenamiento, la evaluación fuera de muestra y las cuatro figuras."""

import argparse
from importlib.metadata import version
import json
from pathlib import Path
import platform
from tempfile import TemporaryDirectory

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .datos import Sample, load_data
from .brti import load_brti, without_brti
from .evaluacion import paired_log_loss, reliability, replay, scores
from .graficos import coverage_plot, paper_plot, probability_plot, training_plot
from .modelos import Models, fit_models
from .protocolo import PERIODS


def _export_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def predict_comparison(models: Models, ablation: Models, sample: Sample) -> pd.DataFrame:
    forecast = models.predict(sample)
    forecast.insert(3, "lstm_binance", ablation.predict(without_brti(sample))["lstm"])
    return forecast


def run(data_dir: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"Elegí un directorio de salida nuevo: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    data = load_data(data_dir)
    samples = load_brti(data_dir, data)
    coverage = data.coverage()
    coverage["con_BRTI"] = [len(samples[period.key].frame) for period in PERIODS]
    print(coverage.to_string(index=False), flush=True)
    train, val, test, transfer = (samples[period.key] for period in PERIODS)
    models, history, selection = fit_models(train, val)
    ablation, _, _ = fit_models(without_brti(train), without_brti(val))
    validation = scores(val, predict_comparison(models, ablation, val))
    test_predictions = predict_comparison(models, ablation, test)
    transfer_predictions = predict_comparison(models, ablation, transfer)
    test_scores = scores(test, test_predictions)
    transfer_scores = scores(transfer, transfer_predictions)
    reliability_table = reliability(test, test_predictions)
    test_period = next(period for period in PERIODS if period.key == "test")
    paper_summary, ledger, daily = replay(test, test_predictions, test_period)

    with TemporaryDirectory(prefix=".btc15m-", dir=output.parent) as temporary:
        destination = Path(temporary) / "entrega"
        figure_dir = destination / "figuras"
        figure_dir.mkdir(parents=True)
        coverage.to_csv(destination / "cobertura.csv", index=False)
        data.observations.groupby(["split", "status"]).size().rename("filas").reset_index().to_csv(
            destination / "filtros.csv", index=False,
        )
        history.to_csv(destination / "entrenamiento.csv", index=False)
        validation.to_csv(destination / "metricas_validacion.csv", index=False)
        test_scores.to_csv(destination / "metricas_prueba.csv", index=False)
        transfer_scores.to_csv(destination / "metricas_transferencia.csv", index=False)
        reliability_table.to_csv(destination / "calibracion_prueba.csv", index=False)
        comparison = paired_log_loss(test, test_predictions, [("lstm", "lstm_binance")])
        comparison.to_csv(destination / "aporte_brti.csv", index=False)
        paper_summary.to_csv(destination / "resumen_paper.csv", index=False)
        ledger.to_csv(destination / "simulacion_por_vencimiento.csv", index=False)
        daily.to_csv(destination / "simulacion_diaria.csv", index=False)
        prediction_table = pd.concat([
            pd.concat([sample.frame, values], axis=1)
            for sample, values in ((test, test_predictions), (transfer, transfer_predictions))
        ], ignore_index=True)
        prediction_table.to_parquet(destination / "predicciones.parquet", index=False, compression="zstd")
        _export_json(destination / "seleccion_modelos.json", selection)
        packages = ("numpy", "pandas", "pyarrow", "scipy", "scikit-learn", "torch", "matplotlib", "threadpoolctl")
        _export_json(destination / "entorno.json", {
            "python": platform.python_version(), "dispositivo": "CPU", "hilos": 2,
            "paquetes": {name: version(name) for name in packages},
        })
        weights = destination / "modelos.npz"
        models.save(weights)
        ablation_weights = destination / "lstm_binance.npz"
        ablation.save(ablation_weights)
        restored = predict_comparison(Models.load(weights), Models.load(ablation_weights), test)
        np.testing.assert_allclose(restored, test_predictions, rtol=0, atol=1e-12)

        baseline = validation.set_index("modelo").loc["mercado", "log_loss"]
        figures = {
            "01_cobertura": coverage_plot(data),
            "02_entrenamiento": training_plot(history, selection, baseline),
            "03_pronosticos": probability_plot(test_scores, reliability_table),
            "04_simulacion": paper_plot(daily),
        }
        for name, figure in figures.items():
            figure.savefig(figure_dir / f"{name}.png", dpi=180)
            plt.close(figure)
        destination.rename(output)

    columns = ["modelo", "log_loss", "delta_mercado", "ic95_mercado_inf", "ic95_mercado_sup"]
    print(test_scores[columns].to_string(index=False))
    print(comparison.to_string(index=False))
    print(paper_summary[["modelo", "intenciones", "neto_usd", "neto_estres_usd"]].to_string(index=False))
    print(f"Resultados en {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datos", default=Path("data"), type=Path)
    parser.add_argument("--salida", default=Path("resultados-repro"), type=Path)
    args = parser.parse_args()
    run(args.datos.resolve(), args.salida.resolve())


if __name__ == "__main__":
    main()
