"""Recorte reproducible del índice de liquidación antes de la decisión."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .auditoria_datos import _read_dev
from .datos import Sample, StudyData, load_data
from .exportar import PANEL_PATH, PANEL_SHA256, sha256
from .protocolo import SECOND

TABLE = "brti_predecision.parquet"
MANIFEST = "procedencia_brti.json"
WINDOW = 60
MAX_AGE = 3 * SECOND
INDEX = Path("data/lake/spotref/brti_hf/symbol=BTC-USD-IDX-BRTI@cfbenchmarks")
HASHES = {
    "06": "9fe05638ba9cb8463e276a7ee1d518c2772511ffb75452f1b068b58612cc070f",
    "07": "50e2778df73ddac9fcddcb13b56b1c89a3c7b1612e166bc4540a56f375c1d72c",
    "08": "5d7a59de9db0dc7e10c4c164fd6691ba4c8b1fa73e6c011c5ddf8be19e76bab4",
}
KEYS = ["ticker", "decision_ns", "info_ns"]


def grids(cutoffs: np.ndarray) -> np.ndarray:
    return (cutoffs // SECOND * SECOND)[:, None] - np.arange(WINDOW - 1, -1, -1) * SECOND


def align(ticks: pd.DataFrame, times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Último tick por segundo según el mayor sello de origen y recepción más 1 s."""
    stamps = ticks[["ts_exchange", "ts_recv"]].to_numpy(dtype=np.int64).max(axis=1) + SECOND
    order = np.argsort(stamps, kind="stable")
    stamps, prices = stamps[order], ticks["price"].to_numpy()[order]
    index = np.searchsorted(stamps, times, side="right") - 1
    last = index.clip(0)
    good = (index >= 0) & (times - stamps[last] <= MAX_AGE)
    return np.where(good, prices[last], np.nan), np.where(good, stamps[last], -1)


def export(original: Path, data_dir: Path) -> None:
    if any((data_dir / name).exists() for name in (TABLE, MANIFEST)):
        raise FileExistsError("Indicá un directorio de datos sin el recorte BRTI")
    data = load_data(data_dir)
    contracts = pd.concat([s.frame[KEYS] for s in data.splits.values()], ignore_index=True)
    panel = _read_dev(original, PANEL_PATH, PANEL_SHA256, "ts_expiry",
                      ["contract_id", "ts_decision", "ts_info", "k_lo", "strike_known"])
    panel = panel.rename(columns={"contract_id": "ticker", "ts_decision": "decision_ns", "ts_info": "info_ns"})
    contracts = contracts.merge(panel, on=KEYS, how="left", validate="one_to_one")
    if not contracts["strike_known"].eq(True).all() or not contracts["k_lo"].gt(0).all():
        raise ValueError("Falta el umbral conocido en el corte para algún contrato")
    source_files = {str(PANEL_PATH): PANEL_SHA256}
    segments = []
    for month, digest in HASHES.items():
        path = INDEX / f"date=2026-{month}-01" / "part-0.parquet"
        part = _read_dev(original, path, digest, "ts_exchange", ["ts_exchange", "ts_recv", "price"])
        segments.append(part.dropna().loc[lambda r: r["price"].gt(0)])
        source_files[str(path)] = digest
    prices, availability = align(pd.concat(segments, ignore_index=True), grids(contracts["info_ns"].to_numpy()))
    contracts = contracts[KEYS + ["k_lo"]].rename(columns={"k_lo": "strike_usd"})
    contracts["brti_usd"] = list(prices)
    contracts["disponible_ns"] = list(availability)
    path = data_dir / TABLE
    contracts.to_parquet(path, index=False, compression="zstd")
    manifest = {
        "archivo": TABLE, "sha256": sha256(path), "filas": len(contracts),
        "fuente": "Grabación de BRTI de narutowindy/kalshiBTC15min, revisión 6c729b59f42e002fc8fe9dce2fe5abbda86a1d11",
        "licencia_registrada": "Apache-2.0 en el manifiesto de origen",
        "uso": "60 puntos as-of anteriores o iguales a info_ns y strike conocido. No guarda etiquetas del índice.",
        "fuentes_locales_sha256": source_files,
    }
    (data_dir / MANIFEST).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_brti(path: Path, data: StudyData) -> dict[str, Sample]:
    manifest = json.loads((path / MANIFEST).read_text(encoding="utf-8"))
    if manifest["archivo"] != TABLE or manifest["sha256"] != sha256(path / TABLE):
        raise ValueError("Huella del recorte BRTI inesperada")
    ticks = pd.read_parquet(path / TABLE)
    expected = pd.concat([s.frame[KEYS] for s in data.splits.values()], ignore_index=True)
    if not ticks[KEYS].equals(expected):
        raise ValueError("El recorte BRTI no corresponde a los mismos contratos y relojes")
    result = {}
    for stage, sample in data.splits.items():
        rows = sample.frame.merge(ticks, on=KEYS, how="left", validate="one_to_one")
        mean = np.vstack(rows.pop("brti_usd"))
        stamps = np.vstack(rows.pop("disponible_ns"))
        step = grids(rows["info_ns"].to_numpy())
        if mean.shape != step.shape or stamps.shape != step.shape:
            raise ValueError("La ventana BRTI debe tener 60 pasos")
        observed = stamps >= 0
        if (np.any(observed & ((stamps > step) | (step - stamps > MAX_AGE)))
                or np.any(observed != (np.isfinite(mean) & (mean > 0)))
                or not rows["strike_usd"].gt(0).all()):
            raise ValueError("Reloj, precio o umbral inválidos en el recorte BRTI")
        valid = observed.all(axis=1)
        if not valid.any():
            raise ValueError(f"Falta cobertura de BRTI en {stage}")
        selected = rows.loc[valid].reset_index(drop=True)
        selected["brti_media60_usd"] = mean[valid].mean(axis=1)
        selected["brti_ultima_disponibilidad_ns"] = stamps[valid, -1]
        distance = 10_000 * np.log(selected["brti_media60_usd"] / selected["strike_usd"])
        result[stage] = Sample(selected, sample.sequence[valid], np.column_stack((sample.context[valid], distance)))
    return result


def without_brti(sample: Sample) -> Sample:
    return Sample(sample.frame, sample.sequence, sample.context[:, :-1])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origen", required=True, type=Path)
    parser.add_argument("--datos", default=Path("data"), type=Path)
    args = parser.parse_args()
    export(args.origen, args.datos)
