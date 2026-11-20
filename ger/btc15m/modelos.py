"""Pronósticos probabilísticos con el logit de Kalshi como punto de partida."""

from dataclasses import dataclass
from pathlib import Path
from collections.abc import Callable

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize
from scipy.special import expit, logit
from threadpoolctl import threadpool_limits
from torch import nn

from .datos import Sample
from .protocolo import EPS, SEEDS, SEQUENCE_NAMES, STATIC_NAMES

MODEL_LABELS = {
    "mercado": "Mercado",
    "mercado_calibrado": "Mercado calibrado",
    "logistica": "Logística",
    "lstm_binance": "LSTM · Binance",
    "lstm": "LSTM · Binance + BRTI",
}
RIDGE_GRID = (0.03, 0.10, 0.30)
CALIBRATION_RIDGE = 0.01
HIDDEN_SIZE = 12
MAX_EPOCHS = 40
PATIENCE = 7
BATCH_SIZE = 64
LEARNING_RATE = 0.003


@dataclass
class Standardizer:
    sequence_mean: np.ndarray
    sequence_scale: np.ndarray
    context_mean: np.ndarray
    context_scale: np.ndarray

    @classmethod
    def fit(cls, sample: Sample) -> "Standardizer":
        return cls(
            sample.sequence.mean(axis=(0, 1)),
            np.maximum(sample.sequence.std(axis=(0, 1)), 1e-8),
            sample.context.mean(axis=0),
            np.maximum(sample.context.std(axis=0), 1e-8),
        )

    def apply(self, sample: Sample) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequence = (sample.sequence - self.sequence_mean) / self.sequence_scale
        context = (sample.context - self.context_mean) / self.context_scale
        offset = logit(np.clip(sample.market, EPS, 1 - EPS))
        return sequence, context, offset


class ResidualLSTM(nn.Module):
    """Último estado recurrente más contexto, con salida en escala logit."""

    def __init__(self, context_size: int = len(STATIC_NAMES)) -> None:
        super().__init__()
        self.lstm = nn.LSTM(len(SEQUENCE_NAMES), HIDDEN_SIZE, batch_first=True)
        self.head = nn.Linear(HIDDEN_SIZE + context_size, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(
        self, sequence: torch.Tensor, context: torch.Tensor, market_logit: torch.Tensor,
    ) -> torch.Tensor:
        outputs, _ = self.lstm(sequence)
        correction = self.head(torch.cat((outputs[:, -1], context), dim=1)).squeeze(1)
        return market_logit + correction


def _design(sequence: np.ndarray, context: np.ndarray) -> np.ndarray:
    return np.column_stack((context, sequence.reshape(len(sequence), -1)))


def _fit_logistic(features: np.ndarray, y: np.ndarray, offset: np.ndarray, penalty: float) -> np.ndarray:
    design = np.column_stack((np.ones(len(y)), features))

    def objective(weights: np.ndarray) -> tuple[float, np.ndarray]:
        logits = offset + design @ weights
        loss = (np.logaddexp(0, logits) - y * logits).mean()
        ridge = penalty * weights[1:]
        gradient = design.T @ (expit(logits) - y) / len(y)
        gradient[1:] += ridge
        return float(loss + 0.5 * np.dot(ridge, weights[1:])), gradient

    result = minimize(objective, np.zeros(design.shape[1]), method="L-BFGS-B", jac=True,
                      options={"maxiter": 500, "ftol": 1e-11})
    if not result.success:
        raise RuntimeError(f"La regresión logística no convergió: {result.message}")
    return result.x


def _as_tensors(values: tuple[np.ndarray, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(torch.as_tensor(value, dtype=torch.float32) for value in values)


def fit_network(
    constructor: Callable[[], nn.Module], train: tuple[np.ndarray, ...], val: tuple[np.ndarray, ...],
    seed: int, *, max_epochs: int = MAX_EPOCHS, patience: int = PATIENCE, batch_size: int = BATCH_SIZE,
) -> tuple[nn.Module, list[dict], int]:
    torch.manual_seed(seed)
    shuffle = torch.Generator().manual_seed(seed)
    model = constructor()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()
    *x_train, y_train = _as_tensors(train)
    *x_val, y_val = _as_tensors(val)
    best_state = None
    best_val = float("inf")
    best_epoch = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch in torch.randperm(len(y_train), generator=shuffle).split(batch_size):
            optimizer.zero_grad()
            logits = model(*(x[batch] for x in x_train))
            loss_fn(logits, y_train[batch]).backward()
            optimizer.step()

        model.eval()
        with torch.inference_mode():
            train_loss = loss_fn(model(*x_train), y_train).item()
            val_loss = loss_fn(model(*x_val), y_val).item()
        history.append({"semilla": seed, "epoca": epoch, "train_log_loss": train_loss, "val_log_loss": val_loss})
        if val_loss < best_val - 1e-5:
            best_val, best_epoch = val_loss, epoch
            best_state = {name: weight.detach().clone() for name, weight in model.state_dict().items()}
        if epoch - best_epoch >= patience:
            break

    if best_state is None:
        raise RuntimeError(f"Entrenamiento sin pérdida finita para semilla {seed}")
    model.load_state_dict(best_state)
    return model.eval(), history, best_epoch


@dataclass
class Models:
    standardizer: Standardizer
    calibrator: np.ndarray
    logistic: np.ndarray
    lstms: tuple[ResidualLSTM, ...]

    def predict(self, sample: Sample) -> pd.DataFrame:
        sequence, context, offset = self.standardizer.apply(sample)
        with torch.inference_mode(), threadpool_limits(limits=2):
            tensors = _as_tensors((sequence, context, offset))
            logits = [model(*tensors).numpy().astype(np.float64) for model in self.lstms]
            p_lstm = expit(np.stack(logits)).mean(axis=0)
            p_cal = expit(offset + self.calibrator[0] + self.calibrator[1] * context[:, 0])
            p_logistic = expit(offset + self.logistic[0] + _design(sequence, context) @ self.logistic[1:])
        return pd.DataFrame({"mercado": sample.market, "mercado_calibrado": p_cal, "logistica": p_logistic, "lstm": p_lstm})

    def save(self, path: Path) -> None:
        arrays = {
            **vars(self.standardizer),
            "calibrator": self.calibrator,
            "logistic": self.logistic,
            "seeds": np.asarray(SEEDS),
        }
        for seed, model in zip(SEEDS, self.lstms, strict=True):
            arrays.update({f"lstm_{seed}_{name}": tensor.detach().numpy() for name, tensor in model.state_dict().items()})
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "Models":
        with np.load(path, allow_pickle=False) as arrays:
            if tuple(arrays["seeds"]) != SEEDS:
                raise ValueError("Las semillas del archivo no coinciden con el protocolo")
            names = ("sequence_mean", "sequence_scale", "context_mean", "context_scale")
            standardizer = Standardizer(*(arrays[name] for name in names))
            models = []
            for seed in SEEDS:
                model = ResidualLSTM(len(standardizer.context_mean))
                state = {name: torch.from_numpy(arrays[f"lstm_{seed}_{name}"]) for name in model.state_dict()}
                model.load_state_dict(state)
                models.append(model.eval())
            return cls(standardizer, arrays["calibrator"], arrays["logistic"], tuple(models))


def fit_models(train: Sample, val: Sample) -> tuple[Models, pd.DataFrame, dict]:
    """Ajusta sobre train y elige L2 y épocas con validación cronológica."""
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    scaler = Standardizer.fit(train)
    train_seq, train_ctx, train_offset = scaler.apply(train)
    val_seq, val_ctx, val_offset = scaler.apply(val)
    train_x, val_x = _design(train_seq, train_ctx), _design(val_seq, val_ctx)

    with threadpool_limits(limits=2):
        calibrator = _fit_logistic(train_ctx[:, :1], train.y, train_offset, CALIBRATION_RIDGE)
        candidates = []
        for strength in RIDGE_GRID:
            weights = _fit_logistic(train_x, train.y, train_offset, strength)
            logits = val_offset + weights[0] + val_x @ weights[1:]
            loss = float((np.logaddexp(0, logits) - val.y * logits).mean())
            candidates.append((loss, strength, weights))
        _, best_strength, logistic = min(candidates, key=lambda result: result[0])

        lstms, history, lstm_choices = [], [], []
        for seed in SEEDS:
            model, epochs, best_epoch = fit_network(
                lambda: ResidualLSTM(train_ctx.shape[1]),
                (train_seq, train_ctx, train_offset, train.y), (val_seq, val_ctx, val_offset, val.y), seed,
            )
            lstms.append(model)
            history.extend(epochs)
            lstm_choices.append({"semilla": seed, "epoca_elegida": best_epoch,
                                 "val_log_loss": epochs[best_epoch - 1]["val_log_loss"]})

    selection = {
        "l2_calibracion": CALIBRATION_RIDGE,
        "l2_logistica_elegida": best_strength,
        "l2_logistica_candidatas": [{"l2": strength, "val_log_loss": loss} for loss, strength, _ in candidates],
        "lstm": lstm_choices,
        "parametros_por_lstm": sum(parameter.numel() for parameter in lstms[0].parameters()),
    }
    return Models(scaler, calibrator, logistic, tuple(lstms)), pd.DataFrame(history), selection
