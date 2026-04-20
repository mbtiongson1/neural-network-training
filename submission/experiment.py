"""
experiment.py

Standard PyTorch training script for an 8-class MLP on the PA3 dataset.

Default behavior:
- Uses the pre-exported balanced partitions in `submission/export/` if present:
  - `training_set.csv`, `training_labels.csv`
  - `validation_set.csv`, `validation_labels.csv`
- Otherwise, falls back to the raw dataset in `submission/dataset/` and performs a
  simple train/val split (optionally SMOTE if `imbalanced-learn` is available).

Outputs (by default):
- Best model checkpoint: `submission/final/pytorch_best.pt`
- Test predictions CSV: `submission/predictions/predictions_for_test_pytorch.csv`

Run:
  python submission/experiment.py --help
  python submission/experiment.py --model A
  python submission/experiment.py --model B --epochs 60
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    raise SystemExit(
        "PyTorch is required for `submission/experiment.py`.\n"
        "Install it in your environment, e.g.:\n"
        "  pip install torch\n"
        f"Original import error: {exc}"
    ) from exc


NUM_CLASSES = 8
INPUT_DIM = 2052


@dataclass(frozen=True)
class TrainConfig:
    model_name: str
    hidden_sizes: Tuple[int, ...]
    activation: str
    dropout: float
    lr: float
    momentum: float
    weight_decay: float
    batch_size: int
    epochs: int
    seed: int
    patience: int
    device: str


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _load_csv_float(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), delimiter=",", dtype=np.float32)


def _load_csv_int(path: Path) -> np.ndarray:
    return np.loadtxt(str(path), delimiter=",", dtype=np.int64)


def _standardize_fit(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std, mean, std


def _standardize_apply(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    std = np.where(std < 1e-8, 1.0, std)
    return (x - mean) / std


def _maybe_load_export_partitions(root: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    export_dir = root / "export"
    train_x_p = export_dir / "training_set.csv"
    train_y_p = export_dir / "training_labels.csv"
    val_x_p = export_dir / "validation_set.csv"
    val_y_p = export_dir / "validation_labels.csv"
    if not (train_x_p.exists() and train_y_p.exists() and val_x_p.exists() and val_y_p.exists()):
        return None

    train_x = _load_csv_float(train_x_p)
    train_y = _load_csv_int(train_y_p)
    val_x = _load_csv_float(val_x_p)
    val_y = _load_csv_int(val_y_p)
    return train_x, train_y, val_x, val_y


def _fallback_split_from_raw(root: Path, seed: int, val_size: int = 800) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    dataset_dir = root / "dataset"
    x = _load_csv_float(dataset_dir / "data.csv")
    y = _load_csv_int(dataset_dir / "data_labels.csv")

    # Labels in this project are typically 1..8; convert to 0..7 for CrossEntropyLoss.
    y = y - 1 if y.min() == 1 else y

    rng = np.random.default_rng(seed)
    idx = np.arange(x.shape[0])
    rng.shuffle(idx)
    val_idx = idx[:val_size]
    train_idx = idx[val_size:]

    train_x, train_y = x[train_idx], y[train_idx]
    val_x, val_y = x[val_idx], y[val_idx]

    # Optional: balance train split with SMOTE if available.
    try:
        from imblearn.over_sampling import SMOTE  # type: ignore

        train_x, train_y = SMOTE(random_state=seed).fit_resample(train_x, train_y)
    except Exception:
        pass

    return train_x, train_y, val_x, val_y


def _load_test_set(root: Path) -> np.ndarray:
    return _load_csv_float(root / "dataset" / "test_set.csv")


def _activation_layer(name: str) -> nn.Module:
    key = name.lower()
    if key == "tanh":
        return nn.Tanh()
    if key in {"relu", "relu6"}:
        return nn.ReLU()
    if key in {"leakyrelu", "leaky_relu", "lrelu"}:
        return nn.LeakyReLU(negative_slope=0.01)
    if key in {"gelu"}:
        return nn.GELU()
    raise ValueError(f"Unknown activation: {name}")


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_sizes: Iterable[int],
        activation: str,
        dropout: float,
    ) -> None:
        super().__init__()
        act = _activation_layer(activation)

        layers: list[nn.Module] = []
        prev = input_dim
        for hidden in hidden_sizes:
            layers.append(nn.Linear(prev, hidden))
            layers.append(act.__class__() if not isinstance(act, nn.LeakyReLU) else nn.LeakyReLU(negative_slope=act.negative_slope))
            if dropout and dropout > 0:
                layers.append(nn.Dropout(p=dropout))
            prev = hidden

        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@torch.no_grad()
def _evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)
        logits = model(xb)
        loss = loss_fn(logits, yb)
        total_loss += float(loss.item()) * xb.size(0)
        preds = logits.argmax(dim=1)
        correct += int((preds == yb).sum().item())
        total += int(xb.size(0))
    avg_loss = total_loss / max(total, 1)
    acc = correct / max(total, 1)
    return avg_loss, acc


def _train_one(cfg: TrainConfig, root: Path) -> dict:
    _set_seed(cfg.seed)

    # Data
    partitions = _maybe_load_export_partitions(root)
    if partitions is None:
        train_x, train_y, val_x, val_y = _fallback_split_from_raw(root, seed=cfg.seed)
    else:
        train_x, train_y, val_x, val_y = partitions

        # Exported labels are commonly 1..8; convert to 0..7 for CrossEntropyLoss.
        train_y = train_y - 1 if train_y.min() == 1 else train_y
        val_y = val_y - 1 if val_y.min() == 1 else val_y

    if train_x.shape[1] != INPUT_DIM:
        raise ValueError(f"Expected train features with dim {INPUT_DIM}, got {train_x.shape}")

    train_x, mean, std = _standardize_fit(train_x)
    val_x = _standardize_apply(val_x, mean, std)
    test_x = _standardize_apply(_load_test_set(root), mean, std)

    train_ds = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))

    device = torch.device(cfg.device)
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, drop_last=False)

    model = MLP(
        input_dim=INPUT_DIM,
        num_classes=NUM_CLASSES,
        hidden_sizes=cfg.hidden_sizes,
        activation=cfg.activation,
        dropout=cfg.dropout,
    ).to(device)

    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=False,
    )

    best = {"epoch": -1, "val_loss": float("inf"), "val_acc": 0.0}
    best_state = None
    history: list[dict] = []
    bad_epochs = 0

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            running += float(loss.item()) * xb.size(0)
            n_seen += int(xb.size(0))

        train_loss = running / max(n_seen, 1)
        val_loss, val_acc = _evaluate(model, val_loader, device)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": val_acc,
            }
        )

        improved = val_loss < best["val_loss"] - 1e-6
        if improved:
            best = {"epoch": epoch, "val_loss": val_loss, "val_acc": val_acc}
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if cfg.patience > 0 and bad_epochs >= cfg.patience:
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Save checkpoint
    out_dir = root / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "pytorch_best.pt"
    torch.save(
        {
            "config": cfg.__dict__,
            "mean": mean,
            "std": std,
            "model_state_dict": best_state,
            "best": best,
            "history": history,
        },
        ckpt_path,
    )

    # Predict on test set
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(test_x).to(device))
        preds = logits.argmax(dim=1).detach().cpu().numpy()

    # Convert back to 1..8 labels for compatibility with the rest of the repo.
    preds_out = preds + 1

    pred_dir = root / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / "predictions_for_test_pytorch.csv"
    with pred_path.open("w", newline="") as f:
        w = csv.writer(f)
        for p in preds_out.tolist():
            w.writerow([int(p)])

    return {
        "checkpoint": str(ckpt_path),
        "predictions": str(pred_path),
        "best": best,
        "epochs_ran": history[-1]["epoch"] if history else 0,
    }


def _preset(model_key: str) -> TrainConfig:
    key = model_key.upper()
    if key == "A":
        return TrainConfig(
            model_name="A",
            hidden_sizes=(256, 128),
            activation="tanh",
            dropout=0.0,
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0,
            batch_size=8,
            epochs=80,
            seed=50,
            patience=15,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    if key == "B":
        return TrainConfig(
            model_name="B",
            hidden_sizes=(512, 256, 128),
            activation="leakyrelu",
            dropout=0.1,
            lr=0.01,
            momentum=0.9,
            weight_decay=0.0,
            batch_size=8,
            epochs=80,
            seed=50,
            patience=15,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
    raise ValueError("`--model` must be one of: A, B")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="PyTorch MLP experiment runner (PA3).")
    p.add_argument("--model", default="A", choices=["A", "B"], help="Preset network config to run.")
    p.add_argument("--epochs", type=int, default=None, help="Override epochs.")
    p.add_argument("--batch-size", type=int, default=None, help="Override batch size.")
    p.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    p.add_argument("--momentum", type=float, default=None, help="Override SGD momentum.")
    p.add_argument("--dropout", type=float, default=None, help="Override dropout probability.")
    p.add_argument("--patience", type=int, default=None, help="Early stopping patience (epochs). 0 disables.")
    p.add_argument("--device", type=str, default=None, help="Force device: cpu or cuda.")
    p.add_argument("--seed", type=int, default=None, help="Random seed.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    root = Path(__file__).resolve().parent

    cfg = _preset(args.model)
    if args.epochs is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "epochs": args.epochs})
    if args.batch_size is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "batch_size": args.batch_size})
    if args.lr is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "lr": args.lr})
    if args.momentum is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "momentum": args.momentum})
    if args.dropout is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "dropout": args.dropout})
    if args.patience is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "patience": args.patience})
    if args.device is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "device": args.device})
    if args.seed is not None:
        cfg = TrainConfig(**{**cfg.__dict__, "seed": args.seed})

    result = _train_one(cfg, root)

    best = result["best"]
    print(
        f"Done. Model {cfg.model_name} ran {result['epochs_ran']} epoch(s). "
        f"Best val_loss={best['val_loss']:.6f}, val_acc={best['val_acc']:.4f} at epoch {best['epoch']}.\n"
        f"Checkpoint: {result['checkpoint']}\n"
        f"Predictions: {result['predictions']}"
    )


if __name__ == "__main__":
    main()
