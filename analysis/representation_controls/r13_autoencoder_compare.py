#!/usr/bin/env python3
"""
R1.3 autoencoder-vs-PCA comparison for Evoscope.

This script trains the convolutional autoencoder on morphology snapshots from
training seeds, validates on validation seeds, and evaluates on held-out seeds.
It reports two autoencoder-based comparisons:

1. autoencoder_prediction_head:
   Direct prediction of global regulatory variables from the supervised head.

2. autoencoder_z8_ridge:
   Ridge regression from the learned 8-dimensional latent vector z to the same
   global regulatory targets. This is directly comparable to the PCA8 + ridge
   baseline used in r13_baseline_controls.py.

Expected input layout:

runs/
  seed_38/
    global_genes.csv
    snapshots/
      grid_001.npy
      ...
      grid_120.npy
  seed_40/
    ...

Output:

r13_outputs/
  r13_autoencoder_comparison.csv
  r13_autoencoder_predictions.csv
  r13_autoencoder_training_history.csv

Usage:

python r13_autoencoder_compare.py --root runs --outdir outputs --device auto  

"""

from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GENES = ["T1", "T2", "I", "R", "M", "K", "S"]

DEFAULT_TRAIN_SEEDS = [38, 40, 53, 65, 89, 90]
DEFAULT_VAL_SEEDS = [96, 101]
DEFAULT_TEST_SEEDS = [104, 107]


def parse_seed_from_dir(path: Path) -> int:
    match = re.search(r"seed_(\d+)$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse seed from directory name: {path}")
    return int(match.group(1))


def snapshot_step_from_name(path: Path) -> int:
    match = re.search(r"grid_(\d+)\.npy$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse snapshot step from filename: {path}")
    return int(match.group(1))


def discover_run_dirs(root: Path) -> List[Path]:
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not run_dirs:
        raise FileNotFoundError(f"No seed_* directories found under {root}")
    return run_dirs


def pearson_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correlations = []

    for j in range(y_true.shape[1]):
        a = y_true[:, j]
        b = y_pred[:, j]

        if np.std(a) == 0 or np.std(b) == 0:
            continue

        correlations.append(np.corrcoef(a, b)[0, 1])

    if not correlations:
        return np.nan

    return float(np.mean(correlations))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "Pearson_mean": pearson_per_target(y_true, y_pred),
    }


def load_morphology_dataset(root: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load morphology snapshots and aligned global regulatory targets.

    Evoscope exports snapshots as grid_001.npy ... grid_120.npy, while
    global_genes.csv stores steps 0 ... 119. We align:

        grid_001.npy -> step 0
        grid_002.npy -> step 1
        ...
        grid_120.npy -> step 119
    """

    grids = []
    targets = []
    meta_rows = []

    for run_dir in discover_run_dirs(root):
        seed = parse_seed_from_dir(run_dir)

        global_path = run_dir / "global_genes.csv"
        snapshots_dir = run_dir / "snapshots"

        if not global_path.exists():
            raise FileNotFoundError(global_path)
        if not snapshots_dir.exists():
            raise FileNotFoundError(snapshots_dir)

        global_df = pd.read_csv(global_path)
        global_by_step = global_df.set_index("step")

        snapshot_files = sorted(snapshots_dir.glob("grid_*.npy"))
        if not snapshot_files:
            raise FileNotFoundError(f"No grid_*.npy snapshots found in {snapshots_dir}")

        for snapshot_path in snapshot_files:
            exported_snapshot_step = snapshot_step_from_name(snapshot_path)
            csv_step = exported_snapshot_step - 1

            if csv_step not in global_by_step.index:
                continue

            grid = np.load(snapshot_path).astype(np.int64)

            # Evoscope grid encoding is -1 for empty and 0..8 for occupied classes.
            # Shift to 0..9 for cross-entropy and one-hot encoding.
            grid = grid + 1

            if grid.min() < 0 or grid.max() > 9:
                raise ValueError(
                    f"Unexpected grid class range in {snapshot_path}: "
                    f"min={grid.min()}, max={grid.max()}"
                )

            target = global_by_step.loc[csv_step, GENES].to_numpy(dtype=np.float32)

            grids.append(grid)
            targets.append(target)
            meta_rows.append(
                {
                    "seed": seed,
                    "step": int(csv_step),
                    "snapshot_file": str(snapshot_path),
                }
            )

    X = np.stack(grids).astype(np.int64)
    y = np.stack(targets).astype(np.float32)
    meta = pd.DataFrame(meta_rows)

    return X, y, meta


class ConvAutoencoderRegressor(nn.Module):
    def __init__(
        self,
        n_classes: int = 10,
        height: int = 40,
        width: int = 60,
        latent_dim: int = 8,
        target_dim: int = 7,
    ):
        super().__init__()

        self.n_classes = n_classes
        self.height = height
        self.width = width

        self.encoder = nn.Sequential(
            nn.Conv2d(n_classes, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_classes, height, width)
            encoded = self.encoder(dummy)

        self.encoded_shape = encoded.shape[1:]
        self.encoded_dim = int(np.prod(self.encoded_shape))

        self.to_latent = nn.Linear(self.encoded_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, self.encoded_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, n_classes, kernel_size=4, stride=2, padding=1),
        )

        self.prediction_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, target_dim),
        )

    def forward(self, x: torch.Tensor):
        x_onehot = F.one_hot(x, num_classes=self.n_classes)
        x_onehot = x_onehot.permute(0, 3, 1, 2).float()

        encoded = self.encoder(x_onehot)
        encoded_flat = encoded.reshape(encoded.shape[0], -1)

        z = self.to_latent(encoded_flat)

        decoded_flat = self.from_latent(z)
        decoded = decoded_flat.reshape(z.shape[0], *self.encoded_shape)

        reconstruction_logits = self.decoder(decoded)
        reconstruction_logits = reconstruction_logits[:, :, : self.height, : self.width]

        prediction = self.prediction_head(z)

        return reconstruction_logits, prediction, z


def make_loader(
    X: np.ndarray,
    y_scaled: np.ndarray,
    meta: pd.DataFrame,
    seeds: List[int],
    batch_size: int,
    shuffle: bool,
):
    mask = meta["seed"].isin(seeds).to_numpy()
    dataset = TensorDataset(
        torch.tensor(X[mask], dtype=torch.long),
        torch.tensor(y_scaled[mask], dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle), mask


def train_autoencoder(
    X: np.ndarray,
    y: np.ndarray,
    meta: pd.DataFrame,
    train_seeds: List[int],
    val_seeds: List[int],
    latent_dim: int,
    max_epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    lambda_reconstruction: float,
    lambda_prediction: float,
    device: str,
):
    train_mask = meta["seed"].isin(train_seeds).to_numpy()

    y_mean = y[train_mask].mean(axis=0, keepdims=True)
    y_std = y[train_mask].std(axis=0, keepdims=True) + 1e-8
    y_scaled = (y - y_mean) / y_std

    height, width = X.shape[1], X.shape[2]

    model = ConvAutoencoderRegressor(
        n_classes=10,
        height=height,
        width=width,
        latent_dim=latent_dim,
        target_dim=len(GENES),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    ce_loss = nn.CrossEntropyLoss()
    mse_loss = nn.MSELoss()

    train_loader, _ = make_loader(
        X, y_scaled, meta, train_seeds, batch_size=batch_size, shuffle=True
    )
    val_loader, _ = make_loader(
        X, y_scaled, meta, val_seeds, batch_size=batch_size, shuffle=False
    )

    best_val_loss = np.inf
    best_state = None
    epochs_without_improvement = 0
    history_rows = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_n = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()

            reconstruction_logits, prediction, _ = model(xb)

            loss_rec = ce_loss(reconstruction_logits, xb)
            loss_pred = mse_loss(prediction, yb)

            loss = lambda_reconstruction * loss_rec + lambda_prediction * loss_pred
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * xb.shape[0]
            train_n += xb.shape[0]

        train_loss = train_loss_sum / max(train_n, 1)

        model.eval()
        val_loss_sum = 0.0
        val_rec_sum = 0.0
        val_pred_sum = 0.0
        val_n = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                reconstruction_logits, prediction, _ = model(xb)

                loss_rec = ce_loss(reconstruction_logits, xb)
                loss_pred = mse_loss(prediction, yb)
                loss = lambda_reconstruction * loss_rec + lambda_prediction * loss_pred

                val_loss_sum += float(loss.item()) * xb.shape[0]
                val_rec_sum += float(loss_rec.item()) * xb.shape[0]
                val_pred_sum += float(loss_pred.item()) * xb.shape[0]
                val_n += xb.shape[0]

        val_loss = val_loss_sum / max(val_n, 1)
        val_rec = val_rec_sum / max(val_n, 1)
        val_pred = val_pred_sum / max(val_n, 1)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_reconstruction_ce": val_rec,
                "val_prediction_mse_scaled": val_pred,
            }
        )

        if epoch == 1 or epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} "
                f"train_loss={train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_ce={val_rec:.4f} "
                f"val_mse_scaled={val_pred:.4f}"
            )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = pd.DataFrame(history_rows)

    return model, y_mean, y_std, history


def predict_with_model(
    model: nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    loader = DataLoader(
        torch.tensor(X, dtype=torch.long),
        batch_size=batch_size,
        shuffle=False,
    )

    preds = []
    latents = []

    model.eval()
    with torch.no_grad():
        for xb in loader:
            xb = xb.to(device)
            _, prediction, z = model(xb)
            preds.append(prediction.cpu().numpy())
            latents.append(z.cpu().numpy())

    return np.vstack(preds), np.vstack(latents)


def fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R1.3 autoencoder comparison.")
    parser.add_argument("--root", type=str, default="runs")
    parser.add_argument("--outdir", type=str, default="r13_outputs")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--val-seeds", type=int, nargs="+", default=DEFAULT_VAL_SEEDS)
    parser.add_argument("--test-seeds", type=int, nargs="+", default=DEFAULT_TEST_SEEDS)
    parser.add_argument("--latent-dim", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lambda-reconstruction", type=float, default=1.0)
    parser.add_argument("--lambda-prediction", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading morphology dataset...")
    X, y, meta = load_morphology_dataset(root)

    print(f"Loaded X shape: {X.shape}")
    print(f"Loaded y shape: {y.shape}")
    print("Samples by seed:")
    print(meta["seed"].value_counts().sort_index())

    model, y_mean, y_std, history = train_autoencoder(
        X=X,
        y=y,
        meta=meta,
        train_seeds=args.train_seeds,
        val_seeds=args.val_seeds,
        latent_dim=args.latent_dim,
        max_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        patience=args.patience,
        lambda_reconstruction=args.lambda_reconstruction,
        lambda_prediction=args.lambda_prediction,
        device=device,
    )

    history.to_csv(outdir / "r13_autoencoder_training_history.csv", index=False)

    train_mask = meta["seed"].isin(args.train_seeds).to_numpy()
    test_mask = meta["seed"].isin(args.test_seeds).to_numpy()

    pred_scaled_all, z_all = predict_with_model(
        model=model,
        X=X,
        batch_size=args.batch_size,
        device=device,
    )

    pred_all = pred_scaled_all * y_std + y_mean

    y_train = y[train_mask]
    y_test = y[test_mask]

    z_train = z_all[train_mask]
    z_test = z_all[test_mask]

    pred_head_test = pred_all[test_mask]

    pred_z_ridge_test = fit_predict_ridge(
        X_train=z_train,
        y_train=y_train,
        X_test=z_test,
        alpha=1.0,
    )

    metric_rows = []

    head_metrics = regression_metrics(y_test, pred_head_test)
    metric_rows.append(
        {
            "model": "autoencoder_prediction_head",
            "input": "morphology_autoencoder",
            "target": "global_regulatory_variables",
            "split": "held_out_seed",
            **head_metrics,
        }
    )

    z_ridge_metrics = regression_metrics(y_test, pred_z_ridge_test)
    metric_rows.append(
        {
            "model": "autoencoder_z8_ridge",
            "input": "autoencoder_latent_z8",
            "target": "global_regulatory_variables",
            "split": "held_out_seed",
            **z_ridge_metrics,
        }
    )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(outdir / "r13_autoencoder_comparison.csv", index=False)

    pred_rows = meta.loc[test_mask, ["seed", "step", "snapshot_file"]].copy()

    for j, gene in enumerate(GENES):
        pred_rows[f"{gene}_true"] = y_test[:, j]
        pred_rows[f"{gene}_pred_head"] = pred_head_test[:, j]
        pred_rows[f"{gene}_pred_z_ridge"] = pred_z_ridge_test[:, j]

    pred_rows.to_csv(outdir / "r13_autoencoder_predictions.csv", index=False)

    print("\nAutoencoder comparison:")
    print(metrics_df)

    existing_baseline_path = outdir / "r13_baseline_comparison.csv"
    if existing_baseline_path.exists():
        baseline_df = pd.read_csv(existing_baseline_path)
        combined = pd.concat([baseline_df, metrics_df], ignore_index=True, sort=False)
        combined.to_csv(outdir / "r13_baseline_plus_autoencoder_comparison.csv", index=False)
        print(f"\nSaved combined comparison to: {outdir / 'r13_baseline_plus_autoencoder_comparison.csv'}")


if __name__ == "__main__":
    main()

