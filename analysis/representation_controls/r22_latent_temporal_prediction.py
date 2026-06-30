#!/usr/bin/env python3
"""
R2.2 latent temporal prediction control.

This script addresses the reviewer question:

    Can the autoencoder latent vector z(t) predict z(t+k)?

The analysis uses the same multi-seed simulation outputs and held-out split used
for the representation-control analyses.

Default split:
    Train:      38, 40, 53, 65, 89, 90
    Validation: 96, 101
    Test:       104, 107

For each seed, morphology snapshots are encoded as 10-class one-hot tensors.
A multitask autoencoder is trained on training seeds, with early stopping on
validation seeds. The learned encoder is then used to extract z(t). Ridge
regression models are trained to predict z(t+k) from z(t), and evaluated on
held-out seeds. Temporal persistence z(t+k) = z(t) is reported as a baseline.

Outputs:
    r22_latent_temporal_prediction.csv


Usage examples: 

python analysis/representation_controls/r22_latent_temporal_prediction.py \
  --runs_dir examples/runs \
  --output_dir analysis/representation_controls/outputs

python analysis/representation_controls/r22_latent_temporal_prediction.py \
  --runs_dir runs \
  --output_dir analysis/representation_controls/outputs    

"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


GLOBAL_TARGETS = ["T1", "T2", "I", "R", "M", "K", "S"]

DEFAULT_TRAIN_SEEDS = [38, 40, 53, 65, 89, 90]
DEFAULT_VAL_SEEDS = [96, 101]
DEFAULT_TEST_SEEDS = [104, 107]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def normalize_grid_labels(X: np.ndarray, n_classes: int = 10) -> np.ndarray:
    """
    Normalize Evoscope grid labels to class indices used by the autoencoder.

    Observed raw snapshot convention:
        -1 = empty site
         0..8 = non-empty cell / identity states

    Autoencoder class convention:
         0..9 = contiguous class labels for cross-entropy

    Therefore:
        -1 -> 0
         0 -> 1
         ...
         8 -> 9
    """
    X = X.astype(np.int64)

    # Raw Evoscope snapshots: -1 empty, 0..8 cell/identity states
    if X.min() == -1 and X.max() <= n_classes - 2:
        return X + 1

    # Already encoded as 0..9 classes
    if X.min() >= 0 and X.max() < n_classes:
        return X

    raise ValueError(
        f"Unsupported grid label range: min={X.min()}, max={X.max()}. "
        f"Expected raw labels -1..{n_classes - 2} or encoded labels 0..{n_classes - 1}."
    )

def load_seed_data(
    runs_dir: Path,
    seed: int,
    n_classes: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load morphology snapshots and global regulatory targets for one seed.

    Expected structure:
        runs/seed_38/
            global_genes.csv
            snapshots/
                grid_001.npy
                ...
                grid_120.npy

    Alignment:
        grid_001.npy -> global_genes row 0
        grid_002.npy -> global_genes row 1
        ...
    """
    seed_dir = runs_dir / f"seed_{seed}"
    snapshot_dir = seed_dir / "snapshots"
    global_csv = seed_dir / "global_genes.csv"

    if not seed_dir.exists():
        raise FileNotFoundError(f"Missing seed directory: {seed_dir}")
    if not snapshot_dir.exists():
        raise FileNotFoundError(f"Missing snapshots directory: {snapshot_dir}")
    if not global_csv.exists():
        raise FileNotFoundError(f"Missing global_genes.csv: {global_csv}")

    snapshot_files = sorted(snapshot_dir.glob("grid_*.npy"))
    if not snapshot_files:
        raise FileNotFoundError(f"No grid_*.npy files found in: {snapshot_dir}")

    grids = []
    for path in snapshot_files:
        grid = np.load(path)
        if grid.ndim != 2:
            raise ValueError(f"Expected 2D grid in {path}, got shape {grid.shape}")
        grids.append(grid.astype(np.int64))

    X = np.stack(grids, axis=0)

    X = normalize_grid_labels(X, n_classes=n_classes)

    if X.min() < 0 or X.max() >= n_classes:
        raise ValueError(
            f"Normalized grid values must be in [0, {n_classes - 1}], "
            f"got min={X.min()}, max={X.max()} for seed {seed}"
        )

    df = pd.read_csv(global_csv)

    missing = [c for c in GLOBAL_TARGETS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {global_csv}: {missing}")

    y = df[GLOBAL_TARGETS].to_numpy(dtype=np.float32)

    n = min(len(X), len(y))
    X = X[:n]
    y = y[:n]

    return X, y


def one_hot_grids(X: np.ndarray, n_classes: int = 10) -> torch.Tensor:
    """
    Convert integer class maps to one-hot tensors.

    Input:
        X: numpy array of shape (N, H, W)

    Output:
        tensor of shape (N, C, H, W), float32
    """
    x = torch.from_numpy(X).long()
    oh = F.one_hot(x, num_classes=n_classes)
    oh = oh.permute(0, 3, 1, 2).float()
    return oh


class MorphologyAutoencoder(nn.Module):
    """
    Multitask morphology autoencoder.

    Input:
        one-hot morphology tensor, shape (N, 10, 40, 60)

    Encoder:
        Conv2D 10 -> 32 -> 64 -> 128, stride 2

    Latent:
        z in R^8

    Decoder:
        ConvTranspose2D 128 -> 64 -> 32 -> 10

    Prediction head:
        z -> 128 -> 64 -> 7 global regulatory variables
    """

    def __init__(
        self,
        n_classes: int = 10,
        latent_dim: int = 8,
        target_dim: int = 7,
        height: int = 40,
        width: int = 60,
    ) -> None:
        super().__init__()

        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.target_dim = target_dim
        self.height = height
        self.width = width

        self.encoder_conv = nn.Sequential(
            nn.Conv2d(n_classes, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, n_classes, height, width)
            conv_out = self.encoder_conv(dummy)
            self.conv_shape = tuple(conv_out.shape[1:])
            self.flat_dim = int(np.prod(self.conv_shape))

        self.fc_z = nn.Linear(self.flat_dim, latent_dim)

        self.fc_dec = nn.Linear(latent_dim, self.flat_dim)
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, n_classes, kernel_size=4, stride=2, padding=1),
        )

        self.pred_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, target_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder_conv(x)
        h = h.reshape(h.shape[0], -1)
        z = self.fc_z(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z)
        h = h.reshape(z.shape[0], *self.conv_shape)
        logits = self.decoder_conv(h)
        logits = logits[:, :, : self.height, : self.width]
        return logits

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        recon_logits = self.decode(z)
        pred = self.pred_head(z)
        return recon_logits, pred, z


class SnapshotDataset(torch.utils.data.Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, n_classes: int = 10) -> None:
        self.X_int = torch.from_numpy(X).long()
        self.X_oh = one_hot_grids(X, n_classes=n_classes)
        self.y = torch.from_numpy(y).float()

    def __len__(self) -> int:
        return self.X_int.shape[0]

    def __getitem__(self, idx: int):
        return self.X_oh[idx], self.X_int[idx], self.y[idx]


def concatenate_seed_data(
    runs_dir: Path,
    seeds: Sequence[int],
    n_classes: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    all_X = []
    all_y = []
    for seed in seeds:
        X, y = load_seed_data(runs_dir, seed, n_classes=n_classes)
        all_X.append(X)
        all_y.append(y)
    return np.concatenate(all_X, axis=0), np.concatenate(all_y, axis=0)


def train_autoencoder(
    runs_dir: Path,
    train_seeds: Sequence[int],
    val_seeds: Sequence[int],
    output_dir: Path,
    n_classes: int = 10,
    latent_dim: int = 8,
    target_dim: int = 7,
    batch_size: int = 16,
    lr: float = 1e-3,
    max_epochs: int = 100,
    patience: int = 20,
    lambda_rec: float = 1.0,
    lambda_reg: float = 1.0,
    device: str = "cpu",
) -> MorphologyAutoencoder:
    X_train, y_train = concatenate_seed_data(runs_dir, train_seeds, n_classes=n_classes)
    X_val, y_val = concatenate_seed_data(runs_dir, val_seeds, n_classes=n_classes)

    train_ds = SnapshotDataset(X_train, y_train, n_classes=n_classes)
    val_ds = SnapshotDataset(X_val, y_val, n_classes=n_classes)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=False
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, drop_last=False
    )

    height, width = X_train.shape[1], X_train.shape[2]

    model = MorphologyAutoencoder(
        n_classes=n_classes,
        latent_dim=latent_dim,
        target_dim=target_dim,
        height=height,
        width=width,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = math.inf
    best_state = None
    epochs_without_improvement = 0

    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []

        for x_oh, x_int, y in train_loader:
            x_oh = x_oh.to(device)
            x_int = x_int.to(device)
            y = y.to(device)

            recon_logits, pred, _ = model(x_oh)
            rec_loss = F.cross_entropy(recon_logits, x_int)
            reg_loss = F.mse_loss(pred, y)
            loss = lambda_rec * rec_loss + lambda_reg * reg_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.detach().cpu()))

        model.eval()
        val_losses = []
        val_rec_losses = []
        val_reg_losses = []

        with torch.no_grad():
            for x_oh, x_int, y in val_loader:
                x_oh = x_oh.to(device)
                x_int = x_int.to(device)
                y = y.to(device)

                recon_logits, pred, _ = model(x_oh)
                rec_loss = F.cross_entropy(recon_logits, x_int)
                reg_loss = F.mse_loss(pred, y)
                loss = lambda_rec * rec_loss + lambda_reg * reg_loss

                val_losses.append(float(loss.detach().cpu()))
                val_rec_losses.append(float(rec_loss.detach().cpu()))
                val_reg_losses.append(float(reg_loss.detach().cpu()))

        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_rec = float(np.mean(val_rec_losses))
        val_reg = float(np.mean(val_reg_losses))

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_reconstruction_ce": val_rec,
                "val_regulatory_mse": val_reg,
            }
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            break

    if best_state is None:
        raise RuntimeError("Training failed: no best model state was saved.")

    model.load_state_dict(best_state)
    model.to(device)

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "r22_autoencoder_best.pt")
    pd.DataFrame(history).to_csv(output_dir / "r22_autoencoder_training_history.csv", index=False)

    return model


def extract_latents_for_seed(
    model: MorphologyAutoencoder,
    runs_dir: Path,
    seed: int,
    n_classes: int = 10,
    device: str = "cpu",
    batch_size: int = 64,
) -> np.ndarray:
    X, _ = load_seed_data(runs_dir, seed, n_classes=n_classes)
    X_oh = one_hot_grids(X, n_classes=n_classes)

    loader = torch.utils.data.DataLoader(X_oh, batch_size=batch_size, shuffle=False)

    model.eval()
    latents = []

    with torch.no_grad():
        for x in loader:
            x = x.to(device)
            z = model.encode(x)
            latents.append(z.detach().cpu().numpy())

    return np.concatenate(latents, axis=0)


def make_temporal_pairs(
    Z_by_seed: Dict[int, np.ndarray],
    seeds: Sequence[int],
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_list = []
    y_list = []

    for seed in seeds:
        Z = Z_by_seed[seed]
        if len(Z) <= horizon:
            continue

        X_list.append(Z[:-horizon])
        y_list.append(Z[horizon:])

    if not X_list:
        raise ValueError(f"No temporal pairs available for horizon={horizon}")

    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def pearson_mean(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    corrs = []
    for j in range(y_true.shape[1]):
        a = y_true[:, j]
        b = y_pred[:, j]
        if np.std(a) == 0 or np.std(b) == 0:
            continue
        corrs.append(float(np.corrcoef(a, b)[0, 1]))
    if not corrs:
        return float("nan")
    return float(np.mean(corrs))


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred, multioutput="uniform_average")
    pearson = pearson_mean(y_true, y_pred)

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "Pearson_mean": float(pearson),
    }


def run_latent_temporal_prediction(
    Z_by_seed: Dict[int, np.ndarray],
    train_seeds: Sequence[int],
    test_seeds: Sequence[int],
    horizons: Sequence[int],
    ridge_alpha: float = 1.0,
) -> pd.DataFrame:
    rows = []

    for horizon in horizons:
        X_train, y_train = make_temporal_pairs(Z_by_seed, train_seeds, horizon)
        X_test, y_test = make_temporal_pairs(Z_by_seed, test_seeds, horizon)

        # Persistence baseline: z(t+k) = z(t)
        y_pred_persistence = X_test.copy()
        metrics = compute_metrics(y_test, y_pred_persistence)
        rows.append(
            {
                "model": "latent_persistence",
                "input": "autoencoder_latent_z_t",
                "target": f"autoencoder_latent_z_t_plus_{horizon}",
                "horizon": horizon,
                "split": "held_out_seed",
                **metrics,
            }
        )

        # Ridge readout: z(t) -> z(t+k)
        ridge = make_pipeline(
            StandardScaler(),
            Ridge(alpha=ridge_alpha),
        )
        ridge.fit(X_train, y_train)
        y_pred_ridge = ridge.predict(X_test)

        metrics = compute_metrics(y_test, y_pred_ridge)
        rows.append(
            {
                "model": "latent_ridge_z_t_to_z_t_plus_k",
                "input": "autoencoder_latent_z_t",
                "target": f"autoencoder_latent_z_t_plus_{horizon}",
                "horizon": horizon,
                "split": "held_out_seed",
                **metrics,
            }
        )

    return pd.DataFrame(rows)


def parse_seed_list(value: str) -> List[int]:
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="R2.2 latent temporal prediction control: z(t) -> z(t+k)."
    )

    parser.add_argument(
        "--runs_dir",
        type=str,
        default="examples/runs",
        help="Directory containing seed_* run folders.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis/representation_controls/outputs",
        help="Directory for output CSV files and trained model.",
    )
    parser.add_argument(
        "--train_seeds",
        type=str,
        default=",".join(map(str, DEFAULT_TRAIN_SEEDS)),
        help="Comma-separated training seeds.",
    )
    parser.add_argument(
        "--val_seeds",
        type=str,
        default=",".join(map(str, DEFAULT_VAL_SEEDS)),
        help="Comma-separated validation seeds.",
    )
    parser.add_argument(
        "--test_seeds",
        type=str,
        default=",".join(map(str, DEFAULT_TEST_SEEDS)),
        help="Comma-separated held-out test seeds.",
    )
    parser.add_argument(
        "--horizons",
        type=str,
        default="1,5,10",
        help="Comma-separated temporal horizons k for z(t) -> z(t+k).",
    )
    parser.add_argument("--n_classes", type=int, default=10)
    parser.add_argument("--latent_dim", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--ridge_alpha", type=float, default=1.0)
    parser.add_argument("--lambda_rec", type=float, default=1.0)
    parser.add_argument("--lambda_reg", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Use 'cpu' or 'cuda'. CPU is recommended for reproducibility.",
    )

    args = parser.parse_args()

    set_seed(args.seed)

    runs_dir = Path(args.runs_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_seeds = parse_seed_list(args.train_seeds)
    val_seeds = parse_seed_list(args.val_seeds)
    test_seeds = parse_seed_list(args.test_seeds)
    horizons = parse_seed_list(args.horizons)

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU.")
        args.device = "cpu"

    split_info = {
        "train_seeds": train_seeds,
        "val_seeds": val_seeds,
        "test_seeds": test_seeds,
        "horizons": horizons,
        "latent_dim": args.latent_dim,
        "ridge_alpha": args.ridge_alpha,
    }
    with open(output_dir / "r22_latent_temporal_prediction_config.json", "w") as f:
        json.dump(split_info, f, indent=2)

    print("Training autoencoder...")
    model = train_autoencoder(
        runs_dir=runs_dir,
        train_seeds=train_seeds,
        val_seeds=val_seeds,
        output_dir=output_dir,
        n_classes=args.n_classes,
        latent_dim=args.latent_dim,
        target_dim=len(GLOBAL_TARGETS),
        batch_size=args.batch_size,
        lr=args.lr,
        max_epochs=args.max_epochs,
        patience=args.patience,
        lambda_rec=args.lambda_rec,
        lambda_reg=args.lambda_reg,
        device=args.device,
    )

    print("Extracting latent trajectories...")
    all_seeds = train_seeds + val_seeds + test_seeds
    Z_by_seed: Dict[int, np.ndarray] = {}

    for seed in all_seeds:
        Z = extract_latents_for_seed(
            model=model,
            runs_dir=runs_dir,
            seed=seed,
            n_classes=args.n_classes,
            device=args.device,
            batch_size=64,
        )
        Z_by_seed[seed] = Z
        np.save(output_dir / f"r22_latents_seed_{seed}.npy", Z)

    print("Running latent temporal prediction controls...")
    df = run_latent_temporal_prediction(
        Z_by_seed=Z_by_seed,
        train_seeds=train_seeds,
        test_seeds=test_seeds,
        horizons=horizons,
        ridge_alpha=args.ridge_alpha,
    )

    output_csv = output_dir / "r22_latent_temporal_prediction.csv"
    df.to_csv(output_csv, index=False)

    print("\nLatent temporal prediction results:")
    print(df.to_string(index=False))
    print(f"\nSaved: {output_csv}")


if __name__ == "__main__":
    main()