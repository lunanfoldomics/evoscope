"""
Convolutional autoencoder models for Evoscope morphology snapshots.

This module implements the neural-network components used to learn latent
representations of Evoscope spatial configurations. The main model encodes
one-hot grid snapshots into a low-dimensional latent space, reconstructs the
input morphology, and optionally predicts associated global or cluster-level
gene/protein targets.

The module also provides training utilities, validation logic, early stopping,
device resolution, history export, latent export, and prediction export. It
can be used both as a command-line script and as an importable training module
inside notebooks or larger workflows.


Evoscope — minimal regulatory spatial model for emergent multicellular organization.

Author: Luca Zammataro
Organization: Lunan Foldomics LLC
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from .datasets import N_CLASSES, TorexDataset
from .latent import export_latents, export_predictions

PathLike = Union[str, os.PathLike]


class ConvAutoencoder(nn.Module):
    """Convolutional autoencoder with an optional gene-prediction head."""

    def __init__(self, height: int, width: int, latent_dim: int, target_dim: int):
        super().__init__()
        self.height = height
        self.width = width
        self.latent_dim = latent_dim
        self.target_dim = target_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(N_CLASSES, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, N_CLASSES, height, width)
            enc = self.encoder(dummy)
            self.enc_shape = enc.shape[1:]
            flat_dim = int(np.prod(self.enc_shape))

        self.to_latent = nn.Linear(flat_dim, latent_dim)
        self.from_latent = nn.Linear(latent_dim, flat_dim)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(32, N_CLASSES, kernel_size=4, stride=2, padding=1),
        )

        self.gene_head = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, target_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x)
        h = h.flatten(1)
        return self.to_latent(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = h.view(-1, *self.enc_shape)
        recon_logits = self.decoder(h)
        return recon_logits[:, :, : self.height, : self.width]

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        recon_logits = self.decode(z)
        gene_pred = self.gene_head(z)
        return recon_logits, gene_pred, z


class EarlyStopper:
    """Minimal early stopper based on validation loss."""

    def __init__(self, patience: int = 20):
        self.patience = patience
        self.best: Optional[float] = None
        self.count = 0

    def step(self, value: float) -> bool:
        if self.best is None or value < self.best:
            self.best = value
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience


def resolve_device(device: Optional[str] = None) -> str:
    """Return a usable device string."""

    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: str,
    recon_weight: float,
    gene_weight: float,
) -> Dict[str, float]:
    model.train()
    recon_loss_fn = nn.CrossEntropyLoss()
    gene_loss_fn = nn.MSELoss()

    total_loss = total_recon = total_gene = 0.0
    total_n = 0

    for batch in loader:
        x = batch["x"].to(device)
        cls = batch["classes"].to(device)
        y = batch["y"].to(device)

        recon_logits, gene_pred, _ = model(x)
        loss_recon = recon_loss_fn(recon_logits, cls)
        loss_gene = gene_loss_fn(gene_pred, y)
        loss = recon_weight * loss_recon + gene_weight * loss_gene

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += loss_recon.item() * bs
        total_gene += loss_gene.item() * bs
        total_n += bs

    return {
        "loss": total_loss / total_n,
        "recon": total_recon / total_n,
        "gene": total_gene / total_n,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    recon_weight: float,
    gene_weight: float,
) -> Dict[str, float]:
    model.eval()
    recon_loss_fn = nn.CrossEntropyLoss()
    gene_loss_fn = nn.MSELoss()

    total_loss = total_recon = total_gene = 0.0
    total_n = 0

    for batch in loader:
        x = batch["x"].to(device)
        cls = batch["classes"].to(device)
        y = batch["y"].to(device)

        recon_logits, gene_pred, _ = model(x)
        loss_recon = recon_loss_fn(recon_logits, cls)
        loss_gene = gene_loss_fn(gene_pred, y)
        loss = recon_weight * loss_recon + gene_weight * loss_gene

        bs = x.size(0)
        total_loss += loss.item() * bs
        total_recon += loss_recon.item() * bs
        total_gene += loss_gene.item() * bs
        total_n += bs

    return {
        "loss": total_loss / total_n,
        "recon": total_recon / total_n,
        "gene": total_gene / total_n,
    }


def make_train_val_loaders(
    dataset: Dataset,
    batch_size: int = 16,
    val_fraction: float = 0.2,
    split_seed: int = 11,
    loader_seed: int = 11,
) -> Tuple[DataLoader, DataLoader]:
    """Split a dataset and return reproducible train/validation loaders.

    Parameters
    ----------
    split_seed:
        Controls which samples are assigned to train vs validation.
    loader_seed:
        Controls the randomized order of training batches. Keeping this fixed
        allows model initialization to be varied independently.
    """

    n_total = len(dataset)
    n_val = max(1, int(n_total * val_fraction))
    n_train = max(1, n_total - n_val)
    if n_train + n_val > n_total:
        n_val = n_total - n_train

    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(split_seed),
    )

    train_generator = torch.Generator().manual_seed(loader_seed)
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader


def train_autoencoder(
    dataset: TorexDataset,
    latent_dim: int = 8,
    batch_size: int = 16,
    epochs: int = 100,
    lr: float = 1e-3,
    recon_weight: float = 1.0,
    gene_weight: float = 1.0,
    val_fraction: float = 0.2,
    seed: Optional[int] = 11,
    model_seed: Optional[int] = None,
    split_seed: Optional[int] = None,
    loader_seed: Optional[int] = None,
    outdir: Optional[PathLike] = None,
    device: Optional[str] = None,
    patience: int = 20,
    verbose: bool = True,
) -> Tuple[ConvAutoencoder, pd.DataFrame]:
    """Train a ConvAutoencoder from a notebook or script.

    Returns
    -------
    model:
        The best-validation model.
    history:
        Training history as a dataframe.
    """

    # Backward-compatible seed handling:
    # --seed acts as the legacy master seed unless explicit seeds are supplied.
    master_seed = 11 if seed is None else int(seed)
    model_seed = master_seed if model_seed is None else int(model_seed)
    split_seed = master_seed if split_seed is None else int(split_seed)
    loader_seed = split_seed if loader_seed is None else int(loader_seed)

    # Model initialization / stochastic neural-network operations.
    random.seed(model_seed)
    np.random.seed(model_seed)
    torch.manual_seed(model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(model_seed)

    device = resolve_device(device)
    train_loader, val_loader = make_train_val_loaders(
        dataset=dataset,
        batch_size=batch_size,
        val_fraction=val_fraction,
        split_seed=split_seed,
        loader_seed=loader_seed,
    )

    model = ConvAutoencoder(
        height=dataset.height,
        width=dataset.width,
        latent_dim=latent_dim,
        target_dim=dataset.target_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    stopper = EarlyStopper(patience=patience)
    history_rows: List[Dict[str, float]] = []
    best_state = None
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, device, recon_weight, gene_weight
        )
        val_metrics = evaluate(model, val_loader, device, recon_weight, gene_weight)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_recon": train_metrics["recon"],
            "train_gene": train_metrics["gene"],
            "val_loss": val_metrics["loss"],
            "val_recon": val_metrics["recon"],
            "val_gene": val_metrics["gene"],
        }
        history_rows.append(row)

        if verbose:
            print(
                f"epoch {epoch:03d} | "
                f"train loss {train_metrics['loss']:.4f} "
                f"(recon {train_metrics['recon']:.4f}, gene {train_metrics['gene']:.4f}) | "
                f"val loss {val_metrics['loss']:.4f} "
                f"(recon {val_metrics['recon']:.4f}, gene {val_metrics['gene']:.4f})"
            )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if stopper.step(val_metrics["loss"]):
            if verbose:
                print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    history = pd.DataFrame(history_rows)

    if outdir is not None:
        out_path = Path(outdir)
        out_path.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), out_path / "torex_autoencoder.pt")
        history.to_csv(out_path / "training_history.csv", index=False)

        reproducibility = {
            "seed": master_seed,
            "model_seed": model_seed,
            "split_seed": split_seed,
            "loader_seed": loader_seed,
            "device": device,
            "latent_dim": latent_dim,
            "batch_size": batch_size,
            "epochs_requested": epochs,
            "learning_rate": lr,
            "recon_weight": recon_weight,
            "gene_weight": gene_weight,
            "val_fraction": val_fraction,
            "patience": patience,
        }
        with open(out_path / "reproducibility.json", "w", encoding="utf-8") as fh:
            json.dump(reproducibility, fh, indent=2)

        export_latents(model, dataset, out_path / "latents.csv", device=device)
        export_predictions(
            model,
            dataset,
            out_path / "predictions.csv",
            device=device,
            target_mode=dataset.target_mode,
        )
        if verbose:
            print(f"Saved model and outputs to: {out_path}")

    return model, history


def train_autoencoder_from_files(
    snapshots_dir: PathLike = "snapshots",
    global_csv: PathLike = "global_genes.csv",
    cluster_csv: PathLike = "cluster_genes.csv",
    target_mode: str = "global",
    **kwargs,
) -> Tuple[ConvAutoencoder, pd.DataFrame, TorexDataset]:
    """Convenience wrapper: build dataset from files and train the autoencoder."""

    dataset = TorexDataset(
        snapshots_dir=snapshots_dir,
        target_mode=target_mode,
        global_csv=global_csv,
        cluster_csv=cluster_csv,
    )
    model, history = train_autoencoder(dataset, **kwargs)
    return model, history, dataset


def parse_args():
    p = argparse.ArgumentParser(description="Train an Evoscope/TorHex morphology autoencoder")
    p.add_argument("--snapshots_dir", type=str, default="snapshots")
    p.add_argument("--global_csv", type=str, default="global_genes.csv")
    p.add_argument("--cluster_csv", type=str, default="cluster_genes.csv")
    p.add_argument("--target_mode", type=str, default="global", choices=["global", "cluster_flat"])
    p.add_argument("--latent_dim", type=int, default=8)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--recon_weight", type=float, default=1.0)
    p.add_argument("--gene_weight", type=float, default=1.0)
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument(
        "--seed",
        type=int,
        default=11,
        help=(
            "Legacy master seed. Used for model, split, and loader seeds unless "
            "the corresponding explicit options are supplied."
        ),
    )
    p.add_argument(
        "--model_seed",
        type=int,
        default=None,
        help="Seed for autoencoder weight initialization and model stochasticity.",
    )
    p.add_argument(
        "--split_seed",
        type=int,
        default=None,
        help="Seed controlling the train/validation split.",
    )
    p.add_argument(
        "--loader_seed",
        type=int,
        default=None,
        help="Seed controlling shuffled training-batch order.",
    )
    p.add_argument("--outdir", type=str, default="ae_outputs")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--patience", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    train_autoencoder_from_files(
        snapshots_dir=args.snapshots_dir,
        global_csv=args.global_csv,
        cluster_csv=args.cluster_csv,
        target_mode=args.target_mode,
        latent_dim=args.latent_dim,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        recon_weight=args.recon_weight,
        gene_weight=args.gene_weight,
        val_fraction=args.val_fraction,
        seed=args.seed,
        model_seed=args.model_seed,
        split_seed=args.split_seed,
        loader_seed=args.loader_seed,
        outdir=args.outdir,
        device=args.device,
        patience=args.patience,
        verbose=True,
    )


if __name__ == "__main__":
    main()
