"""
Latent-space export and prediction utilities for Evoscope autoencoders.

This module contains helper functions for applying a trained autoencoder to
an Evoscope dataset. It extracts latent variables for each simulation step and
exports true versus predicted gene/protein targets.

These outputs support downstream interpretation of learned morphology
coordinates, including trajectory analysis, attractor-like phase portraits,
and correlation with observable regulatory dynamics.

Evoscope v0.9.1
Author: Luca Zammataro
Organization: Lunan Foldomics LLC
"""


from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import pandas as pd
import torch
from torch.utils.data import DataLoader

from .datasets import CLUSTERS, GLOBAL_GENES

PathLike = Union[str, os.PathLike]


def resolve_device(device: Optional[str] = None) -> str:
    if device is not None:
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def encode_dataset(model, dataset, device: Optional[str] = None, batch_size: int = 32) -> pd.DataFrame:
    """Encode an Evoscope dataset and return latent variables as a dataframe."""

    device = resolve_device(device)
    model = model.to(device)
    model.eval()

    rows: List[dict] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    for batch in loader:
        x = batch["x"].to(device)
        steps = batch["step"].cpu().numpy()
        _, _, z = model(x)
        z = z.cpu().numpy()
        for step, zi in zip(steps, z):
            row = {"step": int(step)}
            for i, val in enumerate(zi):
                row[f"z{i + 1}"] = float(val)
            rows.append(row)

    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


@torch.no_grad()
def predict_genes(model, dataset, device: Optional[str] = None, batch_size: int = 32) -> pd.DataFrame:
    """Return true/predicted gene targets for all samples in a dataset."""

    device = resolve_device(device)
    model = model.to(device)
    model.eval()

    rows: List[dict] = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    target_mode = getattr(dataset, "target_mode", "global")

    for batch in loader:
        x = batch["x"].to(device)
        steps = batch["step"].cpu().numpy()
        y_true = batch["y"].cpu().numpy()
        _, y_pred, _ = model(x)
        y_pred = y_pred.cpu().numpy()

        for step, yt, yp in zip(steps, y_true, y_pred):
            row = {"step": int(step)}
            if target_mode == "global":
                for i, gene in enumerate(GLOBAL_GENES):
                    row[f"true_{gene}"] = float(yt[i])
                    row[f"pred_{gene}"] = float(yp[i])
            else:
                k = 0
                for cid in CLUSTERS:
                    for gene in GLOBAL_GENES:
                        row[f"true_c{cid}_{gene}"] = float(yt[k])
                        row[f"pred_c{cid}_{gene}"] = float(yp[k])
                        k += 1
            rows.append(row)

    return pd.DataFrame(rows).sort_values("step").reset_index(drop=True)


def export_latents(model, dataset, out_csv: PathLike, device: Optional[str] = None, batch_size: int = 32) -> pd.DataFrame:
    """Encode a dataset, save latent variables to CSV, and return the dataframe."""

    df = encode_dataset(model, dataset, device=device, batch_size=batch_size)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df


def export_predictions(
    model,
    dataset,
    out_csv: PathLike,
    device: Optional[str] = None,
    target_mode: Optional[str] = None,
    batch_size: int = 32,
) -> pd.DataFrame:
    """Save true/predicted gene targets to CSV and return the dataframe."""

    # target_mode is accepted for backward compatibility with the original script.
    if target_mode is not None and hasattr(dataset, "target_mode"):
        dataset.target_mode = target_mode

    df = predict_genes(model, dataset, device=device, batch_size=batch_size)
    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return df
