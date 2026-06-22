"""
Analysis plots for Evoscope latent trajectories and phase portraits.

This module provides plotting utilities for visualizing the temporal evolution
of learned latent variables and their low-dimensional phase-space behavior.
It supports both global and cluster-level latent representations and can
generate compact summary figures combining latent trajectories with
attractor-like phase portraits.

These plots are designed to help interpret whether simulated morphologies
move through stable, cyclic, transient, or divergent regions of latent space.

Evoscope v0.9.2
Author: Luca Zammataro
Organization: Lunan Foldomics LLC
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd

PathLike = Union[str, os.PathLike]


def read_table(obj: Union[PathLike, pd.DataFrame]) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return pd.read_csv(obj)


def detect_latent_cols(df: pd.DataFrame) -> List[str]:
    """Detect latent columns named z1, z2, ..."""

    return [c for c in df.columns if c.lower().startswith("z")]


def plot_latent_trajectories(
    df: Union[PathLike, pd.DataFrame],
    ax=None,
    title: str = "Latent trajectories",
    step_col: str = "step",
    linewidth: float = 2,
):
    """Plot latent variables as trajectories over simulation step."""

    data = read_table(df)
    latent_cols = detect_latent_cols(data)
    if not latent_cols:
        raise ValueError("No latent columns found. Expected column names like z1, z2, ...")
    if step_col not in data.columns:
        raise ValueError(f"Missing step column: {step_col}")

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    for col in latent_cols:
        ax.plot(data[step_col], data[col], label=col, linewidth=linewidth)

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Latent value")
    ax.legend(ncol=4, fontsize=8)
    return ax


def plot_latent_phase_portrait(
    df: Union[PathLike, pd.DataFrame],
    x: str = "z1",
    y: str = "z2",
    ax=None,
    title: str = "Latent phase portrait",
    step_col: str = "step",
    annotate: bool = True,
    markersize: float = 4,
):
    """Plot a 2D latent phase portrait using two latent dimensions."""

    data = read_table(df)
    for col in (x, y, step_col):
        if col not in data.columns:
            raise ValueError(f"Missing required column: {col}")

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5))

    ax.plot(data[x], data[y], "-o", markersize=markersize)

    if annotate:
        stride = max(1, len(data) // 8)
        for i in range(0, len(data), stride):
            ax.text(data[x].iloc[i], data[y].iloc[i], str(data[step_col].iloc[i]), fontsize=9)

    ax.set_title(title)
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    return ax


def plot_attractor_summary(
    global_latents: Union[PathLike, pd.DataFrame],
    cluster_latents: Union[PathLike, pd.DataFrame],
    outfile: Optional[PathLike] = None,
    figsize: Tuple[float, float] = (14, 10),
):
    """Generate the 2×2 latent trajectory / phase-portrait summary figure."""

    gdf = read_table(global_latents)
    cdf = read_table(cluster_latents)

    fig, axs = plt.subplots(2, 2, figsize=figsize)

    plot_latent_trajectories(gdf, ax=axs[0, 0], title="A Global latent trajectories")
    plot_latent_trajectories(cdf, ax=axs[0, 1], title="B Cluster latent trajectories")
    plot_latent_phase_portrait(gdf, ax=axs[1, 0], title="C Global latent phase portrait (z1,z2)")
    plot_latent_phase_portrait(cdf, ax=axs[1, 1], title="D Cluster latent phase portrait (z1,z2)")

    plt.tight_layout()
    if outfile is not None:
        out_path = Path(outfile)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig, axs


def parse_args():
    parser = argparse.ArgumentParser(description="Generate latent trajectories + phase portraits")
    parser.add_argument("--global_latents", required=True, help="CSV file with global latent variables")
    parser.add_argument("--cluster_latents", required=True, help="CSV file with cluster latent variables")
    parser.add_argument("--outfile", default="Figure4.png", help="Output image file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plot_attractor_summary(
        global_latents=args.global_latents,
        cluster_latents=args.cluster_latents,
        outfile=args.outfile,
    )
    plt.show()
    print(f"Saved: {args.outfile}")


if __name__ == "__main__":
    main()
