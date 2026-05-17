"""Correlation analysis between latent variables and observable Evoscope metrics."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PathLike = Union[str, os.PathLike]


def read_table(obj: Union[PathLike, pd.DataFrame]) -> pd.DataFrame:
    """Read a CSV path or copy an existing dataframe."""

    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    return pd.read_csv(obj)


def detect_latent_cols(df: pd.DataFrame) -> List[str]:
    """Detect latent columns named z1, z2, ..."""

    return [c for c in df.columns if c.lower().startswith("z")]


def merge_latents_and_metrics(
    latents: Union[PathLike, pd.DataFrame],
    metrics: Union[PathLike, pd.DataFrame],
) -> pd.DataFrame:
    """Merge latent variables and metric/gene tables.

    If both inputs contain ``step``, merge on ``step``. Otherwise concatenate by row order.
    """

    lat = read_table(latents)
    met = read_table(metrics)

    if "step" in lat.columns and "step" in met.columns:
        return pd.merge(lat, met, on="step")
    return pd.concat([lat.reset_index(drop=True), met.reset_index(drop=True)], axis=1)


def compute_latent_metric_correlations(
    latents: Union[PathLike, pd.DataFrame],
    metrics: Union[PathLike, pd.DataFrame],
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation matrix: latent variables × observable metrics."""

    df = merge_latents_and_metrics(latents, metrics)
    latent_cols = detect_latent_cols(df)
    metric_cols = [c for c in df.columns if c not in latent_cols and c != "step"]

    if not latent_cols:
        raise ValueError("No latent columns found. Expected column names like z1, z2, ...")
    if not metric_cols:
        raise ValueError("No metric columns found.")

    corr = pd.DataFrame(index=latent_cols, columns=metric_cols, dtype=float)
    for z in latent_cols:
        for m in metric_cols:
            corr.loc[z, m] = df[z].corr(df[m], method=method)
    return corr.astype(float)


def plot_correlation_heatmap(
    corr: pd.DataFrame,
    outfile: Optional[PathLike] = None,
    title: str = "Latent Variables × Observable Metrics",
    figsize: Optional[Tuple[float, float]] = None,
    show_values: bool = True,
    cmap: str = "coolwarm",
    vmin: float = -1,
    vmax: float = 1,
):
    """Plot a latent-variable correlation heatmap."""

    metric_cols = list(corr.columns)
    latent_cols = list(corr.index)

    if figsize is None:
        fig_w = max(10, len(metric_cols) * 0.7)
        fig_h = max(4, len(latent_cols) * 0.6)
        figsize = (fig_w, fig_h)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_yticks(np.arange(len(latent_cols)))
    ax.set_xticklabels(metric_cols, rotation=45, ha="right", fontsize=16)
    ax.set_yticklabels(latent_cols, fontsize=16)

    if show_values:
        for i in range(len(latent_cols)):
            for j in range(len(metric_cols)):
                val = corr.values[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=15, color="black")

    ax.set_title(title, fontsize=18)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Pearson correlation", fontsize=16)
    cbar.ax.tick_params(labelsize=14)

    plt.tight_layout()
    if outfile is not None:
        out_path = Path(outfile)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=300)
    return fig, ax


def correlation_heatmap_from_files(
    latents_csv: PathLike,
    metrics_csv: PathLike,
    outfile: Optional[PathLike] = "latent_heatmap.png",
    title: str = "Latent Variables × Observable Metrics",
    method: str = "pearson",
):
    """Convenience function: compute correlations from CSV files and plot heatmap."""

    corr = compute_latent_metric_correlations(latents_csv, metrics_csv, method=method)
    fig, ax = plot_correlation_heatmap(corr, outfile=outfile, title=title)
    return corr, fig, ax


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate heatmap of correlations: latent variables × observable metrics"
    )
    parser.add_argument("--latents", required=True, help="CSV file containing latent variables")
    parser.add_argument("--metrics", required=True, help="CSV file containing metrics / genes")
    parser.add_argument("--outfile", default="latent_heatmap.png", help="Output image filename")
    parser.add_argument("--title", default="Latent Variables × Observable Metrics", help="Plot title")
    parser.add_argument("--method", default="pearson", choices=["pearson", "spearman", "kendall"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    corr, _, _ = correlation_heatmap_from_files(
        latents_csv=args.latents,
        metrics_csv=args.metrics,
        outfile=args.outfile,
        title=args.title,
        method=args.method,
    )
    plt.show()
    print(f"Saved heatmap: {args.outfile}")


if __name__ == "__main__":
    main()
