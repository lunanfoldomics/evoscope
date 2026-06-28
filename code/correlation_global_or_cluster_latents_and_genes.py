#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description="Generate heatmap of correlations: latent variables × observable metrics"
    )

    parser.add_argument(
        "--latents",
        required=True,
        help="CSV file containing latent variables (e.g. global_latents.csv)"
    )

    parser.add_argument(
        "--metrics",
        required=True,
        help="CSV file containing metrics / genes (e.g. global_genes.csv)"
    )

    parser.add_argument(
        "--outfile",
        default="latent_heatmap.png",
        help="Output image filename (default: latent_heatmap.png)"
    )

    parser.add_argument(
        "--title",
        default="Latent Variables × Observable Metrics",
        help="Plot title"
    )

    args = parser.parse_args()

    # ==============================
    # LOAD DATA
    # ==============================
    lat = pd.read_csv(args.latents)
    met = pd.read_csv(args.metrics)

    # ==============================
    # MERGE
    # ==============================
    common = [c for c in lat.columns if c in met.columns]

    if "step" in common:
        df = pd.merge(lat, met, on="step")
    else:
        df = pd.concat([lat, met], axis=1)

    # ==============================
    # COLUMN DETECTION
    # ==============================
    latent_cols = [c for c in df.columns if c.lower().startswith("z")]
    metric_cols = [c for c in df.columns if c not in latent_cols and c != "step"]

    if not latent_cols:
        raise ValueError("No latent columns found (expected names like z1, z2, ...)")
    if not metric_cols:
        raise ValueError("No metric columns found.")

    # ==============================
    # CORRELATION MATRIX
    # ==============================
    corr = pd.DataFrame(index=latent_cols, columns=metric_cols)

    for z in latent_cols:
        for m in metric_cols:
            corr.loc[z, m] = df[z].corr(df[m])

    corr = corr.astype(float)

    # ==============================
    # FIGURE SIZE AUTO
    # ==============================
    fig_w = max(10, len(metric_cols) * 0.7)
    fig_h = max(4, len(latent_cols) * 0.6)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(
        corr.values,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        aspect="auto"
    )

    # ticks
    ax.set_xticks(np.arange(len(metric_cols)))
    ax.set_yticks(np.arange(len(latent_cols)))

    ax.set_xticklabels(metric_cols, rotation=45, ha="right", fontsize=16)
    ax.set_yticklabels(latent_cols, fontsize=16)

    for i in range(len(latent_cols)):
        for j in range(len(metric_cols)):
            val = corr.values[i, j]
            ax.text(
                j, i,
                f"{val:.2f}",
                ha="center",
                va="center",
                fontsize=15,
                color="black"
            )

    ax.set_title(args.title, fontsize=18)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Pearson correlation", fontsize=16)
    cbar.ax.tick_params(labelsize=14)
    
    plt.tight_layout()
    plt.savefig(args.outfile, dpi=300)
    plt.show()

    print(f"Saved heatmap: {args.outfile}")


if __name__ == "__main__":
    main()
