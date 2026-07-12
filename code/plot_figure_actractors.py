#!/usr/bin/env python3

import argparse
import pandas as pd
import matplotlib.pyplot as plt


def detect_latent_cols(df):
    return [c for c in df.columns if c.lower().startswith("z")]


def plot_latent_trajectories(ax, df, title):
    latent_cols = detect_latent_cols(df)

    for col in latent_cols:
        ax.plot(df["step"], df[col], label=col, linewidth=2)

    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Latent value")
    ax.legend(ncol=4, fontsize=8)


def plot_phase(ax, df, title):
    ax.plot(df["z1"], df["z2"], "-o", markersize=4)

    # annotate every ~15 points
    for i in range(0, len(df), max(1, len(df)//8)):
        ax.text(df["z1"].iloc[i], df["z2"].iloc[i], str(df["step"].iloc[i]), fontsize=9)

    ax.set_title(title)
    ax.set_xlabel("z1")
    ax.set_ylabel("z2")


def main():
    parser = argparse.ArgumentParser(description="Generate Figure 4 latent trajectories + phase portraits")

    parser.add_argument("--global_latents", required=True,
                        help="CSV file with global latent variables")

    parser.add_argument("--cluster_latents", required=True,
                        help="CSV file with cluster latent variables")

    parser.add_argument("--outfile", default="Figure5.png",
                        help="Output image file")

    args = parser.parse_args()

    gdf = pd.read_csv(args.global_latents)
    cdf = pd.read_csv(args.cluster_latents)

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    # A
    plot_latent_trajectories(axs[0, 0], gdf, "A Global latent trajectories")

    # B
    plot_latent_trajectories(axs[0, 1], cdf, "B Cluster latent trajectories")

    # C
    plot_phase(axs[1, 0], gdf, "C Global latent phase portrait (z1,z2)")

    # D
    plot_phase(axs[1, 1], cdf, "D Cluster latent phase portrait (z1,z2)")

    plt.tight_layout()
    plt.savefig(args.outfile, dpi=300)
    plt.show()

    print(f"Saved: {args.outfile}")


if __name__ == "__main__":
    main()
