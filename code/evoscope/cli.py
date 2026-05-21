"""
Command-line interface for running Evoscope simulations.

This module provides a lightweight CLI entry point for configuring and
executing an Evoscope simulation from the terminal. It parses user-specified
grid size, seed, number of epochs, initial cell count, nutrient level, and
plotting options.

After the simulation completes, the CLI exports ASCII frames, prints a final
summary, saves global and cluster-level gene/protein CSV files, and can
optionally generate diagnostic plots.

Evoscope v0.9.1
Author: Luca Zammataro
Organization: Lunan Foldomics LLC
"""

import argparse

from .config import Config
from .simulation import Evoscope
from .io import save_cluster_gene_csv, save_global_gene_csv
from .visualization import plot_clusters, plot_genes


def parse_args():
    parser = argparse.ArgumentParser(description="Evoscope simulation")
    parser.add_argument("--width", type=int, default=46)
    parser.add_argument("--height", type=int, default=38)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--initial_cells", type=int, default=30)
    parser.add_argument("--nutrient", type=float, default=6.9)
    parser.add_argument("--plot", type=str, default="n")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = Config(
        width=args.width,
        height=args.height,
        initial_cells=args.initial_cells,
        initial_medium_nutrient=args.nutrient,
        seed=args.seed,
        verbose=False,
    )

    sim = Evoscope(cfg)
    frames = sim.run_with_ascii_frames(
        epochs=args.epochs,
        every=1,
        include_initial=True,
        clear_screen=True,
        pause=0.08,
    )

    sim.save_ascii_frames("evoscope_ascii_frames.txt", frames)
    print("\nFinal summary:")
    print(sim.summary_line())
    print("\nFinal snapshot:")
    print(sim.ascii_snapshot(colored=True))
    print("\nSaved ASCII frames to: evoscope_ascii_frames.txt")

    save_global_gene_csv(sim)
    save_cluster_gene_csv(sim)

    if args.plot.lower() == "y":
        plot_genes(sim, save_path="global_genes.png")
        for gene in ["T1", "T2", "I", "R", "M", "K", "S"]:
            plot_clusters(sim, gene=gene, save_path=f"{gene}_clusters.png")


if __name__ == "__main__":
    main()
