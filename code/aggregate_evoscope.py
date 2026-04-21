
#!/usr/bin/env python3
"""
Aggregate Evoscope outputs across multiple runs and generate:
1) Figure 4-style summary plot with 8 panels:
   A: global mean ± std for the 7 genes
   B-H: one panel per gene, showing cluster-wise mean ± std across runs
2) CSV tables with aggregated mean/std values

Expected per-run files:
    global_genes.csv
    cluster_genes.csv

Typical folder layout:
runs/
  run_001/global_genes.csv
  run_001/cluster_genes.csv
  run_002/global_genes.csv
  run_002/cluster_genes.csv
  ...

Usage example:
python aggregate_evoscope_fig4.py \
    --root runs \
    --outdir aggregated_outputs \
    --figure-name figure4_aggregated.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


GENES = ["T1", "T2", "I", "R", "M", "K", "S"]
CLUSTERS = list(range(8))


def find_run_files(root_dir: str) -> Tuple[List[Path], List[Path]]:
    root = Path(root_dir)
    global_files = sorted(root.glob("**/global_genes.csv"))
    cluster_files = sorted(root.glob("**/cluster_genes.csv"))

    if not global_files:
        raise FileNotFoundError(f"No global_genes.csv files found under: {root}")
    if not cluster_files:
        raise FileNotFoundError(f"No cluster_genes.csv files found under: {root}")

    return global_files, cluster_files


def get_common_steps(dfs: List[pd.DataFrame], step_col: str = "step") -> List[int]:
    common_steps = set(dfs[0][step_col].tolist())
    for df in dfs[1:]:
        common_steps &= set(df[step_col].tolist())
    common_steps = sorted(common_steps)
    if not common_steps:
        raise ValueError("No common steps found across runs.")
    return common_steps


def load_global_data(global_files: List[Path]) -> Dict[str, np.ndarray]:
    """
    Returns:
        dict:
            'steps' -> array shape (n_steps,)
            gene -> array shape (n_runs, n_steps)
    """
    dfs = [pd.read_csv(f).sort_values("step").reset_index(drop=True) for f in global_files]
    common_steps = get_common_steps(dfs, "step")

    data = {g: [] for g in GENES}
    for df in dfs:
        sub = df[df["step"].isin(common_steps)].sort_values("step")
        for g in GENES:
            data[g].append(sub[g].to_numpy(dtype=float))

    for g in GENES:
        data[g] = np.vstack(data[g])

    data["steps"] = np.array(common_steps, dtype=int)
    return data


def load_cluster_data(cluster_files: List[Path]) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Returns:
        dict:
            'steps' -> array shape (n_steps,)
            gene -> dict cluster_id -> array shape (n_runs, n_steps)
    """
    dfs = [pd.read_csv(f).sort_values(["cluster", "step"]).reset_index(drop=True) for f in cluster_files]
    common_steps = get_common_steps(dfs, "step")

    out = {g: {cid: [] for cid in CLUSTERS} for g in GENES}

    for df in dfs:
        for cid in CLUSTERS:
            sub = df[(df["cluster"] == cid) & (df["step"].isin(common_steps))].sort_values("step")

            if len(sub) != len(common_steps):
                raise ValueError(
                    f"Cluster {cid} is missing some common steps in file: {df}"
                )

            for g in GENES:
                out[g][cid].append(sub[g].to_numpy(dtype=float))

    for g in GENES:
        for cid in CLUSTERS:
            out[g][cid] = np.vstack(out[g][cid])

    out["steps"] = np.array(common_steps, dtype=int)
    return out


def mean_std(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    return np.mean(arr, axis=0), np.std(arr, axis=0, ddof=0)


def build_global_summary_df(global_data: Dict[str, np.ndarray]) -> pd.DataFrame:
    steps = global_data["steps"]
    rows = []

    for idx, step in enumerate(steps):
        row = {"step": int(step)}
        for gene in GENES:
            mu, sd = mean_std(global_data[gene])
            row[f"{gene}_mean"] = float(mu[idx])
            row[f"{gene}_std"] = float(sd[idx])
        rows.append(row)

    return pd.DataFrame(rows)


def build_cluster_summary_df(cluster_data: Dict[str, Dict[int, np.ndarray]]) -> pd.DataFrame:
    steps = cluster_data["steps"]
    rows = []

    for gene in GENES:
        for cid in CLUSTERS:
            mu, sd = mean_std(cluster_data[gene][cid])
            for idx, step in enumerate(steps):
                rows.append({
                    "step": int(step),
                    "gene": gene,
                    "cluster": cid,
                    "mean": float(mu[idx]),
                    "std": float(sd[idx]),
                })

    return pd.DataFrame(rows)


def plot_aggregated_figure4(
    global_data: Dict[str, np.ndarray],
    cluster_data: Dict[str, Dict[int, np.ndarray]],
    output_path: Path,
    title: str = "Aggregated regulatory dynamics across simulations",
):
    steps = global_data["steps"]

    fig, axes = plt.subplots(4, 2, figsize=(16, 18))
    axes = axes.flatten()

    panel_titles = {
        0: "Global mean expression",
        1: "T1 by cluster",
        2: "T2 by cluster",
        3: "I by cluster",
        4: "R by cluster",
        5: "M by cluster",
        6: "K by cluster",
        7: "S by cluster",
    }

    # Panel A: global 7 genes
    ax = axes[0]
    for gene in GENES:
        mu, sd = mean_std(global_data[gene])
        ax.plot(steps, mu, label=gene, linewidth=2)
        ax.fill_between(steps, mu - sd, mu + sd, alpha=0.18)
    ax.set_title(panel_titles[0], fontsize=20)
    ax.set_xlabel("Step")
    ax.set_ylabel("Expression")
    ax.legend(frameon=False, ncol=2, fontsize=12)
    ax.grid(alpha=0.25)

    # Panels B-H: one gene per panel, 8 clusters inside
    gene_order = ["T1", "T2", "I", "R", "M", "K", "S"]
    for i, gene in enumerate(gene_order, start=1):
        ax = axes[i]
        for cid in CLUSTERS:
            mu, sd = mean_std(cluster_data[gene][cid])
            ax.plot(steps, mu, label=f"C{cid}", linewidth=1.8)
            ax.fill_between(steps, mu - sd, mu + sd, alpha=0.12)
        ax.set_title(panel_titles[i], fontsize=20)
        ax.set_xlabel("Step")
        ax.set_ylabel(f"{gene} expression")
        ax.grid(alpha=0.25)
        ax.legend(frameon=False, ncol=2, fontsize=12)

    fig.suptitle(title, fontsize=20)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate Evoscope outputs across multiple runs.")
    parser.add_argument("--root", required=True, help="Root folder containing multiple run subfolders.")
    parser.add_argument("--outdir", default="aggregated_outputs", help="Output directory.")
    parser.add_argument("--figure-name", default="figure4_aggregated.png", help="Output figure filename.")
    parser.add_argument("--title", default="Aggregated regulatory dynamics across simulations", help="Figure title.")
    return parser.parse_args()


def main():
    args = parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    global_files, cluster_files = find_run_files(args.root)
    print(f"Found {len(global_files)} global_genes.csv files")
    print(f"Found {len(cluster_files)} cluster_genes.csv files")

    global_data = load_global_data(global_files)
    cluster_data = load_cluster_data(cluster_files)

    # Save aggregated tables
    global_df = build_global_summary_df(global_data)
    cluster_df = build_cluster_summary_df(cluster_data)

    global_csv = outdir / "global_genes_aggregated_mean_std.csv"
    cluster_csv = outdir / "cluster_genes_aggregated_mean_std.csv"

    global_df.to_csv(global_csv, index=False)
    cluster_df.to_csv(cluster_csv, index=False)

    # Save figure
    fig_path = outdir / args.figure_name
    plot_aggregated_figure4(
        global_data=global_data,
        cluster_data=cluster_data,
        output_path=fig_path,
        title=args.title,
    )

    print(f"Saved figure to: {fig_path}")
    print(f"Saved global summary table to: {global_csv}")
    print(f"Saved cluster summary table to: {cluster_csv}")


if __name__ == "__main__":
    main()
