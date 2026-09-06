#!/usr/bin/env python3
"""
Autoencoder initialization robustness analysis for Evoscope.

This script compares independently initialized autoencoders while keeping the
underlying dataset, train/validation split, and minibatch order fixed.

Expected directory structure
----------------------------
BASE_DIR/
    global_seed_1/latents.csv
    global_seed_2/latents.csv
    ...
    global_seed_5/latents.csv
    cluster_seed_1/latents.csv
    ...
    cluster_seed_5/latents.csv

Each latents.csv must contain:
    step,z1,z2,...,zN

Primary analysis
----------------
1. Distance-geometry reproducibility:
   For each run, compute the full pairwise Euclidean distance matrix between
   time points in the complete latent space. Correlate the upper triangles of
   these matrices across independent model initializations. This comparison is
   invariant to translation, rotation/reflection, and latent-axis permutation.

2. Temporal-neighborhood preservation:
   Compare distances between consecutive time points with distances between
   non-consecutive time points, and compute the association between temporal
   separation and latent-space distance.

3. Orthogonal Procrustes control:
   Center each trajectory and align it to a reference initialization using only
   an orthogonal transformation (rotation/reflection; no scaling or nonlinear
   warping). Report relative residual error after alignment.

4. Visualization:
   After Procrustes alignment, project all trajectories onto the first two PCs
   of the aligned reference trajectory, providing a common 2D view. Heatmaps of
   pairwise distance-geometry correlations are also generated.

Example
-------
python code/autoencoder_initialization_robustness.py \
    --base_dir runs/seed_42/reviewer1 \
    --seeds 1 2 3 4 5 \
    --reference_seed 1 \
    --outdir runs/seed_42/reviewer1/robustness_analysis

Dependencies: numpy, pandas, matplotlib
"""

from __future__ import annotations

import argparse
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def latent_columns(df: pd.DataFrame) -> List[str]:
    """Return latent columns ordered numerically (z1, z2, ...)."""
    cols = [c for c in df.columns if c.lower().startswith("z")]
    if not cols:
        raise ValueError("No latent columns (z1, z2, ...) were found.")

    def key(c: str):
        suffix = c[1:]
        return int(suffix) if suffix.isdigit() else suffix

    return sorted(cols, key=key)


def load_latents(path: Path) -> pd.DataFrame:
    """Load and validate a latent trajectory CSV."""
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    if "step" not in df.columns:
        raise ValueError(f"{path} does not contain a 'step' column.")
    zcols = latent_columns(df)
    if len(zcols) < 2:
        raise ValueError(f"{path} must contain at least two latent dimensions.")
    if df["step"].duplicated().any():
        raise ValueError(f"Duplicate step values found in {path}.")
    return df.sort_values("step").reset_index(drop=True)


def pairwise_euclidean(x: np.ndarray) -> np.ndarray:
    """Euclidean distance matrix without requiring scipy."""
    diff = x[:, None, :] - x[None, :, :]
    return np.sqrt(np.sum(diff * diff, axis=2))


def upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    idx = np.triu_indices_from(matrix, k=1)
    return matrix[idx]


def pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation with explicit constant-vector validation."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size != b.size:
        raise ValueError("Vectors must have the same size.")
    if np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def rankdata_average(x: np.ndarray) -> np.ndarray:
    """Average ranks for ties; lightweight scipy.stats.rankdata equivalent."""
    s = pd.Series(np.asarray(x))
    return s.rank(method="average").to_numpy(dtype=float)


def spearman_r(a: np.ndarray, b: np.ndarray) -> float:
    return pearson_r(rankdata_average(a), rankdata_average(b))


def validate_common_steps(frames: Dict[int, pd.DataFrame]) -> Tuple[np.ndarray, List[str]]:
    """Ensure all seeds contain identical time points and latent dimensionality."""
    seeds = sorted(frames)
    ref = frames[seeds[0]]
    steps = ref["step"].to_numpy()
    zcols = latent_columns(ref)

    for seed in seeds[1:]:
        df = frames[seed]
        if not np.array_equal(df["step"].to_numpy(), steps):
            raise ValueError(
                f"Seed {seed} has different time points from seed {seeds[0]}."
            )
        if latent_columns(df) != zcols:
            raise ValueError(
                f"Seed {seed} has different latent columns from seed {seeds[0]}."
            )
    return steps, zcols


def geometry_correlations(
    frames: Dict[int, pd.DataFrame], model_type: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pairwise correlations between complete latent-space distance geometries."""
    _, zcols = validate_common_steps(frames)
    distance_vectors: Dict[int, np.ndarray] = {}
    for seed, df in frames.items():
        x = df[zcols].to_numpy(dtype=float)
        distance_vectors[seed] = upper_triangle_values(pairwise_euclidean(x))

    rows = []
    seeds = sorted(frames)
    matrix = pd.DataFrame(np.eye(len(seeds)), index=seeds, columns=seeds, dtype=float)

    for a, b in combinations(seeds, 2):
        r = pearson_r(distance_vectors[a], distance_vectors[b])
        rho = spearman_r(distance_vectors[a], distance_vectors[b])
        rows.append(
            {
                "model_type": model_type,
                "seed_i": a,
                "seed_j": b,
                "geometry_pearson_r": r,
                "geometry_spearman_rho": rho,
            }
        )
        matrix.loc[a, b] = r
        matrix.loc[b, a] = r

    return pd.DataFrame(rows), matrix


def temporal_metrics(frames: Dict[int, pd.DataFrame], model_type: str) -> pd.DataFrame:
    """Quantify local temporal continuity in the complete latent space."""
    steps, zcols = validate_common_steps(frames)
    rows = []

    for seed, df in sorted(frames.items()):
        x = df[zcols].to_numpy(dtype=float)
        d = pairwise_euclidean(x)

        i, j = np.triu_indices(len(x), k=1)
        index_gap = j - i
        time_gap = np.abs(steps[j] - steps[i]).astype(float)
        distances = d[i, j]

        adjacent = distances[index_gap == 1]
        nonadjacent = distances[index_gap > 1]

        mean_adj = float(np.mean(adjacent))
        mean_nonadj = float(np.mean(nonadjacent))
        ratio = mean_adj / mean_nonadj

        rows.append(
            {
                "model_type": model_type,
                "model_seed": seed,
                "n_timepoints": len(x),
                "mean_adjacent_distance": mean_adj,
                "mean_nonadjacent_distance": mean_nonadj,
                "adjacent_nonadjacent_ratio": ratio,
                "temporal_distance_pearson_r": pearson_r(time_gap, distances),
                "temporal_distance_spearman_rho": spearman_r(time_gap, distances),
            }
        )

    return pd.DataFrame(rows)


def orthogonal_procrustes_align(x: np.ndarray, reference: np.ndarray):
    """
    Align x to reference using centering + orthogonal rotation/reflection only.

    No scaling and no nonlinear deformation are permitted.
    Returns aligned coordinates, rotation matrix, and relative Frobenius error.
    """
    if x.shape != reference.shape:
        raise ValueError("Trajectory and reference must have identical shapes.")

    x_mean = x.mean(axis=0, keepdims=True)
    ref_mean = reference.mean(axis=0, keepdims=True)
    xc = x - x_mean
    rc = reference - ref_mean

    # Solve min_R || xc R - rc ||_F subject to R^T R = I.
    u, _, vt = np.linalg.svd(xc.T @ rc, full_matrices=False)
    rotation = u @ vt
    aligned_centered = xc @ rotation

    # Place the aligned trajectory at the reference centroid for visualization.
    aligned = aligned_centered + ref_mean

    residual = np.linalg.norm(aligned_centered - rc, ord="fro")
    denom = np.linalg.norm(rc, ord="fro")
    relative_error = float(residual / denom) if denom > 0 else float("nan")

    return aligned, rotation, relative_error


def procrustes_analysis(
    frames: Dict[int, pd.DataFrame], model_type: str, reference_seed: int
) -> Tuple[pd.DataFrame, Dict[int, np.ndarray], np.ndarray, np.ndarray]:
    """Align every seed to a chosen reference seed in the full latent space."""
    steps, zcols = validate_common_steps(frames)
    if reference_seed not in frames:
        raise ValueError(f"Reference seed {reference_seed} is missing for {model_type}.")

    reference = frames[reference_seed][zcols].to_numpy(dtype=float)
    aligned: Dict[int, np.ndarray] = {}
    rows = []

    for seed, df in sorted(frames.items()):
        x = df[zcols].to_numpy(dtype=float)
        if seed == reference_seed:
            a = reference.copy()
            error = 0.0
            rotation = np.eye(reference.shape[1])
        else:
            a, rotation, error = orthogonal_procrustes_align(x, reference)
        aligned[seed] = a

        rows.append(
            {
                "model_type": model_type,
                "model_seed": seed,
                "reference_seed": reference_seed,
                "relative_procrustes_error": error,
                "det_rotation": float(np.linalg.det(rotation)),
            }
        )

    # PCA basis from the centered aligned reference trajectory only.
    ref_centered = reference - reference.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(ref_centered, full_matrices=False)
    basis_2d = vt[:2].T

    return pd.DataFrame(rows), aligned, steps, basis_2d


def project_aligned_to_reference_pca(
    aligned: Dict[int, np.ndarray], reference_seed: int, basis_2d: np.ndarray
) -> Dict[int, np.ndarray]:
    """Project all aligned trajectories into the same reference-defined 2D basis."""
    ref_mean = aligned[reference_seed].mean(axis=0, keepdims=True)
    return {
        seed: (coords - ref_mean) @ basis_2d
        for seed, coords in aligned.items()
    }


def plot_geometry_heatmap(matrix: pd.DataFrame, title: str, outfile: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    im = ax.imshow(matrix.to_numpy(dtype=float), vmin=0.0, vmax=1.0, aspect="equal")
    ax.set_xticks(range(len(matrix.columns)), labels=[str(x) for x in matrix.columns])
    ax.set_yticks(range(len(matrix.index)), labels=[str(x) for x in matrix.index])
    ax.set_xlabel("Model seed")
    ax.set_ylabel("Model seed")
    ax.set_title(title)

    for i in range(len(matrix.index)):
        for j in range(len(matrix.columns)):
            ax.text(j, i, f"{matrix.iloc[i, j]:.3f}", ha="center", va="center")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Pearson r of 8D distance geometry")
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_aligned_trajectories(
    projected: Dict[int, np.ndarray],
    steps: np.ndarray,
    reference_seed: int,
    title: str,
    outfile: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 6.0))

    for seed in sorted(projected):
        xy = projected[seed]
        label = f"seed {seed}" + (" (reference)" if seed == reference_seed else "")
        ax.plot(xy[:, 0], xy[:, 1], marker="o", markersize=3, linewidth=1.5, label=label)

    # Label temporal endpoints only, keeping the plot readable.
    ref_xy = projected[reference_seed]
    ax.annotate(str(steps[0]), (ref_xy[0, 0], ref_xy[0, 1]), xytext=(4, 4), textcoords="offset points")
    ax.annotate(str(steps[-1]), (ref_xy[-1, 0], ref_xy[-1, 1]), xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Reference PCA coordinate 1")
    ax.set_ylabel("Reference PCA coordinate 2")
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_summary(
    geometry_pairs: pd.DataFrame,
    temporal: pd.DataFrame,
    procrustes: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for model_type in sorted(geometry_pairs["model_type"].unique()):
        g = geometry_pairs[geometry_pairs["model_type"] == model_type]
        t = temporal[temporal["model_type"] == model_type]
        p = procrustes[
            (procrustes["model_type"] == model_type)
            & (procrustes["model_seed"] != procrustes["reference_seed"])
        ]

        rows.append(
            {
                "model_type": model_type,
                "n_initializations": int(t["model_seed"].nunique()),
                "mean_geometry_pearson_r": g["geometry_pearson_r"].mean(),
                "min_geometry_pearson_r": g["geometry_pearson_r"].min(),
                "max_geometry_pearson_r": g["geometry_pearson_r"].max(),
                "mean_geometry_spearman_rho": g["geometry_spearman_rho"].mean(),
                "mean_adjacent_nonadjacent_ratio": t["adjacent_nonadjacent_ratio"].mean(),
                "min_adjacent_nonadjacent_ratio": t["adjacent_nonadjacent_ratio"].min(),
                "max_adjacent_nonadjacent_ratio": t["adjacent_nonadjacent_ratio"].max(),
                "mean_temporal_distance_pearson_r": t["temporal_distance_pearson_r"].mean(),
                "mean_temporal_distance_spearman_rho": t["temporal_distance_spearman_rho"].mean(),
                "mean_relative_procrustes_error_vs_reference": p["relative_procrustes_error"].mean(),
                "min_relative_procrustes_error_vs_reference": p["relative_procrustes_error"].min(),
                "max_relative_procrustes_error_vs_reference": p["relative_procrustes_error"].max(),
            }
        )

    return pd.DataFrame(rows)


def load_model_group(base_dir: Path, prefix: str, seeds: Iterable[int]) -> Dict[int, pd.DataFrame]:
    frames = {}
    for seed in seeds:
        path = base_dir / f"{prefix}_seed_{seed}" / "latents.csv"
        frames[int(seed)] = load_latents(path)
    validate_common_steps(frames)
    return frames


def parse_args():
    p = argparse.ArgumentParser(
        description="Quantify robustness of Evoscope autoencoder latent trajectories to independent model initialization."
    )
    p.add_argument(
        "--base_dir",
        type=Path,
        required=True,
        help="Directory containing global_seed_N/ and cluster_seed_N/ folders.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5],
        help="Model initialization seeds to compare (default: 1 2 3 4 5).",
    )
    p.add_argument(
        "--reference_seed",
        type=int,
        default=1,
        help="Reference initialization for orthogonal Procrustes alignment (default: 1).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        required=True,
        help="Output directory for CSV tables and figures.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    groups = {
        "global": load_model_group(args.base_dir, "global", args.seeds),
        "cluster": load_model_group(args.base_dir, "cluster", args.seeds),
    }

    all_geometry = []
    all_temporal = []
    all_procrustes = []

    for model_type, frames in groups.items():
        geometry_pairs, geometry_matrix = geometry_correlations(frames, model_type)
        temporal = temporal_metrics(frames, model_type)
        procrustes, aligned, steps, basis_2d = procrustes_analysis(
            frames, model_type, args.reference_seed
        )
        projected = project_aligned_to_reference_pca(
            aligned, args.reference_seed, basis_2d
        )

        geometry_pairs.to_csv(
            args.outdir / f"{model_type}_geometry_pairwise.csv", index=False
        )
        geometry_matrix.to_csv(
            args.outdir / f"{model_type}_geometry_correlation_matrix.csv"
        )
        temporal.to_csv(
            args.outdir / f"{model_type}_temporal_metrics.csv", index=False
        )
        procrustes.to_csv(
            args.outdir / f"{model_type}_procrustes_metrics.csv", index=False
        )

        plot_geometry_heatmap(
            geometry_matrix,
            title=f"{model_type.capitalize()} AE: latent-distance geometry reproducibility",
            outfile=args.outdir / f"{model_type}_geometry_heatmap.png",
        )
        plot_aligned_trajectories(
            projected,
            steps,
            args.reference_seed,
            title=f"{model_type.capitalize()} AE: Procrustes-aligned latent trajectories",
            outfile=args.outdir / f"{model_type}_procrustes_aligned_trajectories.png",
        )

        # Export aligned coordinates in the common 2D reference-PCA space.
        projected_rows = []
        for seed, xy in sorted(projected.items()):
            for step, (pc1, pc2) in zip(steps, xy):
                projected_rows.append(
                    {
                        "model_type": model_type,
                        "model_seed": seed,
                        "step": step,
                        "reference_pc1": pc1,
                        "reference_pc2": pc2,
                    }
                )
        pd.DataFrame(projected_rows).to_csv(
            args.outdir / f"{model_type}_procrustes_projected_coordinates.csv",
            index=False,
        )

        all_geometry.append(geometry_pairs)
        all_temporal.append(temporal)
        all_procrustes.append(procrustes)

    geometry_all = pd.concat(all_geometry, ignore_index=True)
    temporal_all = pd.concat(all_temporal, ignore_index=True)
    procrustes_all = pd.concat(all_procrustes, ignore_index=True)

    geometry_all.to_csv(args.outdir / "geometry_pairwise_all.csv", index=False)
    temporal_all.to_csv(args.outdir / "temporal_metrics_all.csv", index=False)
    procrustes_all.to_csv(args.outdir / "procrustes_metrics_all.csv", index=False)

    summary = build_summary(geometry_all, temporal_all, procrustes_all)
    summary.to_csv(args.outdir / "robustness_summary.csv", index=False)

    # Human-readable text summary useful for manuscript/rebuttal drafting.
    with open(args.outdir / "robustness_summary.txt", "w", encoding="utf-8") as fh:
        fh.write("Evoscope autoencoder initialization robustness\n")
        fh.write("============================================\n\n")
        fh.write(f"Model seeds: {', '.join(map(str, args.seeds))}\n")
        fh.write(f"Procrustes reference seed: {args.reference_seed}\n\n")
        for _, row in summary.iterrows():
            fh.write(f"[{row['model_type']}]\n")
            fh.write(
                "Pairwise 8D distance-geometry Pearson r: "
                f"mean={row['mean_geometry_pearson_r']:.3f}, "
                f"range={row['min_geometry_pearson_r']:.3f}-"
                f"{row['max_geometry_pearson_r']:.3f}\n"
            )
            fh.write(
                "Adjacent/non-adjacent latent-distance ratio: "
                f"mean={row['mean_adjacent_nonadjacent_ratio']:.3f}, "
                f"range={row['min_adjacent_nonadjacent_ratio']:.3f}-"
                f"{row['max_adjacent_nonadjacent_ratio']:.3f}\n"
            )
            fh.write(
                "Temporal separation vs latent distance (Pearson r): "
                f"mean={row['mean_temporal_distance_pearson_r']:.3f}\n"
            )
            fh.write(
                "Relative orthogonal-Procrustes error vs reference: "
                f"mean={row['mean_relative_procrustes_error_vs_reference']:.3f}, "
                f"range={row['min_relative_procrustes_error_vs_reference']:.3f}-"
                f"{row['max_relative_procrustes_error_vs_reference']:.3f}\n\n"
            )

    print("Robustness analysis complete.")
    print(f"Outputs written to: {args.outdir}")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
