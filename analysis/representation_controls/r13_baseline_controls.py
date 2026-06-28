#!/usr/bin/env python3
"""
R1.3 baseline-control analyses for Evoscope representation learning.

This script compares simple baseline models for predicting global regulatory
variables from:

1. Coarse population-level covariates.
2. PCA morphology representations.
3. Temporal persistence/autocorrelation baselines.

Expected input layout:

runs/
  seed_38/
    global_genes.csv
    population_metrics.csv
    snapshots/
      grid_001.npy
      ...
      grid_120.npy
  seed_40/
    ...

Outputs:

r13_outputs/
  r13_baseline_comparison.csv
  r13_predictions.csv

Usage:

python r13_baseline_controls.py --root runs --outdir outputs  

"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


GENES = ["T1", "T2", "I", "R", "M", "K", "S"]

DEFAULT_TRAIN_SEEDS = [38, 40, 53, 65, 89, 90]
DEFAULT_VAL_SEEDS = [96, 101]
DEFAULT_TEST_SEEDS = [104, 107]


def parse_seed_from_dir(path: Path) -> int:
    match = re.search(r"seed_(\d+)$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse seed from directory name: {path}")
    return int(match.group(1))


def snapshot_step_from_name(path: Path) -> int:
    match = re.search(r"grid_(\d+)\.npy$", path.name)
    if match is None:
        raise ValueError(f"Cannot parse snapshot step from filename: {path}")
    return int(match.group(1))


def pearson_per_target(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Return mean Pearson correlation across target dimensions."""
    correlations = []

    for j in range(y_true.shape[1]):
        a = y_true[:, j]
        b = y_pred[:, j]

        if np.std(a) == 0 or np.std(b) == 0:
            continue

        correlations.append(np.corrcoef(a, b)[0, 1])

    if not correlations:
        return np.nan

    return float(np.mean(correlations))


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": float(r2_score(y_true, y_pred, multioutput="variance_weighted")),
        "Pearson_mean": pearson_per_target(y_true, y_pred),
    }


def discover_run_dirs(root: Path) -> List[Path]:
    run_dirs = sorted([p for p in root.iterdir() if p.is_dir() and p.name.startswith("seed_")])
    if not run_dirs:
        raise FileNotFoundError(f"No seed_* directories found under {root}")
    return run_dirs


def load_covariate_dataset(root: Path) -> pd.DataFrame:
    rows = []

    for run_dir in discover_run_dirs(root):
        seed = parse_seed_from_dir(run_dir)

        global_path = run_dir / "global_genes.csv"
        pop_path = run_dir / "population_metrics.csv"

        if not global_path.exists():
            raise FileNotFoundError(global_path)
        if not pop_path.exists():
            raise FileNotFoundError(pop_path)

        global_df = pd.read_csv(global_path)
        pop_df = pd.read_csv(pop_path)

        merged = pd.merge(pop_df, global_df, on="step", how="inner")
        merged["seed"] = seed
        rows.append(merged)

    return pd.concat(rows, ignore_index=True)


def load_morphology_dataset(root: Path) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load flattened morphology snapshots and aligned global targets.

    Evoscope currently exports snapshots as grid_001.npy ... grid_120.npy,
    whereas global_genes.csv stores step 0 ... 119. We therefore align:

        grid_001.npy -> step 0
        grid_002.npy -> step 1
        ...
        grid_120.npy -> step 119

    This offset should be kept explicit in the Methods.
    """

    X_rows = []
    meta_rows = []

    for run_dir in discover_run_dirs(root):
        seed = parse_seed_from_dir(run_dir)

        global_path = run_dir / "global_genes.csv"
        snapshots_dir = run_dir / "snapshots"

        if not global_path.exists():
            raise FileNotFoundError(global_path)
        if not snapshots_dir.exists():
            raise FileNotFoundError(snapshots_dir)

        global_df = pd.read_csv(global_path)
        global_by_step = global_df.set_index("step")

        snapshot_files = sorted(snapshots_dir.glob("grid_*.npy"))

        if not snapshot_files:
            raise FileNotFoundError(f"No grid_*.npy snapshots found in {snapshots_dir}")

        for snapshot_path in snapshot_files:
            exported_snapshot_step = snapshot_step_from_name(snapshot_path)
            csv_step = exported_snapshot_step - 1

            if csv_step not in global_by_step.index:
                continue

            grid = np.load(snapshot_path)
            X_rows.append(grid.reshape(-1).astype(float))

            target_values = global_by_step.loc[csv_step, GENES].to_dict()

            meta_row = {
                "seed": seed,
                "step": int(csv_step),
                "snapshot_file": str(snapshot_path),
            }
            meta_row.update(target_values)
            meta_rows.append(meta_row)

    X = np.vstack(X_rows)
    meta = pd.DataFrame(meta_rows)

    return X, meta


def split_mask(df: pd.DataFrame, seeds: List[int]) -> np.ndarray:
    return df["seed"].isin(seeds).to_numpy()


def fit_predict_ridge(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    alpha: float = 1.0,
) -> np.ndarray:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=alpha)),
        ]
    )
    model.fit(X_train, y_train)
    return model.predict(X_test)


def run_covariate_baseline(
    df: pd.DataFrame,
    train_seeds: List[int],
    test_seeds: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    covariate_cols = [
        "step",
        "total_cells",
        "empty_sites",
        "occupied_density",
        "committed_cells",
        "committed_density",
        "undetermined_cells",
        "undetermined_density",
        "decommitted_cells",
        "decommitted_density",
        "uncommitted_cells",
        "uncommitted_density",
        "n_clusters_present",
        "largest_cluster",
    ] + [f"cluster_{cid}_size" for cid in range(8)]

    train_mask = split_mask(df, train_seeds)
    test_mask = split_mask(df, test_seeds)

    X_train = df.loc[train_mask, covariate_cols].to_numpy(dtype=float)
    y_train = df.loc[train_mask, GENES].to_numpy(dtype=float)

    X_test = df.loc[test_mask, covariate_cols].to_numpy(dtype=float)
    y_test = df.loc[test_mask, GENES].to_numpy(dtype=float)

    y_pred = fit_predict_ridge(X_train, y_train, X_test)

    metrics = regression_metrics(y_test, y_pred)
    metrics_row = {
        "model": "coarse_covariates_ridge",
        "input": "population_metrics",
        "target": "global_regulatory_variables",
        "split": "held_out_seed",
        **metrics,
    }

    pred_df = df.loc[test_mask, ["seed", "step"]].copy()
    for j, gene in enumerate(GENES):
        pred_df[f"{gene}_true"] = y_test[:, j]
        pred_df[f"{gene}_pred"] = y_pred[:, j]
    pred_df["model"] = "coarse_covariates_ridge"

    return pd.DataFrame([metrics_row]), pred_df


def run_pca_baseline(
    X: np.ndarray,
    meta: pd.DataFrame,
    train_seeds: List[int],
    test_seeds: List[int],
    n_components: int = 8,
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    train_mask = split_mask(meta, train_seeds)
    test_mask = split_mask(meta, test_seeds)

    X_train_raw = X[train_mask]
    X_test_raw = X[test_mask]

    y_train = meta.loc[train_mask, GENES].to_numpy(dtype=float)
    y_test = meta.loc[test_mask, GENES].to_numpy(dtype=float)

    pca_model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=0)),
        ]
    )

    X_train_pca = pca_model.fit_transform(X_train_raw)
    X_test_pca = pca_model.transform(X_test_raw)

    y_pred = fit_predict_ridge(X_train_pca, y_train, X_test_pca)

    metrics = regression_metrics(y_test, y_pred)
    explained = pca_model.named_steps["pca"].explained_variance_ratio_.sum()

    metrics_row = {
        "model": "pca8_morphology_ridge",
        "input": "flattened_morphology_PCA8",
        "target": "global_regulatory_variables",
        "split": "held_out_seed",
        "PCA_explained_variance": float(explained),
        **metrics,
    }

    pred_df = meta.loc[test_mask, ["seed", "step", "snapshot_file"]].copy()
    for j, gene in enumerate(GENES):
        pred_df[f"{gene}_true"] = y_test[:, j]
        pred_df[f"{gene}_pred"] = y_pred[:, j]
    pred_df["model"] = "pca8_morphology_ridge"

    return pd.DataFrame([metrics_row]), pred_df


def run_persistence_baseline(
    df: pd.DataFrame,
    test_seeds: List[int],
    horizons: List[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:

    metric_rows = []
    pred_rows = []

    test_df = df[df["seed"].isin(test_seeds)].copy()

    for horizon in horizons:
        y_true_all = []
        y_pred_all = []
        meta_all = []

        for seed, seed_df in test_df.groupby("seed"):
            seed_df = seed_df.sort_values("step").reset_index(drop=True)

            for i in range(len(seed_df) - horizon):
                current = seed_df.iloc[i]
                future = seed_df.iloc[i + horizon]

                y_pred_all.append(current[GENES].to_numpy(dtype=float))
                y_true_all.append(future[GENES].to_numpy(dtype=float))

                meta_all.append(
                    {
                        "seed": int(seed),
                        "step": int(current["step"]),
                        "future_step": int(future["step"]),
                        "horizon": int(horizon),
                    }
                )

        y_true = np.vstack(y_true_all)
        y_pred = np.vstack(y_pred_all)

        metrics = regression_metrics(y_true, y_pred)

        metric_rows.append(
            {
                "model": f"persistence_t_plus_{horizon}",
                "input": "current_regulatory_state",
                "target": "future_global_regulatory_variables",
                "split": "held_out_seed",
                "horizon": int(horizon),
                **metrics,
            }
        )

        pred_df = pd.DataFrame(meta_all)
        for j, gene in enumerate(GENES):
            pred_df[f"{gene}_true"] = y_true[:, j]
            pred_df[f"{gene}_pred"] = y_pred[:, j]
        pred_df["model"] = f"persistence_t_plus_{horizon}"
        pred_rows.append(pred_df)

    return pd.DataFrame(metric_rows), pd.concat(pred_rows, ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run R1.3 Evoscope baseline controls.")
    parser.add_argument("--root", type=str, default="runs", help="Root directory containing seed_* runs.")
    parser.add_argument("--outdir", type=str, default="r13_outputs", help="Output directory.")
    parser.add_argument("--train-seeds", type=int, nargs="+", default=DEFAULT_TRAIN_SEEDS)
    parser.add_argument("--val-seeds", type=int, nargs="+", default=DEFAULT_VAL_SEEDS)
    parser.add_argument("--test-seeds", type=int, nargs="+", default=DEFAULT_TEST_SEEDS)
    parser.add_argument("--pca-components", type=int, default=8)
    parser.add_argument("--persistence-horizons", type=int, nargs="+", default=[1, 5])
    args = parser.parse_args()

    root = Path(args.root)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print("Loading covariate dataset...")
    covariate_df = load_covariate_dataset(root)
    covariate_df.to_csv(outdir / "r13_merged_covariate_dataset.csv", index=False)

    print("Loading morphology snapshots...")
    X_morph, morph_meta = load_morphology_dataset(root)
    morph_meta.to_csv(outdir / "r13_morphology_metadata.csv", index=False)

    all_metrics = []
    all_predictions = []

    print("Running coarse covariate baseline...")
    metrics, predictions = run_covariate_baseline(
        df=covariate_df,
        train_seeds=args.train_seeds,
        test_seeds=args.test_seeds,
    )
    all_metrics.append(metrics)
    all_predictions.append(predictions)

    print("Running PCA morphology baseline...")
    metrics, predictions = run_pca_baseline(
        X=X_morph,
        meta=morph_meta,
        train_seeds=args.train_seeds,
        test_seeds=args.test_seeds,
        n_components=args.pca_components,
    )
    all_metrics.append(metrics)
    all_predictions.append(predictions)

    print("Running persistence baselines...")
    metrics, predictions = run_persistence_baseline(
        df=covariate_df,
        test_seeds=args.test_seeds,
        horizons=args.persistence_horizons,
    )
    all_metrics.append(metrics)
    all_predictions.append(predictions)

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    predictions_df = pd.concat(all_predictions, ignore_index=True, sort=False)

    metrics_path = outdir / "r13_baseline_comparison.csv"
    predictions_path = outdir / "r13_predictions.csv"

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)

    print(f"Saved metrics to: {metrics_path}")
    print(f"Saved predictions to: {predictions_path}")

    print("\nSummary:")
    print(metrics_df)


if __name__ == "__main__":
    main()
