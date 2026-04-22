import argparse
import os
import re
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split


# ============================================================
# Torex / Evoscope autoencoder
# ------------------------------------------------------------
# Learns a latent representation of morphology snapshots and,
# optionally, predicts associated gene expression profiles.
#
# Supported targets:
#   - global: 7 genes from global_genes.csv
#   - cluster_flat: 8 clusters x 7 genes = 56 values from cluster_genes.csv
#
# Recommended first run:
#   python torex_autoencoder.py \
#       --snapshots_dir snapshots \
#       --global_csv global_genes.csv \
#       --target_mode global \
#       --epochs 100
#
# "Deep" version:
#   python torex_autoencoder.py \
#       --snapshots_dir snapshots \
#       --cluster_csv cluster_genes.csv \
#       --target_mode cluster_flat \
#       --epochs 100
# ============================================================

GRID_VALUE_TO_CLASS = {
    -1: 0,  # empty
    0: 1,   # undetermined
    1: 2,
    2: 3,
    3: 4,
    4: 5,
    5: 6,
    6: 7,
    7: 8,
    8: 9,
}
N_CLASSES = 10
GLOBAL_GENES = ["T1", "T2", "I", "R", "M", "K", "S"]
CLUSTERS = list(range(8))


@dataclass
class SampleRecord:
    step: int
    snapshot_path: str
    target: np.ndarray


def parse_step_from_snapshot(path: str) -> int:
    m = re.search(r"grid_(\d+)\.npy$", os.path.basename(path))
    if not m:
        raise ValueError(f"Cannot parse step from snapshot path: {path}")
    return int(m.group(1))


def list_snapshot_files(snapshots_dir: str) -> List[Tuple[int, str]]:
    files = []
    for name in os.listdir(snapshots_dir):
        if not name.endswith(".npy") or not name.startswith("grid_"):
            continue
        full = os.path.join(snapshots_dir, name)
        step = parse_step_from_snapshot(full)
        files.append((step, full))
    files.sort(key=lambda x: x[0])
    return files


def load_global_targets(global_csv: str) -> Dict[int, np.ndarray]:
    df = pd.read_csv(global_csv)
    required = {"step", *GLOBAL_GENES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {global_csv}: {sorted(missing)}")

    targets: Dict[int, np.ndarray] = {}
    for _, row in df.iterrows():
        step = int(row["step"])
        targets[step] = row[GLOBAL_GENES].to_numpy(dtype=np.float32)
    return targets


def load_cluster_targets(cluster_csv: str) -> Dict[int, np.ndarray]:
    df = pd.read_csv(cluster_csv)
    required = {"step", "cluster", *GLOBAL_GENES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns in {cluster_csv}: {sorted(missing)}")

    targets: Dict[int, np.ndarray] = {}
    for step, sdf in df.groupby("step"):
        arr = np.zeros((8, len(GLOBAL_GENES)), dtype=np.float32)
        for _, row in sdf.iterrows():
            cid = int(row["cluster"])
            if cid < 0 or cid > 7:
                continue
            arr[cid, :] = row[GLOBAL_GENES].to_numpy(dtype=np.float32)
        targets[int(step)] = arr.reshape(-1)
    return targets


class TorexDataset(Dataset):
    def __init__(
        self,
        snapshots_dir: str,
        target_mode: str,
        global_csv: Optional[str] = None,
        cluster_csv: Optional[str] = None,
    ):
        if target_mode not in {"global", "cluster_flat"}:
            raise ValueError("target_mode must be 'global' or 'cluster_flat'")

        snapshot_files = list_snapshot_files(snapshots_dir)
        if not snapshot_files:
            raise FileNotFoundError(f"No .npy snapshots found in {snapshots_dir}")

        if target_mode == "global":
            if not global_csv:
                raise ValueError("global_csv is required for target_mode='global'")
            target_map = load_global_targets(global_csv)
            self.target_dim = len(GLOBAL_GENES)
        else:
            if not cluster_csv:
                raise ValueError("cluster_csv is required for target_mode='cluster_flat'")
            target_map = load_cluster_targets(cluster_csv)
            self.target_dim = 8 * len(GLOBAL_GENES)

        records: List[SampleRecord] = []
        missing_steps: List[int] = []
        for step, path in snapshot_files:
            if step not in target_map:
                missing_steps.append(step)
                continue
            records.append(SampleRecord(step=step, snapshot_path=path, target=target_map[step]))

        if not records:
            raise RuntimeError("No aligned snapshot/target pairs found")

        self.records = records
        self.missing_steps = missing_steps

        # infer shape from first snapshot
        first_grid = np.load(self.records[0].snapshot_path)
        if first_grid.ndim != 2:
            raise ValueError("Snapshots must be 2D arrays")
        self.height, self.width = first_grid.shape

    def __len__(self) -> int:
        return len(self.records)

    def _grid_to_class_map(self, grid: np.ndarray) -> np.ndarray:
        out = np.zeros_like(grid, dtype=np.int64)
        for raw_val, cls in GRID_VALUE_TO_CLASS.items():
            out[grid == raw_val] = cls
        return out

    def _class_map_to_onehot(self, cls_map: np.ndarray) -> np.ndarray:
        onehot = np.eye(N_CLASSES, dtype=np.float32)[cls_map]  # H, W, C
        return np.transpose(onehot, (2, 0, 1))  # C, H, W

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        grid = np.load(rec.snapshot_path)
        cls_map = self._grid_to_class_map(grid)
        x = self._class_map_to_onehot(cls_map)
        y = rec.target.astype(np.float32)
        return {
            "x": torch.from_numpy(x),
            "classes": torch.from_numpy(cls_map),
            "y": torch.from_numpy(y),
            "step": torch.tensor(rec.step, dtype=torch.int64),
        }


class ConvAutoencoder(nn.Module):
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
            self.enc_shape = enc.shape[1:]  # C, H', W'
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
        z = self.to_latent(h)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.from_latent(z)
        h = h.view(-1, *self.enc_shape)
        recon_logits = self.decoder(h)
        recon_logits = recon_logits[:, :, : self.height, : self.width]
        return recon_logits

    def forward(self, x: torch.Tensor):
        z = self.encode(x)
        recon_logits = self.decode(z)
        gene_pred = self.gene_head(z)
        return recon_logits, gene_pred, z


class EarlyStopper:
    def __init__(self, patience: int = 20):
        self.patience = patience
        self.best = None
        self.count = 0

    def step(self, value: float) -> bool:
        if self.best is None or value < self.best:
            self.best = value
            self.count = 0
            return False
        self.count += 1
        return self.count >= self.patience


def train_one_epoch(model, loader, optimizer, device, recon_weight: float, gene_weight: float):
    model.train()
    recon_loss_fn = nn.CrossEntropyLoss()
    gene_loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_recon = 0.0
    total_gene = 0.0
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
def evaluate(model, loader, device, recon_weight: float, gene_weight: float):
    model.eval()
    recon_loss_fn = nn.CrossEntropyLoss()
    gene_loss_fn = nn.MSELoss()

    total_loss = 0.0
    total_recon = 0.0
    total_gene = 0.0
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


@torch.no_grad()
def export_latents(model, dataset, out_csv: str, device: str):
    model.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    for batch in loader:
        x = batch["x"].to(device)
        steps = batch["step"].cpu().numpy()
        _, _, z = model(x)
        z = z.cpu().numpy()
        for step, zi in zip(steps, z):
            row = {"step": int(step)}
            for i, val in enumerate(zi):
                row[f"z{i+1}"] = float(val)
            rows.append(row)

    pd.DataFrame(rows).sort_values("step").to_csv(out_csv, index=False)


@torch.no_grad()
def export_predictions(model, dataset, out_csv: str, device: str, target_mode: str):
    model.eval()
    rows = []
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    for batch in loader:
        x = batch["x"].to(device)
        steps = batch["step"].cpu().numpy()
        y_true = batch["y"].cpu().numpy()
        _, y_pred, _ = model(x)
        y_pred = y_pred.cpu().numpy()

        for step, yt, yp in zip(steps, y_true, y_pred):
            row = {"step": int(step)}
            if target_mode == "global":
                for i, g in enumerate(GLOBAL_GENES):
                    row[f"true_{g}"] = float(yt[i])
                    row[f"pred_{g}"] = float(yp[i])
            else:
                k = 0
                for cid in CLUSTERS:
                    for g in GLOBAL_GENES:
                        row[f"true_c{cid}_{g}"] = float(yt[k])
                        row[f"pred_c{cid}_{g}"] = float(yp[k])
                        k += 1
            rows.append(row)

    pd.DataFrame(rows).sort_values("step").to_csv(out_csv, index=False)


def parse_args():
    p = argparse.ArgumentParser(description="Train a Torex morphology autoencoder")
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
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--outdir", type=str, default="ae_outputs")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dataset = TorexDataset(
        snapshots_dir=args.snapshots_dir,
        target_mode=args.target_mode,
        global_csv=args.global_csv,
        cluster_csv=args.cluster_csv,
    )

    n_total = len(dataset)
    n_val = max(1, int(n_total * args.val_fraction))
    n_train = max(1, n_total - n_val)
    train_set, val_set = random_split(
        dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(args.seed),
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = ConvAutoencoder(
        height=dataset.height,
        width=dataset.width,
        latent_dim=args.latent_dim,
        target_dim=dataset.target_dim,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    stopper = EarlyStopper(patience=20)

    history = []
    best_state = None
    best_val = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args.recon_weight,
            args.gene_weight,
        )
        val_metrics = evaluate(
            model,
            val_loader,
            device,
            args.recon_weight,
            args.gene_weight,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_recon": train_metrics["recon"],
            "train_gene": train_metrics["gene"],
            "val_loss": val_metrics["loss"],
            "val_recon": val_metrics["recon"],
            "val_gene": val_metrics["gene"],
        }
        history.append(row)

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.4f} "
            f"(recon {train_metrics['recon']:.4f}, gene {train_metrics['gene']:.4f}) | "
            f"val loss {val_metrics['loss']:.4f} "
            f"(recon {val_metrics['recon']:.4f}, gene {val_metrics['gene']:.4f})"
        )

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if stopper.step(val_metrics["loss"]):
            print("Early stopping triggered.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), os.path.join(args.outdir, "torex_autoencoder.pt"))
    pd.DataFrame(history).to_csv(os.path.join(args.outdir, "training_history.csv"), index=False)
    export_latents(model, dataset, os.path.join(args.outdir, "latents.csv"), device)
    export_predictions(
        model,
        dataset,
        os.path.join(args.outdir, "predictions.csv"),
        device,
        args.target_mode,
    )

    print(f"Saved model and outputs to: {args.outdir}")


if __name__ == "__main__":
    main()
