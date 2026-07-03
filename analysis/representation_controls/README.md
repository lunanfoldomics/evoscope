
# Representation-control analyses

This directory contains the scripts used to reproduce the held-out seed and baseline-control analyses reported in the revised manuscript and Supplementary Table S1.

The analyses compare morphology-derived representations with simpler baseline models, including:

- coarse population-level covariates + ridge regression
- PCA8 morphology representation + ridge regression
- autoencoder prediction head
- autoencoder latent z8 + ridge regression
- temporal persistence baselines at t+1 and t+5

## Expected input structure

The scripts expect simulation outputs organized as:

```text
runs/
├── seed_38/
│   ├── global_genes.csv
│   ├── cluster_genes.csv
│   ├── population_metrics.csv
│   └── snapshots/
│       ├── grid_001.npy
│       ├── grid_002.npy
│       └── ...
├── seed_40/
│   └── ...
└── ...

The held-out seed split used in the revised manuscript is:

Train:      38, 40, 53, 65, 89, 90
Validation: 96, 101
Test:       104, 107
Outputs

The scripts generate CSV files summarizing held-out performance using MAE, RMSE, R², and mean Pearson correlation across the seven global regulatory variables.

These outputs correspond to the baseline-control results reported in Supplementary Table S1.
```

## Examples

```bash
python ../analysis/representation_controls/r13_baseline_controls.py --root runs --outdir analysis/representation_controls/outputs
python ../analysis/representation_controls/r13_autoencoder_compare.py --root runs --outdir analysis/representation_controls/outputs --device auto
python ../analysis/representation_controls/r22_latent_temporal_prediction.py --runs_dir runs  --output_dir analysis/representation_controls/outputs  
```
