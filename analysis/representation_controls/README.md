
# Representation-control analyses


This directory contains the scripts used to reproduce the held-out seed, baseline-control, and latent temporal prediction analyses reported in the revised manuscript and Supplementary Tables S1 and S2.


The analyses compare morphology-derived representations with simpler baseline models, including:

- coarse population-level covariates + ridge regression
- PCA8 morphology representation + ridge regression
- autoencoder prediction head
- autoencoder latent z8 + ridge regression
- temporal persistence baselines at t+1 and t+5
- latent temporal prediction controls at t+1, t+5, and t+10



## Expected input structure

The scripts expect simulation outputs organized as in the `examples/` directory, which contains random-seed simulation examples:

```text
examples/
└── runs/
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
```


The held-out seed split used in the revised manuscript is:


```text
Train:      38, 40, 53, 65, 89, 90
Validation: 96, 101
Test:       104, 107
```



## Outputs


The scripts generate CSV files summarizing held-out performance using MAE, RMSE, R², and mean Pearson correlation across the seven global regulatory variables.




## Examples

```bash

python analysis/representation_controls/r13_baseline_controls.py  \
  --root examples/runs \
  --outdir examples/analysis/representation_controls/outputs/


python analysis/representation_controls/r13_autoencoder_compare.py \
  --root examples/runs \
  --outdir examples/analysis/representation_controls/outputs \
  --device auto


python analysis/representation_controls/r22_latent_temporal_prediction.py \
  --runs_dir examples/runs  \
  --output_dir examples/analysis/representation_controls/outputs
 
```


## Manuscript reference outputs


The output files used for the revised manuscript and Supplementary Information are provided in this directory:


`analysis/representation_controls/`


These committed outputs correspond to Supplementary Tables S1 and S2 and should be considered the manuscript reference results.

Because autoencoder training involves stochastic initialization and backend-dependent numerical variation, re-running the analyses may produce minor numerical differences, especially for the autoencoder prediction-head and autoencoder latent z8 + ridge outputs. Newly generated outputs should therefore be interpreted as reproducibility checks rather than exact byte-for-byte reproductions of the Supplementary Information.
