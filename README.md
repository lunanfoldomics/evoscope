<img src="images/evoscope_logo.png" alt="logo" style="display:block; margin:auto; width:1000px; height:auto;">

# Evoscope
A minimal agent-based multicellular simulation for emergent morphology, differentiation, and collective organization in dissipative environments.

---

**Evoscope** is a minimal agent-based multicellular simulation designed to study how **mesoscopic organization** can emerge from compact intracellular regulatory rules in a **dissipative environment**.

The model combines:

- heritable identity commitment
- nutrient-dependent regulation
- adhesion, motility, competition, and protection
- local killing and space clearing
- emergent multicellular morphologies
- morphology-to-state inference through autoencoder-based latent representations

Rather than modeling a specific organism, Evoscope provides a **minimal regulatory sandbox** for exploring how structured multicellular behaviors can arise from interpretable local rules.

---
<img src="images/evoscope_run_001_58.png" alt="run" style="display:block; margin:auto; width:300px; height:auto;">

## Concept

Evoscope was developed to investigate a specific question:

> Can a minimal multicellular system generate a biologically interpretable **mesoscopic level of organization** linking intracellular regulatory programs to transient colony-level structure?

The simulation represents cells as agents on a **toroidal hexagonal lattice** coupled to a shared nutrient field. Each cell carries a minimal internal regulatory unit (the **genomoid**) whose state controls behavior such as:

- nutrient uptake
- adhesion / persistence
- motility
- competitive killing
- protection from attack
- commitment to heritable cluster identities

These interactions generate structured colony dynamics including:

- aggregate formation
- territorial expansion
- competition for space and resources
- transient ecological niches
- population collapse under energetic constraints

---

## Main features

- 2D toroidal hexagonal grid
- one cell per lattice site
- diffusive extracellular nutrient field
- intracellular energy bookkeeping
- minimal gene/protein regulatory system
- stochastic commitment to cluster identity
- division, movement, attack, death, and nutrient recycling
- ASCII rendering of simulation states
- exportable outputs for downstream quantitative analysis
- support for morphology-to-state learning with convolutional autoencoders

---

## Current model variables

The current framework uses a compact set of functional variables:

- **T1**: activator / nutrient-responsive regulator
- **T2**: repressor / stress-crowding regulator
- **I**: adhesion / integrin-like persistence program
- **R**: nutrient uptake / receptor program
- **M**: motility program
- **K**: competitive killing program
- **S**: protection / survival program
- **H1-H3**: binary identity variables defining heritable cluster states

These variables do **not** represent specific molecular pathways directly. They are intended as **minimal functional abstractions**.

---

## Installation

Clone the repository and install the required Python packages.

```Bash
git clone https://github.com/<your-username>/evoscope.git
cd evoscope
pip install -r requirements.txt
```

A minimal requirements.txt can include:

```Bash
numpy
matplotlib
pandas
torch
```

If you only want to run the simulator and not the autoencoder analysis, torch is optional.

## Running the simulation

```Bash
python evoscope.py
```

If argument parsing is enabled in your current version, a typical command may look like:

```Bash
python evoscope.py --width 60 --height 40 --seed 42 --epochs 150 --initial_cells 30 --nutrient 6.9
```

Outputs may include:

- console summaries
- ASCII frame dumps
- snapshot arrays
- global gene trajectories
- cluster-resolved gene trajectories

### Example output

Typical simulation dynamics include:

-sparse exploratory cells
-early nucleation of committed clusters
-coexistence of multiple multicellular domains
-competitive reshaping of territorial interfaces
-fragmentation and collapse under energetic stress


The system is intentionally minimal, but it often produces visually rich and interpretable multicellular behaviors.



### Multi-run aggregation

The repository may include utilities for aggregating results across multiple seeds.

For example:

```Bash
python aggregate_evoscope_fig4.py --root runs --outdir aggregated_outputs
```

This can be used to compute:

- mean ± standard deviation of global gene trajectories
- mean ± standard deviation of cluster-resolved gene trajectories
- multi-experiment summary figures

## Representation learning

Evoscope also supports a morphology-to-state workflow in which simulation snapshots are used to train convolutional autoencoders with supervised prediction heads.

The goal is to test whether:

- visible morphology contains recoverable information about hidden internal states
- latent variables can serve as mesoscopic coordinates linking internal regulatory dynamics to emergent multicellular form

This part of the project is intended as a proof of concept for broader applications in:

- digital pathology
- spatial transcriptomics
- synthetic morphogenesis
- interpretable biological representation learning


## Scientific context

Evoscope builds on established traditions in:

- agent-based multicellular modeling
- minimal self-organizing systems
- dissipative biological organization
- artificial life
- latent representation learning

Its contribution is not to reproduce full biochemical realism, but to provide an original minimal integration of:

- regulatory commitment
- ecological interaction
- dissipative multicellular dynamics
- morphology-to-state inference

tailored to the mesoscopic question.

## Status

This project is currently under active development.

At present, Evoscope should be viewed as:

- a research prototype
- a conceptual and computational sandbox
- a platform for testing mesoscopic hypotheses

rather than a calibrated model of any specific tissue or organism.

## Preprint

A preprint describing the framework is planned / available as:

Luca Zammataro. A Minimal Regulatory Spatial Model for Emergent Multicellular Organization in Dissipative Environments.

(A bioRxiv entry will be added here after public release soon).

## Citation

If you use this code, please cite the associated preprint once available.

(A BibTeX entry will be added here after public release soon).

## License

MIT

Add the corresponding LICENSE file before making the repository public.

## Contact

Luca Zammataro
lucazammataro@lunanfoldomicsllc.com  - Lunan Foldomics LLC - github.com/lunanfoldomics
