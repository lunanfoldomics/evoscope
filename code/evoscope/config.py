"""
Configuration parameters for the Evoscope simulation.

This module defines the `Config` dataclass, which centralizes all tunable
parameters controlling the spatial world, nutrient field, energy dynamics,
cell-cycle progression, gene/protein regulation, commitment behavior,
movement, division, attack behavior, and snapshot generation.

The purpose of this module is to keep the biological and physical assumptions
of the model explicit, editable, and reproducible. A `Config` instance is
passed to the main Evoscope simulation engine and determines the initial
conditions and dynamical rules of each run.

Evoscope v0.9.1
Author: Luca Zammataro
Organization: Lunan Foldomics LLC
"""

from dataclasses import dataclass


@dataclass
class Config:
    width: int = 30
    height: int = 20
    seed: int = 7

    # Initialization
    initial_cells: int = 45
    initial_medium_nutrient: float = 1.0
    nutrient_hotspots: int = 0

    # Nutrient dynamics
    diffusion_rate: float = 0.12
    nutrient_decay: float = 0.01
    corpse_release_fraction: float = 0.75

    # Energy dynamics
    energy_init_min: float = 3.0
    energy_init_max: float = 6.0
    energy_max: float = 20.0
    basal_cost: float = 0.08
    protein_cost_factor: float = 0.006
    movement_cost: float = 0.18
    attack_cost: float = 0.35
    division_cost: float = 0.9

    # Cell cycle thresholds
    s_entry_energy: float = 3.5
    m_entry_energy: float = 5.5
    divide_energy_min: float = 6.5
    starvation_threshold: float = 0.8
    critical_energy: float = 0.12
    max_divisions: int = 14
    max_age: int = 350
    critical_epochs_before_death: int = 5

    # Gene/protein dynamics
    synth_rate_base: float = 0.18
    degrade_rate: float = 0.10
    max_protein_level: float = 5.0

    # Commitment dynamics
    commitment_stability_epochs: int = 4
    decommit_stress_epochs: int = 8
    decommit_probability: float = 0.08

    # H bias map amplitude
    h_bias_strength: float = 0.55

    # Behavior scores
    move_empty_bonus: float = 1.5
    move_nutrient_weight: float = 0.9
    move_crowding_penalty: float = 0.35
    move_same_cluster_bonus: float = 0.25
    move_diff_cluster_penalty: float = 0.20

    divide_same_cluster_bonus: float = 0.50
    divide_nutrient_weight: float = 0.40
    divide_less_crowded_bonus: float = 0.25

    attack_threshold: float = 0.5
    mismatch_attack_bonus: float = 0.4

    # Directional polarization dynamics
    directional_noise: float = 0.03
    empty_polarization_bonus: float = 0.45
    occupied_polarization_bonus: float = 0.35
    same_cluster_adhesion_bonus: float = 0.55
    vulnerable_attack_bias: float = 0.60
    daughter_polarity_noise: float = 0.04

    snapshot_every: int = 1

    verbose: bool = False


