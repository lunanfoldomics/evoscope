"""
General utility functions for Evoscope.

This module contains small helper functions used across the simulation,
including numerical clamping, extraction of total protein values from scalar
or directional representations, directional protein lookup, and conversion of
a live simulation state into a 2D NumPy grid.

The exported grid representation encodes empty sites, undetermined cells, and
committed cluster identities, and is used for snapshot export, visualization,
and downstream machine-learning analyses.


Evoscope — minimal regulatory spatial model for emergent multicellular organization.

Author: Luca Zammataro
Organization: Lunan Foldomics LLC
"""

from typing import List, Union
import numpy as np


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def protein_total(value: Union[float, List[float]]) -> float:
    return float(sum(value)) if isinstance(value, list) else float(value)


def directional_value(value: Union[float, List[float]], direction_idx: int) -> float:
    if isinstance(value, list):
        return float(value[direction_idx])
    return float(value)


def grid_to_numpy(sim):
    """Convert the current simulation state into a 2D integer array.

    Values:
    -1 = empty site
     0 = occupied by an undetermined/decommitted cell
     1..8 = committed cluster IDs 0..7 shifted by +1
    """
    grid = np.full((sim.cfg.height, sim.cfg.width), -1)

    for (q, r), cell in sim.occupancy.items():
        if cell.cluster_id is None:
            grid[r, q] = 0
        else:
            grid[r, q] = cell.cluster_id + 1

    return grid
