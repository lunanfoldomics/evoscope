"""
Evoscope package public API.

This module exposes the main simulation objects and utility functions used by
the Evoscope framework. It provides convenient top-level imports for creating
simulation configurations, running the Evoscope model, handling cell states,
and exporting grid states as numerical arrays.

The package is organized around a modular architecture:
configuration, state definitions, grid topology, simulation dynamics,
input/output utilities, visualization, latent-space analysis, and
autoencoder-based morphology learning.

Evoscope — minimal regulatory spatial model for emergent multicellular organization.

Author: Luca Zammataro
Organization: Lunan Foldomics LLC
"""

__version__ = "0.9.2"
__author__ = "Luca Zammataro"
__organization__ = "Lunan Foldomics LLC"

from .config import Config
from .simulation import Evoscope
from .state import Cell, PlannedAction
from .enums import ActionType, CellCyclePhase, CommitmentState
from .utils import grid_to_numpy

__all__ = [
    "Config",
    "Evoscope",
    "Cell",
    "PlannedAction",
    "ActionType",
    "CellCyclePhase",
    "CommitmentState",
    "grid_to_numpy",
]
