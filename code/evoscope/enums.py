"""
Enumerated state definitions for Evoscope cells and actions.

This module defines symbolic labels for the discrete states used throughout
the simulation, including cell-cycle phases, commitment states, and planned
cellular actions. These enums make the simulation logic more readable and
reduce ambiguity compared with using raw strings or integer codes.

Main enums
----------
CellCyclePhase
    Defines the cell-cycle state of a simulated cell.

CommitmentState
    Defines whether a cell is undetermined, committed, or decommitted.

ActionType
    Defines the possible actions selected during the planning phase of each
    simulation step.

Evoscope v0.9.1
Author: Luca Zammataro
Organization: Lunan Foldomics LLC
"""

from enum import Enum, auto


class CellCyclePhase(Enum):
    G = auto()
    S = auto()
    M = auto()
    STALL = auto()


class CommitmentState(Enum):
    UNDETERMINED = auto()
    COMMITTED = auto()
    DECOMMITTED = auto()


class ActionType(Enum):
    NONE = auto()
    REST = auto()
    MOVE = auto()
    DIVIDE = auto()
    ATTACK_AND_DIVIDE = auto()
