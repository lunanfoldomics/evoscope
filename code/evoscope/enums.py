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
