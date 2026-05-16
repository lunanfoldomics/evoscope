from __future__ import annotations

from dataclasses import dataclass, field
import random
from typing import Dict, List, Optional, Tuple, Union

from .enums import ActionType, CellCyclePhase, CommitmentState


@dataclass
class PlannedAction:
    action_type: ActionType = ActionType.NONE
    source: Optional[Tuple[int, int]] = None
    target: Optional[Tuple[int, int]] = None
    attack_target: Optional[Tuple[int, int]] = None


@dataclass
class Cell:
    cell_id: int
    energy: float
    age: int = 0
    divisions_done: int = 0
    phase: CellCyclePhase = CellCyclePhase.G
    commitment: CommitmentState = CommitmentState.UNDETERMINED
    cluster_id: Optional[int] = None  # 0..7 if committed

    # Stress bookkeeping
    low_energy_epochs: int = 0
    stall_epochs: int = 0
    stress_epochs: int = 0

    # Commitment stabilization
    candidate_cluster: Optional[int] = None
    candidate_cluster_epochs: int = 0

    # Genomoid/proteins: 10 proteins / genes
    proteins: Dict[str, Union[float, List[float]]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.proteins:
            self.proteins = {
                "T1": random.uniform(0.2, 1.2),
                "T2": random.uniform(0.1, 0.8),
                "I": self._random_directional_total(random.uniform(0.2, 1.0)),
                "R": random.uniform(0.2, 1.0),
                "M": self._random_directional_total(random.uniform(0.2, 1.0)),
                "K": self._random_directional_total(random.uniform(0.1, 0.7)),
                "S": random.uniform(0.1, 0.7),
                "H1": random.choice([0.0, 1.0]),
                "H2": random.choice([0.0, 1.0]),
                "H3": random.choice([0.0, 1.0]),
            }

    @staticmethod
    def _random_directional_total(total: float) -> List[float]:
        weights = [random.uniform(0.2, 1.0) for _ in range(6)]
        wsum = sum(weights) or 1.0
        return [total * w / wsum for w in weights]

    @staticmethod
    def _halve_protein_value(value: Union[float, List[float]]) -> Union[float, List[float]]:
        if isinstance(value, list):
            return [v * 0.5 for v in value]
        return value * 0.5

    def copy_for_daughter(self, new_id: int, division_energy_fraction: float = 0.45) -> "Cell":
        daughter_energy = self.energy * division_energy_fraction
        self.energy *= (1.0 - division_energy_fraction)

        daughter = Cell(
            cell_id=new_id,
            energy=daughter_energy,
            age=0,
            divisions_done=self.divisions_done + 1,
            phase=CellCyclePhase.G,
            commitment=self.commitment,
            cluster_id=self.cluster_id,
            low_energy_epochs=0,
            stall_epochs=0,
            stress_epochs=0,
            candidate_cluster=self.candidate_cluster,
            candidate_cluster_epochs=self.candidate_cluster_epochs,
            proteins={k: self._halve_protein_value(v) for k, v in self.proteins.items()},
        )

        for k, v in list(self.proteins.items()):
            self.proteins[k] = self._halve_protein_value(v)

        for key in ("I", "M", "K"):
            if isinstance(self.proteins[key], list):
                self.proteins[key] = [max(0.0, x + random.uniform(-0.04, 0.04)) for x in self.proteins[key]]
            if isinstance(daughter.proteins[key], list):
                daughter.proteins[key] = [max(0.0, x + random.uniform(-0.04, 0.04)) for x in daughter.proteins[key]]

        return daughter
