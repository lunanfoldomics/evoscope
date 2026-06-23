"""
Core Evoscope simulation engine.

This module implements the main `Evoscope` class, which evolves a population
of simulated cells on a toroidal hexagonal grid. The simulation couples local
nutrient availability, cell energy, cell-cycle progression, commitment and
decommitment, directional protein-like variables, movement, division,
competition, and cluster-level organization.

At each simulation step, cells sense their local environment, update their
internal regulatory state, select possible actions, and modify the spatial
configuration of the tissue-like system. The engine also records global and
cluster-level gene/protein summaries and exports morphology snapshots for
visualization and downstream analysis.

Conceptually, this module is the dynamical core of Evoscope: it links local
regulatory rules to emergent spatial organization.

Evoscope v0.9.2
Author: Luca Zammataro
Organization: Lunan Foldomics LLC
"""

from __future__ import annotations

import os
import random
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import Config
from .enums import ActionType, CellCyclePhase, CommitmentState
from .grid import ToroidalHexGrid
from .state import Cell, PlannedAction
from .utils import clamp, directional_value, grid_to_numpy, protein_total


class Evoscope:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        random.seed(cfg.seed)
        
        os.makedirs(self.cfg.snapshot_dir, exist_ok=True)

        self.grid = ToroidalHexGrid(cfg.width, cfg.height)
        self.occupancy: Dict[Tuple[int, int], Cell] = {}
        self.nutrient: Dict[Tuple[int, int], float] = {
            pos: cfg.initial_medium_nutrient for pos in self.grid.iter_positions()
        }

        self.epoch = 0
        self.next_cell_id = 1
        self.metrics_history: List[Dict[str, float]] = []

        self._seed_world()

        # Gene History
        self.gene_history = {
            "T1": [],
            "T2": [],
            "I": [],
            "R": [],
            "M": [],
            "K": [],
            "S": []
        }

        self.cluster_gene_history = {
            cid: {g: [] for g in ["T1","T2","I","R","M","K","S"]}
            for cid in range(8)
        }

        self.cluster_size_history = {cid: [] for cid in range(8)}        


    def _compute_global_expression(self):
        totals = {k: 0.0 for k in self.gene_history.keys()}
        n = len(self.occupancy)

        if n == 0:
            return {k: 0.0 for k in totals}

        for cell in self.occupancy.values():
            for k in totals:
                value = cell.proteins.get(k, 0.0)
                totals[k] += protein_total(value)

        return {k: totals[k] / n for k in totals}


    def _compute_cluster_expression(self):
        # accumulo
        totals = {
            cid: {g: 0.0 for g in ["T1","T2","I","R","M","K","S"]}
            for cid in range(8)
        }
        counts = {cid: 0 for cid in range(8)}

        for cell in self.occupancy.values():
            cid = cell.cluster_id
            if cid is None:
                continue

            counts[cid] += 1

            for g in totals[cid]:
                value = cell.proteins.get(g, 0.0)
                totals[cid][g] += protein_total(value)

        # medie
        means = {}
        for cid in range(8):
            if counts[cid] == 0:
                means[cid] = {g: 0.0 for g in totals[cid]}
            else:
                means[cid] = {
                    g: totals[cid][g] / counts[cid]
                    for g in totals[cid]
                }

        return means, counts




    def _seed_world(self) -> None:
        positions = self.grid.iter_positions()
        random.shuffle(positions)

        for pos in positions[: self.cfg.initial_cells]:
            cell = Cell(
                cell_id=self.next_cell_id,
                energy=random.uniform(self.cfg.energy_init_min, self.cfg.energy_init_max),
            )
            self.next_cell_id += 1
            self.occupancy[pos] = cell

        for _ in range(self.cfg.nutrient_hotspots):
            pos = random.choice(positions)
            self.nutrient[pos] += random.uniform(3.0, 7.0)

    def run(self, epochs: int = 100) -> None:
        for _ in range(epochs):
            self.step()

    def step(self) -> None:
        self.epoch += 1

        plans: Dict[Tuple[int, int], PlannedAction] = {}
        dead_positions: List[Tuple[int, int]] = []

        current_positions = list(self.occupancy.keys())
        random.shuffle(current_positions)

        for pos in current_positions:
            cell = self.occupancy.get(pos)
            if cell is None:
                continue

            context = self._sense(pos, cell)
            self._update_regulation(cell, context)
            self._update_energy(cell, pos)
            self._update_cycle(cell)
            self._update_commitment(cell, context)

            if self._should_die(cell):
                dead_positions.append(pos)
                continue

            plan = self._decide_action(pos, cell, context)
            plans[pos] = plan

        for pos in dead_positions:
            self._kill_cell_at(pos, release=True)

        self._apply_movements(plans)
        self._apply_attack_and_divide(plans)
        self._apply_divisions(plans)

        self._diffuse_nutrient()
        self._decay_nutrient()

        self.metrics_history.append(self._compute_metrics())

        # GLOBAL genes
        expr = self._compute_global_expression()
        for k in self.gene_history:
            self.gene_history[k].append(expr[k])        

        if self.cfg.verbose:
            print(self.summary_line())

        # CLUSTER genes
        cluster_expr, cluster_counts = self._compute_cluster_expression()
        for cid in range(8):
            for g in self.cluster_gene_history[cid]:
                self.cluster_gene_history[cid][g].append(cluster_expr[cid][g])

            self.cluster_size_history[cid].append(cluster_counts[cid])


        if self.epoch % self.cfg.snapshot_every == 0:
            np.save(
                os.path.join(self.cfg.snapshot_dir, f"grid_{self.epoch:03d}.npy"),
                grid_to_numpy(self),
            )

    def _sense(self, pos: Tuple[int, int], cell: Cell) -> Dict:
        neighbors = self.grid.neighbors(pos)
        same_cluster = 0
        diff_cluster = 0
        occupied = 0
        local_nutrient = self.nutrient[pos]

        neighbor_info = []
        for dir_idx, npos in enumerate(neighbors):
            ncell = self.occupancy.get(npos)
            if ncell is not None:
                occupied += 1
                if cell.cluster_id is not None and ncell.cluster_id is not None:
                    if cell.cluster_id == ncell.cluster_id:
                        same_cluster += 1
                    else:
                        diff_cluster += 1

            neighbor_info.append(
                {
                    "dir_idx": dir_idx,
                    "pos": npos,
                    "cell": ncell,
                    "nutrient": self.nutrient[npos],
                }
            )

        crowding = occupied / 6.0

        return {
            "neighbors": neighbor_info,
            "local_nutrient": local_nutrient,
            "crowding": crowding,
            "same_cluster": same_cluster,
            "diff_cluster": diff_cluster,
        }

    def _h_bias_profile(self, cluster_id: Optional[int]) -> Dict[str, float]:
        if cluster_id is None:
            return {"adh": 0.0, "mot": 0.0, "upt": 0.0, "agg": 0.0}

        h1 = (cluster_id >> 2) & 1
        h2 = (cluster_id >> 1) & 1
        h3 = cluster_id & 1

        s = self.cfg.h_bias_strength
        return {
            "adh": s * (1 if h1 else -1),
            "mot": s * (1 if h2 else -1),
            "agg": s * (1 if h3 else -1),
            "upt": s * (1 if (h1 ^ h2) else -1),
        }

    def _directional_weights(self, weights: List[float]) -> List[float]:
        adjusted = [max(0.01, w + random.uniform(0.0, self.cfg.directional_noise)) for w in weights]
        total = sum(adjusted) or 1.0
        return [w / total for w in adjusted]

    def _redistribute_directional(self, total_amount: float, weights: List[float]) -> List[float]:
        total_amount = clamp(total_amount, 0.0, self.cfg.max_protein_level)
        norm = self._directional_weights(weights)
        return [total_amount * w for w in norm]

    def _dominant_direction(self, value: Union[float, List[float]]) -> int:
        if isinstance(value, list):
            return max(range(len(value)), key=lambda i: value[i])
        return 0

    def _direction_index(self, src: Tuple[int, int], target: Tuple[int, int]) -> Optional[int]:
        for dir_idx, npos in enumerate(self.grid.neighbors(src)):
            if npos == target:
                return dir_idx
        return None

    def _update_regulation(self, cell: Cell, context: Dict) -> None:
        p = cell.proteins
        nutrient = context["local_nutrient"]
        crowding = context["crowding"]
        bias = self._h_bias_profile(cell.cluster_id)
        neighbors = context["neighbors"]

        t1_synth = self.cfg.synth_rate_base * (0.4 + nutrient / (1.0 + nutrient))
        t2_synth = self.cfg.synth_rate_base * (
            0.2 + crowding + (0.8 if cell.energy < self.cfg.starvation_threshold else 0.0)
        )

        p["T1"] = clamp(float(p["T1"]) + t1_synth - self.cfg.degrade_rate * float(p["T1"]), 0.0, self.cfg.max_protein_level)
        p["T2"] = clamp(float(p["T2"]) + t2_synth - self.cfg.degrade_rate * float(p["T2"]), 0.0, self.cfg.max_protein_level)

        transcription_drive = clamp(0.25 + 0.40 * float(p["T1"]) - 0.25 * float(p["T2"]), 0.0, 2.0)

        m1_total = clamp(
            protein_total(p["I"]) + self.cfg.synth_rate_base * (0.4 * transcription_drive + max(0.0, bias["adh"]))
            - self.cfg.degrade_rate * protein_total(p["I"]),
            0.0,
            self.cfg.max_protein_level,
        )
        p["R"] = clamp(
            float(p["R"]) + self.cfg.synth_rate_base * (0.4 * transcription_drive + max(0.0, bias["upt"]))
            - self.cfg.degrade_rate * float(p["R"]),
            0.0,
            self.cfg.max_protein_level,
        )
        m_total = clamp(
            protein_total(p["M"]) + self.cfg.synth_rate_base * (0.35 * transcription_drive + max(0.0, bias["mot"]))
            - self.cfg.degrade_rate * protein_total(p["M"]),
            0.0,
            self.cfg.max_protein_level,
        )
        k_total = clamp(
            protein_total(p["K"]) + self.cfg.synth_rate_base * (0.28 * transcription_drive + max(0.0, bias["agg"]))
            - self.cfg.degrade_rate * protein_total(p["K"]),
            0.0,
            self.cfg.max_protein_level,
        )
        p["S"] = clamp(
            float(p["S"]) + self.cfg.synth_rate_base * (0.28 * transcription_drive + 0.15 * crowding)
            - self.cfg.degrade_rate * float(p["S"]),
            0.0,
            self.cfg.max_protein_level,
        )

        adh_weights: List[float] = []
        mot_weights: List[float] = []
        atk_weights: List[float] = []

        for entry in neighbors:
            dir_idx = entry["dir_idx"]
            ncell = entry["cell"]
            nutrient_here = entry["nutrient"]
            if ncell is None:
                adh_w = 0.08
                mot_w = 1.0 + self.cfg.empty_polarization_bonus + 0.35 * nutrient_here + max(0.0, bias["mot"])
                atk_w = 0.05
            else:
                same = (
                    cell.cluster_id is not None
                    and ncell.cluster_id is not None
                    and cell.cluster_id == ncell.cluster_id
                )
                diff = (
                    cell.cluster_id is not None
                    and ncell.cluster_id is not None
                    and cell.cluster_id != ncell.cluster_id
                )
                vulnerability = max(0.0, protein_total(p["K"]) - float(ncell.proteins["S"]))
                adh_w = 0.20 + self.cfg.occupied_polarization_bonus + (self.cfg.same_cluster_adhesion_bonus if same else 0.0)
                mot_w = 0.10 + (0.18 if diff else 0.0)
                atk_w = 0.10 + self.cfg.occupied_polarization_bonus + self.cfg.vulnerable_attack_bias * vulnerability
                if diff:
                    atk_w += self.cfg.mismatch_attack_bonus
            adh_weights.append(adh_w)
            mot_weights.append(mot_w)
            atk_weights.append(atk_w)

        old_m1 = p["I"] if isinstance(p["I"], list) else [float(p["I"]) / 6.0] * 6
        old_m = p["M"] if isinstance(p["M"], list) else [float(p["M"]) / 6.0] * 6
        old_k = p["K"] if isinstance(p["K"], list) else [float(p["K"]) / 6.0] * 6

        adh_weights = [0.60 * w + 0.40 * old_m1[i] for i, w in enumerate(adh_weights)]
        mot_weights = [0.65 * w + 0.35 * old_m[i] for i, w in enumerate(mot_weights)]
        atk_weights = [0.65 * w + 0.35 * old_k[i] for i, w in enumerate(atk_weights)]

        p["I"] = self._redistribute_directional(m1_total, adh_weights)
        p["M"] = self._redistribute_directional(m_total, mot_weights)
        p["K"] = self._redistribute_directional(k_total, atk_weights)

        if cell.commitment != CommitmentState.COMMITTED:
            for hk in ("H1", "H2", "H3"):
                pull = 0.12 * random.uniform(-1.0, 1.0) + 0.05 * transcription_drive
                p[hk] = clamp(float(p[hk]) + pull - self.cfg.degrade_rate * 0.25 * float(p[hk]), 0.0, 1.0)
        else:
            cid = cell.cluster_id if cell.cluster_id is not None else 0
            p["H1"] = float((cid >> 2) & 1)
            p["H2"] = float((cid >> 1) & 1)
            p["H3"] = float(cid & 1)

    def _update_energy(self, cell: Cell, pos: Tuple[int, int]) -> None:
        p = cell.proteins
        uptake = self.nutrient[pos] * (0.10 + 0.18 * float(p["R"]))
        uptake = min(uptake, self.nutrient[pos])
        self.nutrient[pos] -= uptake

        protein_load = sum(protein_total(v) for v in p.values())
        cost = self.cfg.basal_cost + self.cfg.protein_cost_factor * protein_load

        cell.energy = clamp(cell.energy + uptake - cost, 0.0, self.cfg.energy_max)

        if cell.energy < self.cfg.starvation_threshold:
            cell.low_energy_epochs += 1
            cell.stress_epochs += 1
        else:
            cell.low_energy_epochs = max(0, cell.low_energy_epochs - 1)
            cell.stress_epochs = max(0, cell.stress_epochs - 1)

    def _update_cycle(self, cell: Cell) -> None:
        cell.age += 1

        if cell.phase == CellCyclePhase.G:
            if cell.energy >= self.cfg.s_entry_energy:
                cell.phase = CellCyclePhase.S
        elif cell.phase == CellCyclePhase.S:
            if cell.energy >= self.cfg.m_entry_energy:
                cell.phase = CellCyclePhase.M
        elif cell.phase == CellCyclePhase.M:
            pass
        elif cell.phase == CellCyclePhase.STALL:
            if cell.energy >= self.cfg.s_entry_energy:
                cell.phase = CellCyclePhase.S
                cell.stall_epochs = 0

    def _update_commitment(self, cell: Cell, context: Dict) -> None:
        p = cell.proteins

        if cell.commitment == CommitmentState.UNDETERMINED:
            candidate = (
                ((1 if p["H1"] >= 0.5 else 0) << 2)
                | ((1 if p["H2"] >= 0.5 else 0) << 1)
                | (1 if p["H3"] >= 0.5 else 0)
            )

            if cell.candidate_cluster == candidate:
                cell.candidate_cluster_epochs += 1
            else:
                cell.candidate_cluster = candidate
                cell.candidate_cluster_epochs = 1

            if cell.candidate_cluster_epochs >= self.cfg.commitment_stability_epochs:
                cell.commitment = CommitmentState.COMMITTED
                cell.cluster_id = candidate

        elif cell.commitment == CommitmentState.COMMITTED:
            if cell.stress_epochs >= self.cfg.decommit_stress_epochs:
                if random.random() < self.cfg.decommit_probability:
                    cell.commitment = CommitmentState.DECOMMITTED
                    cell.cluster_id = None
                    cell.candidate_cluster = None
                    cell.candidate_cluster_epochs = 0

        elif cell.commitment == CommitmentState.DECOMMITTED:
            if cell.energy > self.cfg.s_entry_energy:
                cell.commitment = CommitmentState.UNDETERMINED

    def _should_die(self, cell: Cell) -> bool:
        if cell.energy <= self.cfg.critical_energy and cell.low_energy_epochs >= self.cfg.critical_epochs_before_death:
            return True
        if cell.divisions_done >= self.cfg.max_divisions:
            return True
        if cell.age >= self.cfg.max_age:
            return True
        return False

    def _decide_action(self, pos: Tuple[int, int], cell: Cell, context: Dict) -> PlannedAction:
        neighbors = context["neighbors"]
        empty_neighbors = [n for n in neighbors if n["cell"] is None]

        if cell.phase == CellCyclePhase.M and cell.energy >= self.cfg.divide_energy_min:
            if empty_neighbors:
                target = self._choose_division_target(cell, empty_neighbors)
                return PlannedAction(ActionType.DIVIDE, source=pos, target=target)
            attack_target = self._choose_attack_target(cell, neighbors)
            if attack_target is not None:
                return PlannedAction(ActionType.ATTACK_AND_DIVIDE, source=pos, target=attack_target, attack_target=attack_target)
            cell.phase = CellCyclePhase.STALL
            cell.stall_epochs += 1
            return PlannedAction(ActionType.REST, source=pos)

        move_target = self._choose_move_target(cell, neighbors)
        if move_target is not None:
            return PlannedAction(ActionType.MOVE, source=pos, target=move_target)

        return PlannedAction(ActionType.REST, source=pos)

    def _choose_division_target(self, cell: Cell, empty_neighbors: List[Dict]) -> Tuple[int, int]:
        best_score = -1e9
        best_pos = empty_neighbors[0]["pos"]

        for entry in empty_neighbors:
            npos = entry["pos"]
            dir_idx = entry["dir_idx"]
            nutrient = entry["nutrient"]
            crowd = self._local_crowding(npos)
            same_bonus = self._same_cluster_count_around(cell.cluster_id, npos) * self.cfg.divide_same_cluster_bonus
            score = (
                self.cfg.divide_nutrient_weight * nutrient
                + same_bonus
                + self.cfg.divide_less_crowded_bonus * (1.0 - crowd)
                + 0.55 * directional_value(cell.proteins["M"], dir_idx)
                - 0.20 * directional_value(cell.proteins["I"], dir_idx)
                + random.uniform(-0.05, 0.05)
            )
            if score > best_score:
                best_score = score
                best_pos = npos

        return best_pos

    def _choose_move_target(self, cell: Cell, neighbors: List[Dict]) -> Optional[Tuple[int, int]]:
        stay_pressure = 0.55 * protein_total(cell.proteins["I"]) - 0.35 * protein_total(cell.proteins["M"])
        if stay_pressure > random.uniform(0.1, 1.4):
            return None

        candidates = [n for n in neighbors if n["cell"] is None]
        if not candidates:
            return None

        best_score = -1e9
        best_pos: Optional[Tuple[int, int]] = None

        for entry in candidates:
            npos = entry["pos"]
            dir_idx = entry["dir_idx"]
            nutrient = entry["nutrient"]
            crowd = self._local_crowding(npos)
            same = self._same_cluster_count_around(cell.cluster_id, npos)
            diff = self._diff_cluster_count_around(cell.cluster_id, npos)
            m_dir = directional_value(cell.proteins["M"], dir_idx)
            m1_dir = directional_value(cell.proteins["I"], dir_idx)

            score = (
                self.cfg.move_empty_bonus
                + self.cfg.move_nutrient_weight * nutrient
                - self.cfg.move_crowding_penalty * crowd
                + self.cfg.move_same_cluster_bonus * same
                - self.cfg.move_diff_cluster_penalty * diff
                + 0.80 * m_dir
                - 0.35 * m1_dir
                + random.uniform(-0.08, 0.08)
            )
            if score > best_score:
                best_score = score
                best_pos = npos

        return best_pos

    def _choose_attack_target(self, cell: Cell, neighbors: List[Dict]) -> Optional[Tuple[int, int]]:
        best_score = -1e9
        best_target = None

        for entry in neighbors:
            ncell = entry["cell"]
            if ncell is None:
                continue

            dir_idx = entry["dir_idx"]
            mismatch_bonus = 0.0
            if cell.cluster_id is not None and ncell.cluster_id is not None and cell.cluster_id != ncell.cluster_id:
                mismatch_bonus = self.cfg.mismatch_attack_bonus

            score = (
                directional_value(cell.proteins["K"], dir_idx)
                - float(ncell.proteins["S"])
                + mismatch_bonus
                + 0.15 * directional_value(cell.proteins["M"], dir_idx)
                + random.uniform(-0.05, 0.05)
            )
            if score > best_score:
                best_score = score
                best_target = entry["pos"]

        if best_score >= self.cfg.attack_threshold:
            return best_target
        return None

    def _apply_movements(self, plans: Dict[Tuple[int, int], PlannedAction]) -> None:
        move_requests: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        for src, plan in plans.items():
            if plan.action_type == ActionType.MOVE and plan.target is not None:
                if src in self.occupancy and plan.target not in self.occupancy:
                    move_requests.setdefault(plan.target, []).append(src)

        for target, sources in move_requests.items():
            winner_src = random.choice(sources)
            if winner_src not in self.occupancy or target in self.occupancy:
                continue

            cell = self.occupancy.pop(winner_src)
            cell.energy = max(0.0, cell.energy - self.cfg.movement_cost)
            self.occupancy[target] = cell

    def _apply_attack_and_divide(self, plans: Dict[Tuple[int, int], PlannedAction]) -> None:
        requests: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        for src, plan in plans.items():
            if plan.action_type == ActionType.ATTACK_AND_DIVIDE and plan.attack_target is not None:
                if src in self.occupancy:
                    requests.setdefault(plan.attack_target, []).append(src)

        for target, attackers in requests.items():
            defender = self.occupancy.get(target)
            if defender is None:
                continue

            winner_src = self._choose_best_attacker(target, attackers)
            if winner_src is None or winner_src not in self.occupancy:
                continue

            attacker = self.occupancy[winner_src]
            dir_idx = self._direction_index(winner_src, target)
            if dir_idx is None:
                continue

            attack_strength = directional_value(attacker.proteins["K"], dir_idx) - float(defender.proteins["S"])
            if attacker.cluster_id is not None and defender.cluster_id is not None and attacker.cluster_id != defender.cluster_id:
                attack_strength += self.cfg.mismatch_attack_bonus
            attack_strength += 0.15 * directional_value(attacker.proteins["M"], dir_idx)

            if attack_strength >= self.cfg.attack_threshold and attacker.energy >= (self.cfg.attack_cost + self.cfg.division_cost):
                attacker.energy -= self.cfg.attack_cost
                self._kill_cell_at(target, release=True)

                daughter = attacker.copy_for_daughter(self.next_cell_id)
                self.next_cell_id += 1

                attacker.energy = max(0.0, attacker.energy - self.cfg.division_cost)
                attacker.phase = CellCyclePhase.G
                attacker.divisions_done += 1
                self.occupancy[target] = daughter


    def _apply_divisions(self, plans: Dict[Tuple[int, int], PlannedAction]) -> None:
        requests: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}

        for src, plan in plans.items():
            if plan.action_type == ActionType.DIVIDE and plan.target is not None:
                if src in self.occupancy and plan.target not in self.occupancy:
                    requests.setdefault(plan.target, []).append(src)

        for target, sources in requests.items():
            winner_src = random.choice(sources)
            if winner_src not in self.occupancy or target in self.occupancy:
                continue

            mother = self.occupancy[winner_src]
            if mother.energy < self.cfg.division_cost:
                continue

            daughter = mother.copy_for_daughter(self.next_cell_id)
            self.next_cell_id += 1

            mother.energy = max(0.0, mother.energy - self.cfg.division_cost)
            mother.phase = CellCyclePhase.G
            mother.divisions_done += 1
            self.occupancy[target] = daughter

    def _choose_best_attacker(self, target: Tuple[int, int], attackers: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        best_src = None
        best_score = -1e9
        defender = self.occupancy.get(target)
        if defender is None:
            return None

        for src in attackers:
            attacker = self.occupancy.get(src)
            if attacker is None:
                continue
            dir_idx = self._direction_index(src, target)
            if dir_idx is None:
                continue
            score = directional_value(attacker.proteins["K"], dir_idx) - float(defender.proteins["S"]) + random.uniform(-0.05, 0.05)
            score += 0.15 * directional_value(attacker.proteins["M"], dir_idx)
            if attacker.cluster_id is not None and defender.cluster_id is not None and attacker.cluster_id != defender.cluster_id:
                score += self.cfg.mismatch_attack_bonus
            if score > best_score:
                best_score = score
                best_src = src

        return best_src

    def _kill_cell_at(self, pos: Tuple[int, int], release: bool = True) -> None:
        cell = self.occupancy.pop(pos, None)
        if cell is None:
            return
        if release:
            release_amount = self.cfg.corpse_release_fraction * cell.energy
            self.nutrient[pos] += release_amount

    def _diffuse_nutrient(self) -> None:
        new_field = dict(self.nutrient)
        for pos in self.grid.iter_positions():
            current = self.nutrient[pos]
            neigh = self.grid.neighbors(pos)
            neigh_mean = sum(self.nutrient[np] for np in neigh) / 6.0
            new_field[pos] = current + self.cfg.diffusion_rate * (neigh_mean - current)
        self.nutrient = new_field

    def _decay_nutrient(self) -> None:
        for pos in self.nutrient:
            self.nutrient[pos] = max(0.0, self.nutrient[pos] * (1.0 - self.cfg.nutrient_decay))

    def _local_crowding(self, pos: Tuple[int, int]) -> float:
        occ = sum(1 for np in self.grid.neighbors(pos) if np in self.occupancy)
        return occ / 6.0

    def _same_cluster_count_around(self, cluster_id: Optional[int], pos: Tuple[int, int]) -> int:
        if cluster_id is None:
            return 0

        count = 0
        for np in self.grid.neighbors(pos):
            c = self.occupancy.get(np)
            if c is not None and c.cluster_id == cluster_id:
                count += 1
        return count

    def _diff_cluster_count_around(self, cluster_id: Optional[int], pos: Tuple[int, int]) -> int:
        if cluster_id is None:
            return 0

        count = 0
        for np in self.grid.neighbors(pos):
            c = self.occupancy.get(np)
            if c is not None and c.cluster_id is not None and c.cluster_id != cluster_id:
                count += 1
        return count

    def _compute_metrics(self) -> Dict[str, float]:
        alive = len(self.occupancy)
        total_energy = sum(c.energy for c in self.occupancy.values())
        mean_energy = total_energy / alive if alive else 0.0
        medium_nutrient = sum(self.nutrient.values())

        '''
        committed = sum(1 for c in self.occupancy.values() if c.commitment == CommitmentState.COMMITTED)
        clusters: Dict[int, int] = {}
        for c in self.occupancy.values():
            if c.cluster_id is not None:
                clusters[c.cluster_id] = clusters.get(c.cluster_id, 0) + 1
        '''

        committed = sum(
            1 for c in self.occupancy.values()
            if c.commitment == CommitmentState.COMMITTED
        )
        undetermined = sum(
            1 for c in self.occupancy.values()
            if c.commitment == CommitmentState.UNDETERMINED
        )
        decommitted = sum(
            1 for c in self.occupancy.values()
            if c.commitment == CommitmentState.DECOMMITTED
        )

        clusters: Dict[int, int] = {}
        for c in self.occupancy.values():
            if c.cluster_id is not None:
                clusters[c.cluster_id] = clusters.get(c.cluster_id, 0) + 1
                

        return {
            "epoch": float(self.epoch),
            "alive": float(alive),
            "mean_energy": mean_energy,
            "medium_nutrient": medium_nutrient,
            "committed": float(committed),
            "undetermined": float(undetermined),
            "decommitted": float(decommitted),
            "uncommitted": float(undetermined + decommitted),
            "n_clusters_present": float(len(clusters)),
            "largest_cluster": float(max(clusters.values()) if clusters else 0),
        }

    def summary_line(self) -> str:
        m = self.metrics_history[-1] if self.metrics_history else self._compute_metrics()
        return (
            f"epoch={int(m['epoch'])} alive={int(m['alive'])} committed={int(m['committed'])} "
            f"clusters={int(m['n_clusters_present'])} largest={int(m['largest_cluster'])} "
            f"meanE={m['mean_energy']:.2f} mediumN={m['medium_nutrient']:.2f}"
        )




    def ascii_snapshot(self, colored: bool = False, block_style: bool = True) -> str:
        rows = []

        reset = "\033[0m"

        # foreground (testo)
        fg_dark = "\033[30m"   # nero
        fg_light = "\033[97m"  # bianco brillante

        # background per cluster
        bg_map = {
            None: "\033[47m",   # undetermined: sfondo bianco/grigio
            0: "\033[41m",      # rosso
            1: "\033[42m",      # verde
            2: "\033[43m",      # giallo
            3: "\033[44m",      # blu
            4: "\033[45m",      # magenta
            5: "\033[46m",      # cyan
            6: "\033[107m",     # bianco brillante
            7: "\033[100m",     # grigio scuro
        }

        # per scegliere testo nero o bianco a seconda dello sfondo
        dark_text_clusters = {2, 5, 6, None}   # giallo, cyan, bianco, undetermined

        for r in range(self.cfg.height):
            prefix = " " if r % 2 else ""
            chars = []

            for q in range(self.cfg.width):
                c = self.occupancy.get((q, r))

                if c is None:
                    chars.append(" . ")
                elif c.cluster_id is None:
                    symbol = "u"
                    if colored:
                        fg = fg_dark if None in dark_text_clusters else fg_light
                        if block_style:
                            chars.append(f"{bg_map[None]}{fg} {symbol} {reset}")
                        else:
                            chars.append(f"{bg_map[None]}{fg}[{symbol}]{reset}")
                    else:
                        chars.append(f" {symbol} ")
                else:
                    cid = c.cluster_id
                    symbol = hex(cid)[-1]
                    if colored:
                        fg = fg_dark if cid in dark_text_clusters else fg_light
                        if block_style:
                            chars.append(f"{bg_map[cid]}{fg} {symbol} {reset}")
                        else:
                            chars.append(f"{bg_map[cid]}{fg}[{symbol}]{reset}")
                    else:
                        chars.append(f" {symbol} ")

            rows.append(prefix + "".join(chars))

        return "\n".join(rows)


    def ascii_legend(self, colored: bool = False, block_style: bool = True) -> str:
        reset = "\033[0m"
        fg_dark = "\033[30m"
        fg_light = "\033[97m"

        bg_map = {
            None: "\033[47m",
            0: "\033[41m",
            1: "\033[42m",
            2: "\033[43m",
            3: "\033[44m",
            4: "\033[45m",
            5: "\033[46m",
            6: "\033[107m",
            7: "\033[100m",
        }

        dark_text_clusters = {2, 5, 6, None}

        def tile(symbol: str, cid):
            if not colored:
                return f"[{symbol}]"
            fg = fg_dark if cid in dark_text_clusters else fg_light
            if block_style:
                return f"{bg_map[cid]}{fg} {symbol} {reset}"
            return f"{bg_map[cid]}{fg}[{symbol}]{reset}"

        parts = [". = empty", f"{tile('u', None)} = undetermined"]
        for i in range(8):
            parts.append(f"{tile(hex(i)[-1], i)} = cluster {i}")
        return " | ".join(parts)


    def frame_with_header(self, colored: bool = False) -> str:
        return f"{self.summary_line()}\n{self.ascii_legend(colored=colored)}\n{self.ascii_snapshot(colored=colored)}"

    def collect_ascii_frames(self, every: int = 1, include_initial: bool = True, colored: bool = False) -> List[str]:
        frames: List[str] = []

        if include_initial:
            frames.append(
                f"epoch=0 alive={len(self.occupancy)}\n"
                f"{self.ascii_legend(colored=colored)}\n"
                f"{self.ascii_snapshot(colored=colored)}"
            )

        if every <= 0:
            every = 1

        if self.metrics_history:
            frames.append(self.frame_with_header(colored=colored))

        return frames

    def run_with_ascii_frames(
        self,
        epochs: int = 100,
        every: int = 1,
        include_initial: bool = True,
        clear_screen: bool = False,
        pause: float = 0.0,
        colored: bool = True,
    ) -> List[str]:
        import time

        if every <= 0:
            every = 1

        frames: List[str] = []

        if include_initial:
            frame0 = (
                f"epoch=0 alive={len(self.occupancy)}\n"
                f"{self.ascii_legend(colored=colored)}\n"
                f"{self.ascii_snapshot(colored=colored)}"
            )
            frames.append(frame0)

            if clear_screen:
                print("\033[2J\033[H" + frame0, flush=True)
                if pause > 0:
                    time.sleep(pause)

        for _ in range(epochs):
            self.step()
            #if self.epoch % every == 0:
            if self.epoch % self.cfg.snapshot_every == 0:    
                frame = self.frame_with_header(colored=colored)
                frames.append(frame)

                if clear_screen:
                    print("\033[2J\033[H" + frame, flush=True)
                    if pause > 0:
                        time.sleep(pause)

        return frames

    def save_ascii_frames(self, path: str, frames: List[str]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for i, frame in enumerate(frames):
                f.write(frame)
                if i < len(frames) - 1:
                    f.write("\n\n" + ("=" * 80) + "\n\n")



