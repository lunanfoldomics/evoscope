"""
Toroidal hexagonal grid topology for Evoscope.

This module defines the axial-coordinate hexagonal lattice used by the
simulation. The grid is toroidal, meaning that positions wrap around at the
boundaries. This removes edge effects and allows local cell-cell interactions
to be studied in a closed spatial environment.

The module provides neighbor lookup, coordinate wrapping, and iteration over
all grid positions. These operations are used by the simulation engine to
evaluate local crowding, nutrient context, movement, division, and interaction
rules.


Evoscope — minimal regulatory spatial model for emergent multicellular organization.

Author: Luca Zammataro
Organization: Lunan Foldomics LLC
"""

from typing import List, Tuple


HEX_DIRECTIONS: List[Tuple[int, int]] = [
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
]


class ToroidalHexGrid:
    """Toroidal axial-coordinate hexagonal grid.

    Teaching note:
    The grid wraps at boundaries, so cells leaving one side re-enter from the opposite side.
    This avoids edge effects and lets students focus on local rules.
    """

    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height

    def wrap(self, pos: Tuple[int, int]) -> Tuple[int, int]:
        q, r = pos
        return (q % self.width, r % self.height)

    def neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        q, r = pos
        return [self.wrap((q + dq, r + dr)) for dq, dr in HEX_DIRECTIONS]

    def iter_positions(self) -> List[Tuple[int, int]]:
        return [(q, r) for r in range(self.height) for q in range(self.width)]
