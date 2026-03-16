"""
Grid system for multiple hexapods with triangular tiling.

Manages a grid of hexapod platforms with corner-to-corner connections.
Each platform knows its neighbors and which corners connect.

Grid layout (triangular tiling -- only UP positions are populated):
    Row 0:  ^     ^     ^
    Row 1:     ^     ^
    Row 2:  ^     ^     ^

Where ^ = UP orientation (corner 0 points +X), created at (row+col)%2 == 0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from hexapod_grid_sim.physics.solver import HexapodGeometry
from hexapod_grid_sim.grid.hexapod_unit import (
    HexapodUnit,
    HexapodState,
    create_hexapod,
)


# ------------------------------------------------------------------
# Enums & small data types
# ------------------------------------------------------------------

class Orientation(Enum):
    """Platform orientation in the grid."""
    UP = "up"      # Corner 0 points toward +X (standard)
    DOWN = "down"  # Rotated 180 deg (corner 0 points toward -X)


@dataclass
class GridPosition:
    """Position of a hexapod in the grid (row, col). Hashable."""
    row: int
    col: int

    def __hash__(self) -> int:
        return hash((self.row, self.col))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, GridPosition):
            return NotImplemented
        return self.row == other.row and self.col == other.col

    def __repr__(self) -> str:
        return f"GridPosition(row={self.row}, col={self.col})"


@dataclass
class CornerConnection:
    """
    Connection between two corners of adjacent platforms.

    ``pos_a / corner_a`` is one side; ``pos_b / corner_b`` the other.
    """
    pos_a: GridPosition
    corner_a: int   # 0, 1, or 2

    pos_b: GridPosition
    corner_b: int   # 0, 1, or 2

    def involves(self, pos: GridPosition) -> bool:
        """Return True if this connection involves *pos*."""
        return self.pos_a == pos or self.pos_b == pos

    def get_neighbor_corner(
        self, pos: GridPosition, corner: int
    ) -> Optional[Tuple[GridPosition, int]]:
        """
        Given one side of the connection, return the other side.

        Returns ``None`` if *(pos, corner)* does not match either side.
        """
        if self.pos_a == pos and self.corner_a == corner:
            return (self.pos_b, self.corner_b)
        if self.pos_b == pos and self.corner_b == corner:
            return (self.pos_a, self.corner_a)
        return None


@dataclass
class GridPlatform:
    """A platform placed in the grid with world-space transform and neighbor info."""

    position: GridPosition
    orientation: Orientation
    hexapod: HexapodUnit
    world_offset: np.ndarray   # [x, y, z] offset in world coords
    rotation: float            # Rotation angle in radians

    # corner_index -> (neighbor_pos, neighbor_corner)
    neighbors: Dict[int, Tuple[GridPosition, int]] = field(default_factory=dict)

    @property
    def state(self) -> HexapodState:
        """Solve and return current hexapod state."""
        return self.hexapod.solve()

    def get_corner_world_position(self, corner: int) -> Optional[np.ndarray]:
        """
        Get the world position of the given corner index (0, 1, or 2).

        Applies the platform rotation and world offset to the local vertex.
        """
        state = self.state
        if state.vertices is None:
            return None

        local_pos = state.vertices[corner]

        cos_r = np.cos(self.rotation)
        sin_r = np.sin(self.rotation)
        rotated = np.array([
            local_pos[0] * cos_r - local_pos[2] * sin_r,
            local_pos[1],
            local_pos[0] * sin_r + local_pos[2] * cos_r,
        ])

        return rotated + self.world_offset


# ------------------------------------------------------------------
# Main grid
# ------------------------------------------------------------------

class HexapodGrid:
    """
    Grid of hexapod platforms with triangular tiling and corner connections.

    Only UP-orientation positions (where ``(row + col) % 2 == 0``) are
    populated -- this produces a "Triforce" / checkerboard pattern.
    """

    def __init__(
        self,
        geometry: HexapodGeometry,
        rows: int = 3,
        cols: int = 3,
    ) -> None:
        self.geometry = geometry
        self.rows = rows
        self.cols = cols

        # Spacing for UP-only triangular grid:
        #   horizontal (between columns) = 1.5 * R
        #   vertical   (between rows)    = sqrt(3)/2 * R
        R = geometry.platform_radius
        self.spacing_x: float = R * 1.5
        self.spacing_z: float = R * math.sqrt(3) / 2

        self.platforms: Dict[GridPosition, GridPlatform] = {}
        self.connections: List[CornerConnection] = []

        self._create_grid()
        self._create_connections()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_orientation(row: int, col: int) -> Orientation:
        """Checkerboard pattern: UP when (row+col) is even, DOWN otherwise."""
        return Orientation.UP if (row + col) % 2 == 0 else Orientation.DOWN

    def _get_world_position(self, row: int, col: int) -> np.ndarray:
        """Map grid (row, col) to a world-space XZ position (Y=0)."""
        x = col * self.spacing_x
        z = row * self.spacing_z
        return np.array([x, 0.0, z])

    @staticmethod
    def _get_rotation(orientation: Orientation) -> float:
        """Rotation angle in radians for the given orientation."""
        return 0.0 if orientation == Orientation.UP else math.pi

    def _create_grid(self) -> None:
        """Populate platforms at every UP position inside (rows x cols)."""
        for row in range(self.rows):
            for col in range(self.cols):
                orientation = self._get_orientation(row, col)
                if orientation != Orientation.UP:
                    continue

                pos = GridPosition(row, col)
                world_pos = self._get_world_position(row, col)
                rotation = self._get_rotation(orientation)

                hexapod = create_hexapod(
                    rail_start=self.geometry.rail_start,
                    rail_length=self.geometry.rail_length,
                    rod_length=self.geometry.rod_length,
                    platform_radius=self.geometry.platform_radius,
                )

                self.platforms[pos] = GridPlatform(
                    position=pos,
                    orientation=orientation,
                    hexapod=hexapod,
                    world_offset=world_pos,
                    rotation=rotation,
                )

    def _create_connections(self) -> None:
        """
        Discover corner connections by computing world positions at a
        reference height and finding corners that fall within tolerance.
        """
        self.connections.clear()
        for platform in self.platforms.values():
            platform.neighbors.clear()

        # Solve every platform at a reference pose so we can inspect corners
        corner_positions: Dict[Tuple[GridPosition, int], np.ndarray] = {}
        for pos, platform in self.platforms.items():
            platform.hexapod.set_pose(height=70, pitch=0, roll=0)
            state = platform.hexapod.solve()
            if state.vertices is not None:
                for i in range(3):
                    world_pos = platform.get_corner_world_position(i)
                    if world_pos is not None:
                        corner_positions[(pos, i)] = world_pos

        tolerance = self.geometry.platform_radius * 0.1
        processed: Set[Tuple[Tuple[int, int, int], ...]] = set()

        for (pos_a, corner_a), world_a in corner_positions.items():
            for (pos_b, corner_b), world_b in corner_positions.items():
                if pos_a == pos_b:
                    continue
                key = tuple(sorted([
                    (pos_a.row, pos_a.col, corner_a),
                    (pos_b.row, pos_b.col, corner_b),
                ]))
                if key in processed:
                    continue

                dist_xz = math.sqrt(
                    (world_a[0] - world_b[0]) ** 2
                    + (world_a[2] - world_b[2]) ** 2
                )
                if dist_xz < tolerance:
                    self._add_connection(pos_a, corner_a, pos_b, corner_b)
                    processed.add(key)

    def _add_connection(
        self,
        pos_a: GridPosition,
        corner_a: int,
        pos_b: GridPosition,
        corner_b: int,
    ) -> None:
        """Register a corner connection (idempotent)."""
        for conn in self.connections:
            if (
                conn.pos_a == pos_a
                and conn.corner_a == corner_a
                and conn.pos_b == pos_b
                and conn.corner_b == corner_b
            ):
                return
            if (
                conn.pos_a == pos_b
                and conn.corner_a == corner_b
                and conn.pos_b == pos_a
                and conn.corner_b == corner_a
            ):
                return

        self.connections.append(
            CornerConnection(pos_a, corner_a, pos_b, corner_b)
        )
        self.platforms[pos_a].neighbors[corner_a] = (pos_b, corner_b)
        self.platforms[pos_b].neighbors[corner_b] = (pos_a, corner_a)

    def _get_corner_world_positions(
        self,
        world_offset: np.ndarray,
        rotation: float,
        height: float,
    ) -> List[np.ndarray]:
        """Compute world positions of all three corners for a (possibly ghost) platform."""
        R = self.geometry.platform_radius
        cos_r = math.cos(rotation)
        sin_r = math.sin(rotation)

        corners: List[np.ndarray] = []
        for i in range(3):
            angle = math.radians(i * 120)
            local_x = R * math.cos(angle)
            local_z = R * math.sin(angle)

            world_x = local_x * cos_r - local_z * sin_r + world_offset[0]
            world_z = local_x * sin_r + local_z * cos_r + world_offset[2]
            corners.append(np.array([world_x, height, world_z]))

        return corners

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_platform(self, row: int, col: int) -> Optional[GridPlatform]:
        """Return the platform at *(row, col)* or ``None``."""
        return self.platforms.get(GridPosition(row, col))

    def get_all_platforms(self) -> List[GridPlatform]:
        """Return all platforms in insertion order."""
        return list(self.platforms.values())

    def solve_all(self) -> Dict[GridPosition, HexapodState]:
        """Solve every platform and return its state keyed by position."""
        return {pos: p.hexapod.solve() for pos, p in self.platforms.items()}

    def get_connection_mismatches(self) -> List[Tuple[CornerConnection, float]]:
        """
        Find connections where the connected corner heights disagree.

        Returns a list of ``(connection, height_difference)`` tuples whose
        height difference exceeds a small threshold.
        """
        mismatches: List[Tuple[CornerConnection, float]] = []
        for conn in self.connections:
            platform_a = self.platforms[conn.pos_a]
            platform_b = self.platforms[conn.pos_b]

            pos_a = platform_a.get_corner_world_position(conn.corner_a)
            pos_b = platform_b.get_corner_world_position(conn.corner_b)

            if pos_a is not None and pos_b is not None:
                height_diff = abs(float(pos_a[1]) - float(pos_b[1]))
                if height_diff > 0.1:
                    mismatches.append((conn, height_diff))

        return mismatches

    def get_empty_slots(self) -> List[GridPosition]:
        """
        Return grid positions that are currently unoccupied but share at
        least one corner with an existing platform.
        """
        occupied = set(self.platforms.keys())
        if not occupied:
            return []

        min_row = min(p.row for p in occupied) - 2
        max_row = max(p.row for p in occupied) + 2
        min_col = min(p.col for p in occupied) - 2
        max_col = max(p.col for p in occupied) + 2

        tolerance = self.geometry.platform_radius * 0.15
        empty: List[GridPosition] = []

        for row in range(min_row, max_row + 1):
            for col in range(min_col, max_col + 1):
                pos = GridPosition(row, col)
                if pos in occupied:
                    continue

                ghost_orientation = self._get_orientation(row, col)
                ghost_world = self._get_world_position(row, col)
                ghost_rotation = self._get_rotation(ghost_orientation)

                ghost_corners = self._get_corner_world_positions(
                    ghost_world, ghost_rotation, height=70.0
                )

                shares_corner = False
                for existing_pos in occupied:
                    existing_platform = self.platforms[existing_pos]
                    existing_corners = self._get_corner_world_positions(
                        existing_platform.world_offset,
                        existing_platform.rotation,
                        height=70.0,
                    )
                    for gc in ghost_corners:
                        for ec in existing_corners:
                            dist = math.sqrt(
                                (gc[0] - ec[0]) ** 2 + (gc[2] - ec[2]) ** 2
                            )
                            if dist < tolerance:
                                shares_corner = True
                                break
                        if shares_corner:
                            break
                    if shares_corner:
                        break

                if shares_corner:
                    empty.append(pos)

        return empty

    # ------------------------------------------------------------------
    # Dynamic modification
    # ------------------------------------------------------------------

    def add_platform(self, row: int, col: int) -> Optional[GridPlatform]:
        """
        Add a new platform at *(row, col)*.

        Returns the new ``GridPlatform``, or ``None`` if the position is
        already occupied.
        """
        pos = GridPosition(row, col)
        if pos in self.platforms:
            return None

        orientation = self._get_orientation(row, col)
        world_pos = self._get_world_position(row, col)
        rotation = self._get_rotation(orientation)

        hexapod = create_hexapod(
            rail_start=self.geometry.rail_start,
            rail_length=self.geometry.rail_length,
            rod_length=self.geometry.rod_length,
            platform_radius=self.geometry.platform_radius,
        )
        hexapod.set_pose(height=70, pitch=0, roll=0)

        platform = GridPlatform(
            position=pos,
            orientation=orientation,
            hexapod=hexapod,
            world_offset=world_pos,
            rotation=rotation,
        )
        self.platforms[pos] = platform

        self.rows = max(self.rows, row + 1)
        self.cols = max(self.cols, col + 1)

        self._create_connections()
        return platform

    def remove_platform(self, row: int, col: int) -> bool:
        """
        Remove the platform at *(row, col)*.

        Returns ``True`` if a platform was removed, ``False`` if none existed.
        """
        pos = GridPosition(row, col)
        if pos not in self.platforms:
            return False

        del self.platforms[pos]
        self._create_connections()
        return True

    def __str__(self) -> str:
        return (
            f"HexapodGrid {self.rows}x{self.cols}\n"
            f"  Platforms: {len(self.platforms)}\n"
            f"  Connections: {len(self.connections)}"
        )
