"""
Constraint propagation via BFS for a hexapod grid.

Propagates corner-height constraints from an anchor platform outward to
its neighbors, ensuring connected corners stay at matching heights.

Each platform is a "node" in the propagation graph with connections as
edges. The system provides data suitable for visualisation overlays.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

from hexapod_grid_sim.grid.topology import (
    HexapodGrid,
    GridPosition,
    GridPlatform,
    CornerConnection,
)
from hexapod_grid_sim.grid.hexapod_unit import HexapodState


# ------------------------------------------------------------------
# Enums & data structures
# ------------------------------------------------------------------

class PropagationMode(Enum):
    """How constraints propagate through the grid."""
    DISABLED = "disabled"  # Platforms solve independently
    RIGID = "rigid"        # Connected corners must match exactly


@dataclass
class PropagationNode:
    """A node in the propagation graph, representing one platform."""

    position: GridPosition
    depth: int = -1          # -1 = unvisited, 0 = anchor, 1+ = BFS depth
    is_anchor: bool = False
    is_locked: bool = False  # When True the platform is not modified

    # corner_idx -> target_height received from a neighbour
    corner_constraints: Dict[int, float] = field(default_factory=dict)

    # corner_idx -> GridPosition of the neighbour that supplied the constraint
    constraint_sources: Dict[int, GridPosition] = field(default_factory=dict)

    # Solve result
    solved: bool = False
    valid: bool = False
    error_message: str = ""


@dataclass
class PropagationEdge:
    """An edge in the propagation graph -- one corner connection."""

    node_a: GridPosition
    corner_a: int
    node_b: GridPosition
    corner_b: int

    # Propagation state
    propagated: bool = False   # Has a constraint been sent along this edge?
    direction: int = 0         # 0 = none, 1 = a->b, -1 = b->a
    height_value: float = 0.0  # The height being propagated


# ------------------------------------------------------------------
# Propagation graph
# ------------------------------------------------------------------

class PropagationGraph:
    """
    Graph structure for BFS constraint propagation over a ``HexapodGrid``.

    Typical workflow::

        graph = PropagationGraph(grid)
        graph.mode = PropagationMode.RIGID
        graph.set_anchor(some_position)
        states = graph.propagate()
    """

    def __init__(self, grid: HexapodGrid) -> None:
        self.grid = grid
        self.mode: PropagationMode = PropagationMode.DISABLED

        # Build nodes from platforms
        self.nodes: Dict[GridPosition, PropagationNode] = {
            pos: PropagationNode(position=pos) for pos in grid.platforms
        }

        # Build edges from connections
        self.edges: List[PropagationEdge] = [
            PropagationEdge(
                node_a=conn.pos_a,
                corner_a=conn.corner_a,
                node_b=conn.pos_b,
                corner_b=conn.corner_b,
            )
            for conn in grid.connections
        ]

        self.anchor_pos: Optional[GridPosition] = None
        self.propagation_order: List[GridPosition] = []

    # ------------------------------------------------------------------
    # Anchor management
    # ------------------------------------------------------------------

    def set_anchor(self, pos: GridPosition) -> bool:
        """
        Designate *pos* as the propagation anchor.

        Returns ``True`` on success, ``False`` if *pos* is not in the graph.
        """
        if pos not in self.nodes:
            return False

        if self.anchor_pos is not None and self.anchor_pos in self.nodes:
            self.nodes[self.anchor_pos].is_anchor = False

        self.anchor_pos = pos
        self.nodes[pos].is_anchor = True
        return True

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all propagation state while keeping the anchor."""
        for node in self.nodes.values():
            node.depth = -1
            node.corner_constraints.clear()
            node.constraint_sources.clear()
            node.solved = False
            node.valid = False
            node.error_message = ""

        for edge in self.edges:
            edge.propagated = False
            edge.direction = 0
            edge.height_value = 0.0

        self.propagation_order.clear()

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(self) -> Dict[GridPosition, HexapodState]:
        """
        Run BFS from the anchor, propagating corner heights outward.

        Returns a dict mapping every grid position to its solved state.
        """
        self.reset()

        # When propagation is disabled or no anchor, solve independently
        if self.mode == PropagationMode.DISABLED or self.anchor_pos is None:
            return self.grid.solve_all()

        states: Dict[GridPosition, HexapodState] = {}

        # -- Initialise anchor --
        anchor_node = self.nodes[self.anchor_pos]
        anchor_node.depth = 0
        self.propagation_order.append(self.anchor_pos)

        anchor_platform = self.grid.platforms[self.anchor_pos]
        anchor_state = anchor_platform.hexapod.solve()
        states[self.anchor_pos] = anchor_state
        anchor_node.solved = True
        anchor_node.valid = anchor_state.is_valid

        if not anchor_state.is_valid:
            anchor_node.error_message = "Anchor solve failed"
            # Fall back: solve remaining platforms independently
            for pos in self.nodes:
                if pos not in states:
                    states[pos] = self.grid.platforms[pos].hexapod.solve()
            return states

        anchor_corners = self._get_corner_heights(anchor_platform, anchor_state)

        # -- BFS --
        queue: deque[GridPosition] = deque([self.anchor_pos])

        while queue:
            current_pos = queue.popleft()
            current_node = self.nodes[current_pos]
            current_platform = self.grid.platforms[current_pos]
            current_state = states.get(current_pos)
            if current_state is None:
                continue

            current_corners = self._get_corner_heights(current_platform, current_state)

            for edge in self.edges:
                neighbor_pos: Optional[GridPosition] = None
                my_corner: int = -1
                their_corner: int = -1

                if edge.node_a == current_pos:
                    neighbor_pos = edge.node_b
                    my_corner = edge.corner_a
                    their_corner = edge.corner_b
                elif edge.node_b == current_pos:
                    neighbor_pos = edge.node_a
                    my_corner = edge.corner_b
                    their_corner = edge.corner_a
                else:
                    continue

                neighbor_node = self.nodes[neighbor_pos]
                if neighbor_node.depth >= 0:
                    continue  # Already visited

                # Record edge propagation
                edge.propagated = True
                edge.direction = 1 if edge.node_a == current_pos else -1
                edge.height_value = current_corners.get(my_corner, 0.0)

                # Push corner constraint to the neighbour
                neighbor_node.corner_constraints[their_corner] = current_corners.get(my_corner, 70.0)
                neighbor_node.constraint_sources[their_corner] = current_pos

                neighbor_node.depth = current_node.depth + 1
                queue.append(neighbor_pos)
                self.propagation_order.append(neighbor_pos)

        # -- Solve non-anchor platforms with propagated constraints --
        for pos in self.propagation_order[1:]:
            node = self.nodes[pos]
            platform = self.grid.platforms[pos]

            state = self._solve_with_constraints(platform, node)
            states[pos] = state
            node.solved = True
            node.valid = state.is_valid
            if not state.is_valid:
                node.error_message = "Solve failed with constraints"

        # -- Catch any unreachable platforms --
        for pos in self.nodes:
            if pos not in states:
                platform = self.grid.platforms[pos]
                states[pos] = platform.hexapod.solve()
                self.nodes[pos].solved = True
                self.nodes[pos].valid = states[pos].is_valid

        return states

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _get_corner_heights(
        platform: GridPlatform,
        state: HexapodState,
    ) -> Dict[int, float]:
        """Return ``{corner_idx: world_Y}`` for all three corners."""
        heights: Dict[int, float] = {}
        if state.vertices is not None:
            for i in range(3):
                world_pos = platform.get_corner_world_position(i)
                if world_pos is not None:
                    heights[i] = float(world_pos[1])
        return heights

    def _solve_with_constraints(
        self,
        platform: GridPlatform,
        node: PropagationNode,
    ) -> HexapodState:
        """
        Solve a platform that has 0, 1, 2, or 3 corner-height constraints.

        Strategy varies with the number of constrained corners:
        - 0: solve normally (no propagation data).
        - 1: shift height to match the single constrained corner.
        - 2: estimate the third corner and solve via ``set_corners``.
        - 3: fully determined -- solve via ``set_corners``.
        """
        constraints = node.corner_constraints

        if len(constraints) == 0:
            return platform.hexapod.solve()

        if len(constraints) == 1:
            corner_idx, target_height = next(iter(constraints.items()))

            current_state = platform.hexapod.solve()
            if not current_state.is_valid:
                return current_state

            current_corners = self._get_corner_heights(platform, current_state)
            current_h = current_corners.get(corner_idx, 70.0)
            height_diff = target_height - current_h

            current_pose_height = (
                current_state.pose.height if current_state.pose else 70.0
            )
            platform.hexapod.set_pose(
                height=current_pose_height + height_diff, pitch=0, roll=0
            )
            return platform.hexapod.solve()

        if len(constraints) == 2:
            items = list(constraints.items())
            c1_idx, c1_height = items[0]
            c2_idx, c2_height = items[1]

            # The missing index is 0+1+2 - present indices
            c3_idx = 3 - c1_idx - c2_idx
            c3_height = (c1_height + c2_height) / 2.0

            heights = [0.0, 0.0, 0.0]
            heights[c1_idx] = c1_height
            heights[c2_idx] = c2_height
            heights[c3_idx] = c3_height

            platform.hexapod.set_corners(heights[0], heights[1], heights[2])
            return platform.hexapod.solve()

        # 3 corners -- fully constrained
        heights = [70.0, 70.0, 70.0]
        for idx, h in constraints.items():
            heights[idx] = h
        platform.hexapod.set_corners(heights[0], heights[1], heights[2])
        return platform.hexapod.solve()

    # ------------------------------------------------------------------
    # Visualisation data
    # ------------------------------------------------------------------

    def get_visualization_data(self) -> dict:
        """
        Return a dict describing the current propagation graph state,
        suitable for rendering an overlay or node-graph editor.

        Keys:
            ``nodes``, ``edges``, ``order``, ``anchor``, ``mode``
        """
        node_data: List[dict] = []
        for pos, node in self.nodes.items():
            platform = self.grid.platforms[pos]
            node_data.append({
                "position": pos,
                "world_pos": platform.world_offset.copy(),
                "depth": node.depth,
                "is_anchor": node.is_anchor,
                "is_locked": node.is_locked,
                "solved": node.solved,
                "valid": node.valid,
                "num_constraints": len(node.corner_constraints),
                "error": node.error_message,
            })

        edge_data: List[dict] = []
        for edge in self.edges:
            platform_a = self.grid.platforms[edge.node_a]
            platform_b = self.grid.platforms[edge.node_b]
            edge_data.append({
                "node_a": edge.node_a,
                "node_b": edge.node_b,
                "corner_a": edge.corner_a,
                "corner_b": edge.corner_b,
                "world_a": platform_a.world_offset.copy(),
                "world_b": platform_b.world_offset.copy(),
                "propagated": edge.propagated,
                "direction": edge.direction,
                "height": edge.height_value,
            })

        return {
            "nodes": node_data,
            "edges": edge_data,
            "order": list(self.propagation_order),
            "anchor": self.anchor_pos,
            "mode": self.mode.value,
        }
