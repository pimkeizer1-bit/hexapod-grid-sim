"""
Hexapod unit - the self-contained "black box" platform.

This is the main interface for a single hexapod. It:
- Accepts various input types (carriages, pose, corners, mixed)
- Validates constraints and reports status
- Solves kinematics using the rigid body solver
- Provides clean output for visualization

Usage:
    hexapod = HexapodUnit(geometry)

    # Various input methods:
    hexapod.set_carriages(50, 50, 50)
    hexapod.set_pose(height=60, pitch=0.1, roll=0.05)
    hexapod.set_corners(60, 55, 65)
    hexapod.set_mixed_constraints(height=60, corner_0=55)

    # Get state
    state = hexapod.solve()
    if state.is_valid:
        print(state.pose)
        print(state.carriage_positions)
"""

from __future__ import annotations

import numpy as np
from numpy import ndarray
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum

from hexapod_grid_sim.physics.solver import (
    HexapodGeometry,
    HexapodSolver,
    SolveOutput,
    SolveResult,
    PlatformPose,
)
from hexapod_grid_sim.constraints.constraint_set import (
    ConstraintType,
    ConstraintSet,
    ConstraintAnalysis,
    SolveStatus,
)


class InputMode(Enum):
    """How the hexapod is being controlled."""
    CARRIAGES = "carriages"
    POSE = "pose"
    CORNERS = "corners"
    MIXED = "mixed"


@dataclass
class HexapodState:
    """Complete state of a hexapod unit."""

    # Constraint status
    constraint_status: SolveStatus
    constraint_analysis: ConstraintAnalysis
    input_mode: InputMode

    # Solve status
    is_valid: bool
    solve_message: str

    # Platform pose (if valid)
    pose: Optional[PlatformPose]

    # Carriage state
    carriage_positions: ndarray   # [3] normalized 0-100
    carriage_distances: ndarray   # [3] actual distances from center
    carriage_world: ndarray       # [3, 3] world coordinates

    # Platform geometry (if valid)
    vertices: Optional[ndarray]   # [3, 3] world coordinates
    center: Optional[ndarray]     # [3] center position
    normal: Optional[ndarray]     # [3] up vector
    corner_heights: Optional[Tuple[float, float, float]]

    # Rod state
    rod_lengths: ndarray          # [3] actual rod lengths
    rod_error: float              # RMS error (should be ~0)

    # Geometry reference
    geometry: HexapodGeometry

    def get_visualization_data(self) -> Dict[str, Any]:
        """Get data needed for 3D visualization."""
        return {
            "is_valid": self.is_valid,
            "carriage_world": self.carriage_world.tolist(),
            "vertices": self.vertices.tolist() if self.vertices is not None else None,
            "center": self.center.tolist() if self.center is not None else None,
            "normal": self.normal.tolist() if self.normal is not None else None,
            "corner_heights": self.corner_heights,
            "rod_lengths": self.rod_lengths.tolist(),
            "constraint_status": self.constraint_status.value,
            "solve_message": self.solve_message,
        }

    def __str__(self) -> str:
        lines = [
            f"HexapodState [{self.constraint_status.value.upper()}]",
            f"  Input mode: {self.input_mode.value}",
            f"  Valid: {self.is_valid}",
        ]
        if self.pose:
            lines.extend([
                f"  Height: {self.pose.height:.2f}",
                f"  Pitch: {np.degrees(self.pose.pitch):.2f} deg",
                f"  Roll: {np.degrees(self.pose.roll):.2f} deg",
            ])
        if self.corner_heights is not None:
            h0, h1, h2 = self.corner_heights
            lines.append(f"  Corners: {h0:.1f}, {h1:.1f}, {h2:.1f}")
        lines.append(
            f"  Carriages: {self.carriage_positions[0]:.1f}, "
            f"{self.carriage_positions[1]:.1f}, "
            f"{self.carriage_positions[2]:.1f}"
        )
        if not self.is_valid:
            lines.append(f"  Message: {self.solve_message}")
        return "\n".join(lines)


class HexapodUnit:
    """
    A single hexapod unit with flexible constraint-based control.

    This is the "black box" that accepts inputs and produces outputs.
    The internal kinematics are handled automatically.
    """

    def __init__(self, geometry: HexapodGeometry) -> None:
        self.geometry = geometry
        self.solver = HexapodSolver(geometry)
        self.constraints = ConstraintSet()

        # Caching for performance
        self._dirty: bool = True
        self._cached_state: Optional[HexapodState] = None
        self._last_valid_state: Optional[HexapodState] = None

        # Default state: carriages at 50%
        self.set_carriages(50, 50, 50)

    # ------------------------------------------------------------------
    # Dirty-tracking
    # ------------------------------------------------------------------

    def _mark_dirty(self) -> None:
        """Mark that constraints changed and re-solve is needed."""
        self._dirty = True

    # ------------------------------------------------------------------
    # Constraint management
    # ------------------------------------------------------------------

    def clear_constraints(self) -> HexapodUnit:
        """Clear all constraints. Returns self for chaining."""
        self.constraints.clear()
        self._mark_dirty()
        return self

    def set_constraint(self, constraint_type: ConstraintType, value: float) -> HexapodUnit:
        """Set a single constraint. Returns self for chaining."""
        self.constraints.set(constraint_type, value)
        self._mark_dirty()
        return self

    def remove_constraint(self, constraint_type: ConstraintType) -> HexapodUnit:
        """Remove a constraint. Returns self for chaining."""
        self.constraints.remove(constraint_type)
        self._mark_dirty()
        return self

    # ------------------------------------------------------------------
    # Convenience setters
    # ------------------------------------------------------------------

    def set_carriages(self, c0: float, c1: float, c2: float) -> HexapodUnit:
        """Set all three carriage positions (0-100 normalized)."""
        self.constraints.clear()
        self.constraints.set(ConstraintType.CARRIAGE_0, c0)
        self.constraints.set(ConstraintType.CARRIAGE_1, c1)
        self.constraints.set(ConstraintType.CARRIAGE_2, c2)
        self._mark_dirty()
        return self

    def set_pose(
        self,
        height: float,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> HexapodUnit:
        """Set platform pose (height in units, pitch/roll in radians)."""
        self.constraints.clear()
        self.constraints.set(ConstraintType.HEIGHT, height)
        self.constraints.set(ConstraintType.PITCH, pitch)
        self.constraints.set(ConstraintType.ROLL, roll)
        self._mark_dirty()
        return self

    def set_corners(self, h0: float, h1: float, h2: float) -> HexapodUnit:
        """Set target corner heights."""
        self.constraints.clear()
        self.constraints.set(ConstraintType.CORNER_0, h0)
        self.constraints.set(ConstraintType.CORNER_1, h1)
        self.constraints.set(ConstraintType.CORNER_2, h2)
        self._mark_dirty()
        return self

    def set_mixed_constraints(self, **kwargs: float) -> HexapodUnit:
        """
        Set arbitrary combination of constraints.

        Keyword arguments map directly to ``ConstraintType`` values:
            carriage_0, carriage_1, carriage_2, height, pitch, roll,
            corner_0, corner_1, corner_2

        Example::

            hexapod.set_mixed_constraints(height=60, corner_0=55, roll=0.1)
        """
        self.constraints.clear()
        for name, value in kwargs.items():
            try:
                ctype = ConstraintType(name)
            except ValueError:
                raise ValueError(f"Unknown constraint: {name}") from None
            self.constraints.set(ctype, value)
        self._mark_dirty()
        return self

    # ------------------------------------------------------------------
    # Analysis helpers
    # ------------------------------------------------------------------

    def analyze_constraints(self) -> ConstraintAnalysis:
        """Analyze current constraints without solving."""
        return self.constraints.analyze()

    def _determine_input_mode(self) -> InputMode:
        """Determine input mode from active constraints."""
        active = self.constraints.active_types

        carriage_types = {
            ConstraintType.CARRIAGE_0,
            ConstraintType.CARRIAGE_1,
            ConstraintType.CARRIAGE_2,
        }
        pose_types = {
            ConstraintType.HEIGHT,
            ConstraintType.PITCH,
            ConstraintType.ROLL,
        }
        corner_types = {
            ConstraintType.CORNER_0,
            ConstraintType.CORNER_1,
            ConstraintType.CORNER_2,
        }

        if active == carriage_types:
            return InputMode.CARRIAGES
        if active == pose_types:
            return InputMode.POSE
        if active == corner_types:
            return InputMode.CORNERS
        return InputMode.MIXED

    # ------------------------------------------------------------------
    # Solving
    # ------------------------------------------------------------------

    def solve(self) -> HexapodState:
        """
        Solve kinematics based on current constraints.

        Returns a ``HexapodState`` containing constraint analysis, solve
        result, platform pose and carriage positions.
        """
        if not self._dirty and self._cached_state is not None:
            return self._cached_state

        analysis = self.analyze_constraints()
        input_mode = self._determine_input_mode()

        # Handle underdefined case
        if analysis.status == SolveStatus.UNDERDEFINED:
            return self._create_invalid_state(
                analysis,
                input_mode,
                f"Underdefined: need {analysis.missing_dof} more constraint(s)",
            )

        # Attempt solve
        try:
            solve_output = self._solve_from_constraints()
        except Exception as exc:
            return self._create_invalid_state(
                analysis, input_mode, f"Solver error: {exc}"
            )

        # Detect conflicting overdefined constraints
        if analysis.status == SolveStatus.OVERDEFINED and solve_output.error > 0.1:
            analysis = ConstraintAnalysis(
                status=SolveStatus.CONFLICTING,
                effective_dof=analysis.effective_dof,
                missing_dof=0,
                suggestions=[],
                active_constraints=analysis.active_constraints,
                redundant_constraints=analysis.redundant_constraints,
            )

        # On solver failure, fall back to last valid state or safe defaults
        # so downstream code never sees None for required array fields.
        if solve_output.result != SolveResult.SUCCESS:
            return self._create_invalid_state(
                analysis, input_mode, solve_output.message
            )

        # Build state — solver succeeded, all fields are populated
        state = HexapodState(
            constraint_status=analysis.status,
            constraint_analysis=analysis,
            input_mode=input_mode,
            is_valid=True,
            solve_message=solve_output.message,
            pose=solve_output.pose,
            carriage_positions=solve_output.carriage_positions,
            carriage_distances=solve_output.carriage_distances,
            carriage_world=solve_output.carriage_world,
            vertices=solve_output.pose.vertices,
            center=solve_output.pose.center,
            normal=solve_output.pose.normal,
            corner_heights=solve_output.pose.corner_heights,
            rod_lengths=solve_output.actual_rod_lengths,
            rod_error=solve_output.error,
            geometry=self.geometry,
        )

        if state.is_valid:
            self._last_valid_state = state

        self._cached_state = state
        self._dirty = False
        return state

    def _solve_from_constraints(self) -> SolveOutput:
        """Route to the appropriate solver based on available constraints."""
        c = self.constraints

        has_all_carriages = all(
            c.has(ct)
            for ct in (ConstraintType.CARRIAGE_0, ConstraintType.CARRIAGE_1, ConstraintType.CARRIAGE_2)
        )
        has_full_pose = all(
            c.has(ct)
            for ct in (ConstraintType.HEIGHT, ConstraintType.PITCH, ConstraintType.ROLL)
        )
        has_all_corners = all(
            c.has(ct)
            for ct in (ConstraintType.CORNER_0, ConstraintType.CORNER_1, ConstraintType.CORNER_2)
        )

        # Priority: carriages > pose > corners > mixed
        if has_all_carriages:
            return self.solver.solve_from_carriages(
                np.array([
                    c.get(ConstraintType.CARRIAGE_0),
                    c.get(ConstraintType.CARRIAGE_1),
                    c.get(ConstraintType.CARRIAGE_2),
                ])
            )

        if has_full_pose:
            return self.solver.solve_from_pose(
                c.get(ConstraintType.HEIGHT),
                c.get(ConstraintType.PITCH),
                c.get(ConstraintType.ROLL),
            )

        if has_all_corners:
            return self.solver.solve_from_corners(
                c.get(ConstraintType.CORNER_0),
                c.get(ConstraintType.CORNER_1),
                c.get(ConstraintType.CORNER_2),
            )

        return self._solve_mixed_constraints()

    def _solve_mixed_constraints(self) -> SolveOutput:
        """
        Solve with a mix of constraint types via iterative numerical refinement.

        Gathers known carriage, pose, and corner constraints and uses a simple
        gradient-descent loop to converge on a pose that satisfies all of them.
        """
        c = self.constraints
        g = self.geometry

        # Collect known values by category
        known_carriages: Dict[int, float] = {}
        known_pose: Dict[str, float] = {}
        known_corners: Dict[int, float] = {}

        for ct in (ConstraintType.CARRIAGE_0, ConstraintType.CARRIAGE_1, ConstraintType.CARRIAGE_2):
            if c.has(ct):
                idx = int(ct.value[-1])
                known_carriages[idx] = c.get(ct)

        for ct in (ConstraintType.HEIGHT, ConstraintType.PITCH, ConstraintType.ROLL):
            if c.has(ct):
                known_pose[ct.value] = c.get(ct)

        for ct in (ConstraintType.CORNER_0, ConstraintType.CORNER_1, ConstraintType.CORNER_2):
            if c.has(ct):
                idx = int(ct.value[-1])
                known_corners[idx] = c.get(ct)

        # Initial guess
        height: float = known_pose.get("height", 50.0)
        pitch: float = known_pose.get("pitch", 0.0)
        roll: float = known_pose.get("roll", 0.0)

        # Use known corners to improve the initial height guess
        if known_corners and "height" not in known_pose:
            if len(known_corners) >= 2:
                height = float(np.mean(list(known_corners.values())))

        # Iterative refinement
        for _ in range(50):
            output = self.solver.solve_from_pose(height, pitch, roll)
            if not output.pose:
                break

            errors = []
            for idx, target_h in known_corners.items():
                actual_h = output.pose.corner_heights[idx]
                errors.append(("corner", idx, actual_h - target_h))

            if not errors or max(abs(e[2]) for e in errors) < 0.01:
                break

            for etype, idx, err in errors:
                if etype == "corner":
                    if idx == 0:
                        pitch -= 0.1 * err / g.platform_radius
                    elif idx == 1:
                        pitch += 0.05 * err / g.platform_radius
                        roll -= 0.087 * err / g.platform_radius
                    else:
                        pitch += 0.05 * err / g.platform_radius
                        roll += 0.087 * err / g.platform_radius

            if known_corners and "height" not in known_pose:
                target_avg = float(np.mean(list(known_corners.values())))
                actual_avg = float(np.mean(output.pose.corner_heights))
                height += 0.5 * (target_avg - actual_avg)

        return self.solver.solve_from_pose(height, pitch, roll)

    def _create_invalid_state(
        self,
        analysis: ConstraintAnalysis,
        input_mode: InputMode,
        message: str,
    ) -> HexapodState:
        """Create an invalid state, using last valid state for visualisation if available."""
        if self._last_valid_state is not None:
            lv = self._last_valid_state
            return HexapodState(
                constraint_status=analysis.status,
                constraint_analysis=analysis,
                input_mode=input_mode,
                is_valid=False,
                solve_message=message,
                pose=lv.pose,
                carriage_positions=lv.carriage_positions,
                carriage_distances=lv.carriage_distances,
                carriage_world=lv.carriage_world,
                vertices=lv.vertices,
                center=lv.center,
                normal=lv.normal,
                corner_heights=lv.corner_heights,
                rod_lengths=lv.rod_lengths,
                rod_error=lv.rod_error,
                geometry=self.geometry,
            )

        return HexapodState(
            constraint_status=analysis.status,
            constraint_analysis=analysis,
            input_mode=input_mode,
            is_valid=False,
            solve_message=message,
            pose=None,
            carriage_positions=np.array([50.0, 50.0, 50.0]),
            carriage_distances=np.zeros(3),
            carriage_world=np.zeros((3, 3)),
            vertices=None,
            center=None,
            normal=None,
            corner_heights=None,
            rod_lengths=np.zeros(3),
            rod_error=float("inf"),
            geometry=self.geometry,
        )


# ------------------------------------------------------------------
# Factory
# ------------------------------------------------------------------

def create_hexapod(
    rail_start: float = 30.0,
    rail_length: float = 70.0,
    rod_length: float = 80.0,
    platform_radius: float = 40.0,
) -> HexapodUnit:
    """
    Create a hexapod unit with specified geometry.

    Args:
        rail_start: Distance from center to start of rails.
        rail_length: Length of each rail.
        rod_length: Length of rigid connecting rods.
        platform_radius: Distance from platform center to vertices.

    Returns:
        A configured ``HexapodUnit``.
    """
    geometry = HexapodGeometry(
        rail_start=rail_start,
        rail_length=rail_length,
        rod_length=rod_length,
        platform_radius=platform_radius,
    )
    return HexapodUnit(geometry)
