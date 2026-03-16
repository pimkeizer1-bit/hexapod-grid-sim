"""Tests for the hexapod physics engine (solver and geometry)."""

import numpy as np
import pytest

from hexapod_grid_sim.physics import (
    HexapodGeometry,
    HexapodSolver,
    SolveResult,
)


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def geometry() -> HexapodGeometry:
    """Standard test geometry."""
    return HexapodGeometry(
        rail_start=30.0,
        rail_length=70.0,
        rod_length=80.0,
        platform_radius=40.0,
    )


@pytest.fixture
def solver(geometry: HexapodGeometry) -> HexapodSolver:
    return HexapodSolver(geometry)


# ---------------------------------------------------------------------------
# Geometry properties
# ---------------------------------------------------------------------------

def test_geometry_properties(geometry: HexapodGeometry) -> None:
    """rail_end and platform_edge_length are derived correctly."""
    assert geometry.rail_end == pytest.approx(100.0)  # 30 + 70
    assert geometry.platform_edge_length == pytest.approx(
        40.0 * np.sqrt(3.0)
    )


# ---------------------------------------------------------------------------
# Height range
# ---------------------------------------------------------------------------

def test_height_range(geometry: HexapodGeometry) -> None:
    """compute_height_range returns a valid (min, max) pair."""
    h_min, h_max = geometry.compute_height_range()
    assert h_min >= 0.0
    assert h_max > h_min
    assert h_max <= geometry.rod_length


# ---------------------------------------------------------------------------
# Forward kinematics
# ---------------------------------------------------------------------------

def test_forward_kinematics(solver: HexapodSolver) -> None:
    """Forward solve at centred carriages yields a valid, low-error result."""
    carriages = np.array([0.5, 0.5, 0.5])
    output = solver.solve_from_carriages(carriages)

    assert output.result == SolveResult.SUCCESS
    assert output.pose is not None
    assert output.error < 1e-6
    assert output.pose.height > 0.0


# ---------------------------------------------------------------------------
# Inverse kinematics
# ---------------------------------------------------------------------------

def test_inverse_kinematics(solver: HexapodSolver, geometry: HexapodGeometry) -> None:
    """Inverse solve at a reachable height returns valid carriages in range."""
    output = solver.solve_from_pose(height=70.0, pitch=0.0, roll=0.0)

    assert output.result == SolveResult.SUCCESS
    assert output.carriage_distances is not None
    assert all(
        geometry.rail_start <= d <= geometry.rail_end
        for d in output.carriage_distances
    )


# ---------------------------------------------------------------------------
# Round-trip consistency
# ---------------------------------------------------------------------------

def test_roundtrip(solver: HexapodSolver) -> None:
    """Forward -> pose -> inverse -> compare carriages."""
    original = np.array([0.4, 0.5, 0.6])
    fwd = solver.solve_from_carriages(original)
    assert fwd.result == SolveResult.SUCCESS

    inv = solver.solve_from_pose(
        height=fwd.pose.height,
        pitch=fwd.pose.pitch,
        roll=fwd.pose.roll,
    )
    assert inv.result == SolveResult.SUCCESS

    np.testing.assert_allclose(
        inv.carriage_positions, original, atol=1e-4
    )


# ---------------------------------------------------------------------------
# Corner-height solve
# ---------------------------------------------------------------------------

def test_corner_solve(solver: HexapodSolver) -> None:
    """Equal corner heights produce a level platform."""
    h = 70.0
    output = solver.solve_from_corners(h, h, h)

    assert output.result == SolveResult.SUCCESS
    assert output.pose is not None
    assert output.pose.pitch == pytest.approx(0.0, abs=1e-4)
    assert output.pose.roll == pytest.approx(0.0, abs=1e-4)
    assert output.pose.height == pytest.approx(h, abs=0.5)


# ---------------------------------------------------------------------------
# Invalid / extreme pose
# ---------------------------------------------------------------------------

def test_invalid_pose(solver: HexapodSolver) -> None:
    """An unreachable height returns NO_SOLUTION."""
    output = solver.solve_from_pose(height=999.0, pitch=0.0, roll=0.0)
    assert output.result == SolveResult.NO_SOLUTION
