"""Tests for the hexapod grid system."""

import pytest

from hexapod_grid_sim.physics import HexapodGeometry, SolveResult
from hexapod_grid_sim.grid import HexapodGrid, GridPosition, Orientation


# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def geometry() -> HexapodGeometry:
    return HexapodGeometry(
        rail_start=30.0,
        rail_length=70.0,
        rod_length=80.0,
        platform_radius=100.0,
    )


@pytest.fixture
def grid(geometry: HexapodGeometry) -> HexapodGrid:
    return HexapodGrid(geometry, rows=3, cols=3)


# ---------------------------------------------------------------------------
# Grid creation
# ---------------------------------------------------------------------------

def test_grid_creation(grid: HexapodGrid) -> None:
    """Only UP-oriented positions get platforms."""
    # In a 3x3 checkerboard, UP positions are those where (row+col) is even.
    expected_up = sum(
        1
        for r in range(3)
        for c in range(3)
        if (r + c) % 2 == 0
    )
    assert len(grid.platforms) == expected_up
    assert all(p.orientation == Orientation.UP for p in grid.platforms.values())


# ---------------------------------------------------------------------------
# Connections
# ---------------------------------------------------------------------------

def test_connections(grid: HexapodGrid) -> None:
    """At least some corner connections exist between adjacent platforms."""
    assert len(grid.connections) > 0


# ---------------------------------------------------------------------------
# Solve all
# ---------------------------------------------------------------------------

def test_solve_all(grid: HexapodGrid) -> None:
    """All platforms in the grid solve successfully."""
    # Set every platform to a reachable pose before solving
    for platform in grid.get_all_platforms():
        platform.hexapod.set_pose(height=70, pitch=0, roll=0)

    states = grid.solve_all()

    assert len(states) == len(grid.platforms)
    for pos, state in states.items():
        assert state.is_valid, f"Platform at {pos} failed to solve"


# ---------------------------------------------------------------------------
# Dynamic add / remove
# ---------------------------------------------------------------------------

def test_add_remove_platform(geometry: HexapodGeometry) -> None:
    """Platforms can be added to and removed from the grid dynamically."""
    grid = HexapodGrid(geometry, rows=3, cols=3)
    initial_count = len(grid.platforms)

    # Find an empty slot that is adjacent to an existing platform.
    # In the checkerboard only UP positions are filled, so pick a DOWN
    # position that neighbours an existing one (e.g. row=0, col=1).
    empty_slots = grid.get_empty_slots()
    assert len(empty_slots) > 0, "No empty slots adjacent to existing platforms"

    target = empty_slots[0]
    added = grid.add_platform(target.row, target.col)
    assert added is not None
    assert len(grid.platforms) == initial_count + 1

    # Remove it
    removed = grid.remove_platform(target.row, target.col)
    assert removed is True
    assert len(grid.platforms) == initial_count
