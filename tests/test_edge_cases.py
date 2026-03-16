"""Edge-case and boundary-condition tests.

Covers carriage limits (0/100), extreme geometry, solver failure
propagation, and ensures the HexapodUnit never exposes None arrays.
"""

import numpy as np
import pytest

from hexapod_grid_sim.physics import HexapodGeometry, HexapodSolver, SolveResult
from hexapod_grid_sim.grid.hexapod_unit import HexapodUnit, HexapodState


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def geometry() -> HexapodGeometry:
    return HexapodGeometry(
        rail_start=30.0, rail_length=70.0,
        rod_length=80.0, platform_radius=40.0,
    )


@pytest.fixture
def solver(geometry: HexapodGeometry) -> HexapodSolver:
    return HexapodSolver(geometry)


@pytest.fixture
def unit(geometry: HexapodGeometry) -> HexapodUnit:
    return HexapodUnit(geometry)


# ---------------------------------------------------------------------------
# Carriage boundary values
# ---------------------------------------------------------------------------

class TestCarriageBoundaries:
    """Carriages at their 0 and 100 extremes."""

    def test_carriages_at_zero(self, solver: HexapodSolver) -> None:
        """Carriages at 0 (minimum) should solve or fail gracefully."""
        out = solver.solve_from_carriages(np.array([0.0, 0.0, 0.0]))
        # Might succeed or fail depending on geometry, but must not crash
        assert out.result in (SolveResult.SUCCESS, SolveResult.NO_SOLUTION,
                              SolveResult.NUMERICAL_ERROR)

    def test_carriages_at_100(self, solver: HexapodSolver) -> None:
        """Carriages at 100 (maximum) should solve or fail gracefully."""
        out = solver.solve_from_carriages(np.array([100.0, 100.0, 100.0]))
        assert out.result in (SolveResult.SUCCESS, SolveResult.NO_SOLUTION,
                              SolveResult.NUMERICAL_ERROR)

    def test_carriages_mixed_extremes(self, solver: HexapodSolver) -> None:
        """One carriage at 0, others at 100."""
        out = solver.solve_from_carriages(np.array([0.0, 100.0, 100.0]))
        assert out.result in (SolveResult.SUCCESS, SolveResult.NO_SOLUTION,
                              SolveResult.NUMERICAL_ERROR)

    def test_carriages_at_zero_via_unit(self, unit: HexapodUnit) -> None:
        """HexapodUnit with carriages at 0 returns valid state with no None arrays."""
        unit.set_carriages(0, 0, 0)
        state = unit.solve()
        _assert_state_safe(state)

    def test_carriages_at_100_via_unit(self, unit: HexapodUnit) -> None:
        """HexapodUnit with carriages at 100 returns valid state with no None arrays."""
        unit.set_carriages(100, 100, 100)
        state = unit.solve()
        _assert_state_safe(state)

    def test_carriages_mixed_extremes_via_unit(self, unit: HexapodUnit) -> None:
        """HexapodUnit with mixed extreme carriages returns safe state."""
        unit.set_carriages(0, 100, 50)
        state = unit.solve()
        _assert_state_safe(state)


# ---------------------------------------------------------------------------
# Extreme geometry parameters
# ---------------------------------------------------------------------------

class TestExtremeGeometry:
    """Geometry where the solver is expected to fail."""

    def test_rod_too_short(self) -> None:
        """Rod shorter than platform radius — solver can't close the loop."""
        geo = HexapodGeometry(
            rail_start=30.0, rail_length=70.0,
            rod_length=10.0, platform_radius=100.0,
        )
        unit = HexapodUnit(geo)
        unit.set_carriages(50, 50, 50)
        state = unit.solve()
        _assert_state_safe(state)
        # The solver should flag this as invalid
        assert not state.is_valid

    def test_very_small_platform(self) -> None:
        """Tiny platform radius — should still work."""
        geo = HexapodGeometry(
            rail_start=30.0, rail_length=70.0,
            rod_length=80.0, platform_radius=1.0,
        )
        unit = HexapodUnit(geo)
        unit.set_carriages(50, 50, 50)
        state = unit.solve()
        _assert_state_safe(state)

    def test_very_large_platform(self) -> None:
        """Platform radius larger than rod — typically unsolvable."""
        geo = HexapodGeometry(
            rail_start=30.0, rail_length=70.0,
            rod_length=40.0, platform_radius=200.0,
        )
        unit = HexapodUnit(geo)
        unit.set_carriages(50, 50, 50)
        state = unit.solve()
        _assert_state_safe(state)
        assert not state.is_valid


# ---------------------------------------------------------------------------
# Solver failure propagation
# ---------------------------------------------------------------------------

class TestSolverFailurePropagation:
    """Ensure solver failures never leak None arrays into HexapodState."""

    def test_unreachable_height(self) -> None:
        """Pose solve with impossible height."""
        geo = HexapodGeometry(
            rail_start=30.0, rail_length=70.0,
            rod_length=80.0, platform_radius=40.0,
        )
        unit = HexapodUnit(geo)
        unit.set_pose(height=999.0, pitch=0.0, roll=0.0)
        state = unit.solve()
        _assert_state_safe(state)
        assert not state.is_valid

    def test_failure_then_success(self) -> None:
        """Recover from a failed solve back to a valid one."""
        geo = HexapodGeometry(
            rail_start=30.0, rail_length=70.0,
            rod_length=80.0, platform_radius=40.0,
        )
        unit = HexapodUnit(geo)

        # First: valid solve
        unit.set_carriages(50, 50, 50)
        state1 = unit.solve()
        assert state1.is_valid

        # Second: trigger failure
        unit.set_pose(height=999.0, pitch=0.0, roll=0.0)
        state2 = unit.solve()
        _assert_state_safe(state2)
        assert not state2.is_valid

        # Third: recover
        unit.set_carriages(50, 50, 50)
        state3 = unit.solve()
        assert state3.is_valid
        _assert_state_safe(state3)

    def test_first_solve_failure(self) -> None:
        """Very first solve is a failure (no last_valid_state yet)."""
        geo = HexapodGeometry(
            rail_start=30.0, rail_length=70.0,
            rod_length=10.0, platform_radius=100.0,
        )
        unit = HexapodUnit(geo)
        unit.constraints.clear()
        unit.set_pose(height=999.0, pitch=0.0, roll=0.0)
        state = unit.solve()
        _assert_state_safe(state)
        assert not state.is_valid


# ---------------------------------------------------------------------------
# State serialization safety
# ---------------------------------------------------------------------------

class TestStateSerialization:
    """get_visualization_data and __str__ must not crash."""

    def test_valid_state_serializes(self, unit: HexapodUnit) -> None:
        unit.set_carriages(50, 50, 50)
        state = unit.solve()
        data = state.get_visualization_data()
        assert isinstance(data, dict)
        assert data["is_valid"] is True
        str(state)  # must not raise

    def test_invalid_state_serializes(self) -> None:
        geo = HexapodGeometry(
            rail_start=30.0, rail_length=70.0,
            rod_length=10.0, platform_radius=100.0,
        )
        unit = HexapodUnit(geo)
        unit.set_carriages(50, 50, 50)
        state = unit.solve()
        data = state.get_visualization_data()
        assert isinstance(data, dict)
        assert data["is_valid"] is False
        str(state)  # must not raise


# ---------------------------------------------------------------------------
# Normalization range
# ---------------------------------------------------------------------------

class TestNormalizationRange:
    """Verify [0, 100] normalization round-trips correctly."""

    def test_normalize_0(self, geometry: HexapodGeometry) -> None:
        d = geometry.carriage_normalized_to_distance(0.0)
        assert d == pytest.approx(geometry.rail_start)

    def test_normalize_100(self, geometry: HexapodGeometry) -> None:
        d = geometry.carriage_normalized_to_distance(100.0)
        assert d == pytest.approx(geometry.rail_end)

    def test_normalize_50(self, geometry: HexapodGeometry) -> None:
        d = geometry.carriage_normalized_to_distance(50.0)
        mid = geometry.rail_start + geometry.rail_length * 0.5
        assert d == pytest.approx(mid)

    def test_denormalize_roundtrip(self, geometry: HexapodGeometry) -> None:
        for val in [0.0, 25.0, 50.0, 75.0, 100.0]:
            d = geometry.carriage_normalized_to_distance(val)
            back = geometry.carriage_distance_to_normalized(d)
            assert back == pytest.approx(val, abs=1e-10)

    def test_clamp_below_zero(self, geometry: HexapodGeometry) -> None:
        d = geometry.carriage_normalized_to_distance(-10.0)
        assert d == pytest.approx(geometry.rail_start)

    def test_clamp_above_100(self, geometry: HexapodGeometry) -> None:
        d = geometry.carriage_normalized_to_distance(110.0)
        assert d == pytest.approx(geometry.rail_end)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_state_safe(state: HexapodState) -> None:
    """Assert that a HexapodState has no None values in its required array fields."""
    assert state.carriage_positions is not None, "carriage_positions is None"
    assert state.carriage_distances is not None, "carriage_distances is None"
    assert state.carriage_world is not None, "carriage_world is None"
    assert state.rod_lengths is not None, "rod_lengths is None"

    # Check shapes
    assert state.carriage_positions.shape == (3,)
    assert state.carriage_distances.shape == (3,)
    assert state.carriage_world.shape == (3, 3)
    assert state.rod_lengths.shape == (3,)
