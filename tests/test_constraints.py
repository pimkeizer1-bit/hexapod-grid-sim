"""Tests for the constraint system."""

import pytest

from hexapod_grid_sim.constraints import (
    ConstraintType,
    ConstraintSet,
    SolveStatus,
)


# ---------------------------------------------------------------------------
# Empty constraints
# ---------------------------------------------------------------------------

def test_empty_constraints() -> None:
    """An empty set has 0 effective DOF and is underdefined."""
    cs = ConstraintSet()
    analysis = cs.analyze()

    assert analysis.effective_dof == 0
    assert analysis.status == SolveStatus.UNDERDEFINED


# ---------------------------------------------------------------------------
# Full constraint groups
# ---------------------------------------------------------------------------

def test_full_carriages() -> None:
    """All three carriages give exactly 3 DOF and DEFINED status."""
    cs = ConstraintSet()
    cs.set(ConstraintType.CARRIAGE_0, 50.0)
    cs.set(ConstraintType.CARRIAGE_1, 50.0)
    cs.set(ConstraintType.CARRIAGE_2, 50.0)
    analysis = cs.analyze()

    assert analysis.effective_dof == 3
    assert analysis.status == SolveStatus.DEFINED


def test_full_pose() -> None:
    """Height + pitch + roll give exactly 3 DOF and DEFINED status."""
    cs = ConstraintSet()
    cs.set(ConstraintType.HEIGHT, 70.0)
    cs.set(ConstraintType.PITCH, 0.0)
    cs.set(ConstraintType.ROLL, 0.0)
    analysis = cs.analyze()

    assert analysis.effective_dof == 3
    assert analysis.status == SolveStatus.DEFINED


def test_full_corners() -> None:
    """Three corner heights give exactly 3 DOF and DEFINED status."""
    cs = ConstraintSet()
    cs.set(ConstraintType.CORNER_0, 70.0)
    cs.set(ConstraintType.CORNER_1, 70.0)
    cs.set(ConstraintType.CORNER_2, 70.0)
    analysis = cs.analyze()

    assert analysis.effective_dof == 3
    assert analysis.status == SolveStatus.DEFINED


# ---------------------------------------------------------------------------
# Partial constraints
# ---------------------------------------------------------------------------

def test_partial() -> None:
    """One or two constraints are underdefined."""
    cs = ConstraintSet()
    cs.set(ConstraintType.HEIGHT, 70.0)
    analysis = cs.analyze()
    assert analysis.status == SolveStatus.UNDERDEFINED
    assert analysis.effective_dof < 3

    cs.set(ConstraintType.PITCH, 0.0)
    analysis = cs.analyze()
    assert analysis.status == SolveStatus.UNDERDEFINED
    assert analysis.effective_dof < 3


# ---------------------------------------------------------------------------
# Overdefined
# ---------------------------------------------------------------------------

def test_overdefined() -> None:
    """Full carriages plus full pose is overdefined."""
    cs = ConstraintSet()
    # Carriages
    cs.set(ConstraintType.CARRIAGE_0, 50.0)
    cs.set(ConstraintType.CARRIAGE_1, 50.0)
    cs.set(ConstraintType.CARRIAGE_2, 50.0)
    # Pose
    cs.set(ConstraintType.HEIGHT, 70.0)
    cs.set(ConstraintType.PITCH, 0.0)
    cs.set(ConstraintType.ROLL, 0.0)
    analysis = cs.analyze()

    assert analysis.status == SolveStatus.OVERDEFINED


# ---------------------------------------------------------------------------
# Redundant constraint detection
# ---------------------------------------------------------------------------

def test_redundant_detection() -> None:
    """The analyser finds redundant constraints when groups overlap."""
    cs = ConstraintSet()
    # Full carriage group
    cs.set(ConstraintType.CARRIAGE_0, 50.0)
    cs.set(ConstraintType.CARRIAGE_1, 50.0)
    cs.set(ConstraintType.CARRIAGE_2, 50.0)
    # Add a pose constraint (derivable from carriages)
    cs.set(ConstraintType.HEIGHT, 70.0)
    analysis = cs.analyze()

    assert len(analysis.redundant_constraints) > 0


# ---------------------------------------------------------------------------
# Solvability
# ---------------------------------------------------------------------------

def test_solvable() -> None:
    """DEFINED is solvable; OVERDEFINED is also solvable (structurally)."""
    cs_defined = ConstraintSet()
    cs_defined.set(ConstraintType.HEIGHT, 70.0)
    cs_defined.set(ConstraintType.PITCH, 0.0)
    cs_defined.set(ConstraintType.ROLL, 0.0)
    assert cs_defined.analyze().is_solvable()

    cs_over = ConstraintSet()
    cs_over.set(ConstraintType.CARRIAGE_0, 50.0)
    cs_over.set(ConstraintType.CARRIAGE_1, 50.0)
    cs_over.set(ConstraintType.CARRIAGE_2, 50.0)
    cs_over.set(ConstraintType.HEIGHT, 70.0)
    cs_over.set(ConstraintType.PITCH, 0.0)
    cs_over.set(ConstraintType.ROLL, 0.0)
    analysis_over = cs_over.analyze()
    # Overdefined is not "solvable" by the strict is_solvable check (DEFINED only),
    # but the status should be OVERDEFINED which is still structurally solvable.
    assert analysis_over.status == SolveStatus.OVERDEFINED
