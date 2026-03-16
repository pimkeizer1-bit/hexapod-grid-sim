"""Constraint system for a 3-DOF hexapod grid simulator.

Manages input constraints and determines solve status for a hexapod platform
with three degrees of freedom (height, pitch, roll) controlled by three
linear actuators (carriages).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, FrozenSet, List, Optional, Set


class ConstraintType(Enum):
    """Types of constraints that can be applied to the hexapod platform."""

    # Direct actuator control
    CARRIAGE_0 = "carriage_0"
    CARRIAGE_1 = "carriage_1"
    CARRIAGE_2 = "carriage_2"

    # Platform pose
    HEIGHT = "height"
    PITCH = "pitch"
    ROLL = "roll"

    # Individual vertex (corner) heights
    CORNER_0 = "corner_0"
    CORNER_1 = "corner_1"
    CORNER_2 = "corner_2"


class SolveStatus(Enum):
    """Status of the constraint system's solvability."""

    UNDERDEFINED = "underdefined"
    DEFINED = "defined"
    OVERDEFINED = "overdefined"
    CONFLICTING = "conflicting"


# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

CONSTRAINT_GROUPS: Dict[str, Set[ConstraintType]] = {
    "actuators": {ConstraintType.CARRIAGE_0, ConstraintType.CARRIAGE_1, ConstraintType.CARRIAGE_2},
    "pose": {ConstraintType.HEIGHT, ConstraintType.PITCH, ConstraintType.ROLL},
    "corners": {ConstraintType.CORNER_0, ConstraintType.CORNER_1, ConstraintType.CORNER_2},
}

# When a full group is specified, the other two groups can be derived from it.
DERIVABLE_FROM: Dict[FrozenSet[str], Set[str]] = {
    frozenset({"actuators"}): {"pose", "corners"},
    frozenset({"pose"}): {"corners", "actuators"},
    frozenset({"corners"}): {"pose", "actuators"},
}


# ---------------------------------------------------------------------------
# Helper: reverse lookup — which group does a constraint type belong to?
# ---------------------------------------------------------------------------

_TYPE_TO_GROUP: Dict[ConstraintType, str] = {}
for _group_name, _members in CONSTRAINT_GROUPS.items():
    for _ct in _members:
        _TYPE_TO_GROUP[_ct] = _group_name


@dataclass
class Constraint:
    """A single constraint binding a ConstraintType to a numeric value."""

    type: ConstraintType
    value: float


@dataclass
class ConstraintAnalysis:
    """Result of analysing a ConstraintSet."""

    status: SolveStatus
    effective_dof: int
    missing_dof: int
    suggestions: List[str]
    active_constraints: Set[ConstraintType]
    redundant_constraints: Set[ConstraintType]

    def is_solvable(self) -> bool:
        """Return True when the system is exactly defined (3 effective DOF)."""
        return self.status == SolveStatus.DEFINED

    def __str__(self) -> str:
        active = ", ".join(sorted(c.name for c in self.active_constraints)) or "(none)"
        redundant = ", ".join(sorted(c.name for c in self.redundant_constraints)) or "(none)"
        lines = [
            f"Status:        {self.status.name}",
            f"Effective DOF: {self.effective_dof} / 3",
            f"Missing DOF:   {self.missing_dof}",
            f"Active:        {active}",
            f"Redundant:     {redundant}",
        ]
        if self.suggestions:
            lines.append("Suggestions:")
            for suggestion in self.suggestions:
                lines.append(f"  - {suggestion}")
        return "\n".join(lines)


class ConstraintSet:
    """Mutable collection of constraints for a 3-DOF hexapod platform.

    Provides a fluent interface — mutating methods return *self* so calls
    can be chained:

        cs = ConstraintSet()
        cs.set(ConstraintType.HEIGHT, 10.0).set(ConstraintType.PITCH, 0.0)
    """

    _DOF: int = 3  # total degrees of freedom for the platform

    def __init__(self) -> None:
        self.constraints: Dict[ConstraintType, float] = {}

    # ------------------------------------------------------------------
    # Fluent mutators
    # ------------------------------------------------------------------

    def set(self, constraint_type: ConstraintType, value: float) -> ConstraintSet:
        """Set (or overwrite) a constraint. Returns *self* for chaining."""
        self.constraints[constraint_type] = value
        return self

    def remove(self, constraint_type: ConstraintType) -> ConstraintSet:
        """Remove a constraint if present. Returns *self* for chaining."""
        self.constraints.pop(constraint_type, None)
        return self

    def clear(self) -> ConstraintSet:
        """Remove all constraints. Returns *self* for chaining."""
        self.constraints.clear()
        return self

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get(self, constraint_type: ConstraintType) -> Optional[float]:
        """Return the value for *constraint_type*, or ``None``."""
        return self.constraints.get(constraint_type)

    def has(self, constraint_type: ConstraintType) -> bool:
        """Return whether *constraint_type* is currently set."""
        return constraint_type in self.constraints

    @property
    def active_types(self) -> Set[ConstraintType]:
        """The set of constraint types that are currently set."""
        return set(self.constraints.keys())

    # ------------------------------------------------------------------
    # DOF / analysis helpers
    # ------------------------------------------------------------------

    def _complete_groups(self) -> Set[str]:
        """Return the names of groups for which every member is constrained."""
        active = self.active_types
        return {
            name
            for name, members in CONSTRAINT_GROUPS.items()
            if members <= active
        }

    def _derivable_groups(self) -> Set[str]:
        """Return group names that are derivable from complete groups."""
        complete = frozenset(self._complete_groups())
        derivable: Set[str] = set()
        for source_groups, derived in DERIVABLE_FROM.items():
            if source_groups <= complete:
                derivable |= derived
        return derivable

    def count_effective_dof(self) -> int:
        """Count the effective degrees of freedom consumed by constraints.

        * Each complete group accounts for exactly 3 DOF.
        * Additional constraints from a *derivable* group are redundant and
          do not add DOF.
        * Partial constraints from non-derivable groups each add 1 DOF.
        """
        if not self.constraints:
            return 0

        active = self.active_types
        complete = self._complete_groups()
        derivable = self._derivable_groups()

        # Start with 3 DOF if at least one group is complete.
        if complete:
            dof = self._DOF
        else:
            dof = 0

        for group_name, members in CONSTRAINT_GROUPS.items():
            present = members & active
            if not present:
                continue

            if group_name in complete:
                # Already counted as part of the base 3.
                continue

            if group_name in derivable:
                # These are redundant — they can be derived from a complete
                # group, so they add no independent DOF.
                continue

            # Partial, non-derivable group: each member adds 1 DOF.
            dof += len(present)

        return dof

    def _find_redundant_constraints(self) -> Set[ConstraintType]:
        """Return constraint types that are redundant given the current set."""
        active = self.active_types
        complete = self._complete_groups()
        derivable = self._derivable_groups()
        redundant: Set[ConstraintType] = set()

        for group_name, members in CONSTRAINT_GROUPS.items():
            if group_name in derivable and group_name not in complete:
                # Any constraint in a derivable (but not itself complete)
                # group is redundant.
                redundant |= members & active
            elif group_name in derivable and group_name in complete:
                # The entire group is derivable AND complete — the whole
                # group is redundant (the DOF is already covered).
                redundant |= members & active

        return redundant

    def _suggest_missing_constraints(self) -> List[str]:
        """Return human-readable suggestions for reaching a solvable state."""
        effective = self.count_effective_dof()
        if effective >= self._DOF:
            return []

        needed = self._DOF - effective
        active = self.active_types
        complete = self._complete_groups()
        derivable = self._derivable_groups()
        suggestions: List[str] = []

        for group_name, members in CONSTRAINT_GROUPS.items():
            if group_name in complete or group_name in derivable:
                continue
            missing = members - active
            if not missing:
                continue
            missing_names = ", ".join(sorted(m.name for m in missing))
            present_count = len(members & active)
            remaining = len(missing)
            if present_count > 0:
                suggestions.append(
                    f"Complete the '{group_name}' group by adding: {missing_names} "
                    f"({remaining} more needed)"
                )
            else:
                suggestions.append(
                    f"Add constraints from the '{group_name}' group: {missing_names}"
                )

        if not suggestions and needed > 0:
            suggestions.append(f"Add {needed} more independent constraint(s)")

        return suggestions

    # ------------------------------------------------------------------
    # Main analysis entry-point
    # ------------------------------------------------------------------

    def analyze(self) -> ConstraintAnalysis:
        """Analyse the current constraint set and return a ConstraintAnalysis."""
        effective = self.count_effective_dof()
        missing = max(0, self._DOF - effective)
        redundant = self._find_redundant_constraints()
        active = self.active_types

        # Determine status
        if effective < self._DOF:
            status = SolveStatus.UNDERDEFINED
        elif effective == self._DOF and not redundant:
            status = SolveStatus.DEFINED
        elif effective == self._DOF and redundant:
            # We have enough DOF but also redundant constraints. If the
            # total raw constraint count exceeds 3, we are overdefined.
            # (Conflicting would require value-level checks which are
            # outside the scope of this structural analysis.)
            status = SolveStatus.OVERDEFINED
        else:
            # effective > _DOF — more independent constraints than DOF
            status = SolveStatus.CONFLICTING

        suggestions = self._suggest_missing_constraints() if not (
            status == SolveStatus.DEFINED
        ) else []

        if status == SolveStatus.OVERDEFINED:
            redundant_names = ", ".join(sorted(c.name for c in redundant))
            suggestions.append(
                f"Remove redundant constraint(s) to reach DEFINED: {redundant_names}"
            )

        return ConstraintAnalysis(
            status=status,
            effective_dof=effective,
            missing_dof=missing,
            suggestions=suggestions,
            active_constraints=active,
            redundant_constraints=redundant,
        )
