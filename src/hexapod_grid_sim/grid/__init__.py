"""Grid module -- hexapod units, topology, and constraint propagation."""

from .hexapod_unit import HexapodUnit, HexapodState, InputMode, create_hexapod
from .topology import HexapodGrid, GridPosition, GridPlatform, Orientation, CornerConnection
from .propagation import PropagationGraph, PropagationMode, PropagationNode

__all__ = [
    "HexapodUnit",
    "HexapodState",
    "InputMode",
    "create_hexapod",
    "HexapodGrid",
    "GridPosition",
    "GridPlatform",
    "Orientation",
    "CornerConnection",
    "PropagationGraph",
    "PropagationMode",
    "PropagationNode",
]
