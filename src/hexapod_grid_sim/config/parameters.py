"""Global simulation parameters for the hexapod grid simulator."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HexapodParams:
    """Parameters for a single hexapod unit."""

    rail_start_distance: float = 50.0
    rail_length: float = 100.0
    rod_length: float = 80.0
    platform_radius: float = 40.0


@dataclass
class SimulationConfig:
    """Global simulation configuration."""

    # Hexapod geometry
    hexapod: HexapodParams = field(default_factory=HexapodParams)

    # Grid settings
    grid_rows: int = 3
    grid_cols: int = 3
    grid_spacing: float = 200.0

    # Default carriage positions (0-100)
    default_carriage: float = 50.0

    # Display settings
    window_width: int = 1400
    window_height: int = 900
