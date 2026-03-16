# Hexapod Grid Simulator

GPU-accelerated 3-DOF hexapod parallel platform simulator for studying emergent behavior in hexapod grids.

## Features

- **Rigid body kinematics** — forward/inverse/corner-based solving with Newton-Raphson
- **Constraint system** — flexible DOF tracking with mixed constraint support
- **Grid topology** — triangular tiling with corner-to-corner connections
- **Constraint propagation** — BFS-based height propagation from anchor platforms
- **3D visualization** — real-time Taichi GGUI rendering with interactive controls
- **Visual scripting** — DearPyGUI node editor for constraint/control logic
- **Tactical ops center** — system metrics dashboard with sci-fi styling

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (optimized for RTX 3090)

## Installation

```bash
pip install -e ".[dev]"
```

## Usage

```bash
# Single hexapod viewer
python -m hexapod_grid_sim.main

# Grid simulation
python -m hexapod_grid_sim.main_grid
```

## Controls

| Key | Action |
|-----|--------|
| RMB + drag | Orbit camera |
| MMB + drag | Pan viewport |
| Scroll / W/S | Zoom |
| A/D | Lateral pan |
| Q/E | Vertical pan |
| LMB | Select/deploy unit |

## Architecture

```
src/hexapod_grid_sim/
├── physics/          # Rigid body solver, kinematics
├── constraints/      # DOF tracking, constraint sets
├── grid/             # Hexapod unit, topology, propagation
├── visualization/    # Taichi GGUI 3D renderers
├── ui/               # Node editor, tactical ops, window config
├── config/           # Simulation parameters
├── main.py           # Single hexapod entry point
└── main_grid.py      # Grid simulation entry point
```

## Testing

```bash
pytest
```
