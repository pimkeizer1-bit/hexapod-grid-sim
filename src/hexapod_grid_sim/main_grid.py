"""Hexapod Grid Simulator -- multi-unit tactical entry point.

Launches a grid of hexapod platforms with neighbour connections
and real-time 3D visualization.

Usage:
    python -m hexapod_grid_sim.main_grid
"""

import taichi as ti

# Initialize Taichi with CUDA backend
ti.init(arch=ti.cuda)

from hexapod_grid_sim.physics import HexapodGeometry
from hexapod_grid_sim.grid import HexapodGrid
from hexapod_grid_sim.visualization.grid_viewer import GridViewer


def main() -> None:
    """Run the grid simulation loop."""
    print()
    print("+============================================================+")
    print("|     <<< SWARM OPERATIONS CENTER >>>                        |")
    print("|     HEXAPOD TACTICAL CONTROL SYSTEM v2.0                   |")
    print("+============================================================+")
    print("|  INITIALIZATION SEQUENCE ACTIVE...                         |")
    print("+============================================================+")
    print()
    print("/// OPERATOR INTERFACE ///")
    print("  [RMB + DRAG]  Orbit surveillance")
    print("  [MMB + DRAG]  Pan viewport")
    print("  [SCROLL/W/S]  Zoom control")
    print("  [A/D]         Lateral pan")
    print("  [Q/E]         Vertical pan")
    print("  [LMB]         Deploy unit at target")
    print("  [GUI]         Master control panel")
    print()

    # Create geometry (shared by all hexapods)
    geometry = HexapodGeometry(
        rail_start=30.0,
        rail_length=70.0,
        rod_length=80.0,
        platform_radius=100.0,
    )

    # Create grid
    grid = HexapodGrid(geometry, rows=3, cols=3)
    print(f">>> DEPLOYING SWARM MATRIX: {grid.rows}x{grid.cols}")
    print(f">>> UNITS ONLINE: {len(grid.platforms)}")
    print(f">>> NEURAL LINKS ESTABLISHED: {len(grid.connections)}")

    # Initialize all platforms to standby height
    for platform in grid.get_all_platforms():
        platform.hexapod.set_pose(height=70, pitch=0, roll=0)
    print(">>> ALL UNITS: STANDBY POSITION")
    print()

    # Create viewer
    print("/// INITIALIZING VISUAL INTERFACE ///")
    viewer = GridViewer(width=1400, height=900)

    # Main loop
    while not viewer.should_close():
        viewer.update_from_grid(grid)
        viewer.render()
        viewer.show_gui(grid)
        viewer.finish_frame()


if __name__ == "__main__":
    main()
