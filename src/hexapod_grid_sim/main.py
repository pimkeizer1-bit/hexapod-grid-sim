"""Hexapod Simulator -- single-unit entry point.

Launches a real-time 3D visualization of one hexapod platform
with interactive GUI controls for carriage, pose, and corner modes.

Usage:
    python -m hexapod_grid_sim.main
"""

import taichi as ti

# Initialize Taichi with CUDA backend
ti.init(arch=ti.cuda)

from hexapod_grid_sim.grid import create_hexapod
from hexapod_grid_sim.visualization.single_viewer import HexapodViewer


def main() -> None:
    """Run the single-hexapod simulation loop."""
    print("Hexapod Simulator")
    print("=================")
    print("Controls:")
    print("  - Right-click + drag: Rotate camera")
    print("  - Scroll: Zoom")
    print("  - Use GUI buttons to switch control modes")
    print()

    # Create hexapod with default geometry
    hexapod = create_hexapod(
        rail_start=30.0,
        rail_length=70.0,
        rod_length=80.0,
        platform_radius=100.0,
    )

    # Create viewer
    viewer = HexapodViewer(width=1200, height=800)

    # Sync viewer geometry with hexapod
    viewer.rail_start = hexapod.geometry.rail_start
    viewer.rail_length = hexapod.geometry.rail_length
    viewer.rod_length = hexapod.geometry.rod_length
    viewer.platform_radius = hexapod.geometry.platform_radius

    # Main loop
    while not viewer.should_close():
        viewer.apply_controls_to_hexapod(hexapod)

        state = hexapod.solve()

        viewer.update_from_state(state)
        viewer.render()
        viewer.show_gui()
        viewer.finish_frame()


if __name__ == "__main__":
    main()
