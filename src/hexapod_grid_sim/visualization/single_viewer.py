"""Interactive 3D viewer for a single hexapod using Taichi GGUI.

Provides real-time visualization of a hexapod parallel platform with:
- Three control modes: carriages, pose (height/pitch/roll), corners
- Adjustable geometry parameters (rail start, rail length, platform radius, rod length)
- Color-coded validity feedback (blue=valid, red=invalid)
- Y-up coordinate system (Taichi default)
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import taichi as ti
import taichi.math as tm

from hexapod_grid_sim.grid.hexapod_unit import HexapodState, HexapodUnit
from hexapod_grid_sim.physics.solver import HexapodGeometry
from hexapod_grid_sim.constraints.constraint_set import SolveStatus


@ti.data_oriented
class HexapodViewer:
    """Real-time 3D visualization of a single hexapod platform.

    Uses Taichi GGUI with a Y-up coordinate system.  Three rails lie on the
    XZ ground plane at 0 deg, 120 deg, and 240 deg from the X axis.  The
    triangular platform floats above, connected to carriages by rigid rods.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(self, width: int = 1200, height: int = 800) -> None:
        self.width: int = width
        self.height: int = height

        # --- Taichi GGUI objects ---
        self.window = ti.ui.Window("Hexapod Viewer", (width, height), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()

        # Default camera (Y-up)
        self.camera.position(150.0, 100.0, 150.0)
        self.camera.lookat(0.0, 50.0, 0.0)

        # --- Line geometry buffers ---
        self.max_lines: int = 100
        self.line_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.max_lines * 2)
        self.line_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.max_lines * 2)
        self.num_lines = ti.field(dtype=ti.i32, shape=())

        # --- Point geometry buffers ---
        self.max_points: int = 50
        self.point_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.max_points)
        self.point_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.max_points)
        self.num_points = ti.field(dtype=ti.i32, shape=())

        # --- Triangle meshes ---
        self.platform_triangle = ti.Vector.field(3, dtype=ti.f32, shape=3)
        self.base_triangle = ti.Vector.field(3, dtype=ti.f32, shape=3)

        # --- Ground grid ---
        self.grid_vertices = ti.Vector.field(3, dtype=ti.f32, shape=200)
        self.num_grid_verts: int = 0
        self._init_ground_grid()

        # --- Control mode ---
        # 0 = carriages, 1 = pose (height / pitch / roll), 2 = corners
        self.control_mode: int = 0

        # Carriage slider values (normalised 0-100)
        self.carriage_a: float = 50.0
        self.carriage_b: float = 50.0
        self.carriage_c: float = 50.0

        # Pose slider values
        self.target_height: float = 70.0
        self.target_pitch: float = 0.0   # degrees
        self.target_roll: float = 0.0    # degrees

        # Corner slider values
        self.corner_0: float = 70.0
        self.corner_1: float = 70.0
        self.corner_2: float = 70.0

        # --- Geometry sliders ---
        self.rail_start: float = 30.0
        self.rail_length: float = 70.0
        self.platform_radius: float = 100.0  # default = rail_start + rail_length
        self.rod_length: float = 80.0

        # --- Cached state for status panel ---
        self.current_state: Optional[HexapodState] = None
        self.platform_color: Tuple[float, float, float] = (0.3, 0.6, 0.9)

    # ------------------------------------------------------------------ #
    #  Ground grid
    # ------------------------------------------------------------------ #

    def _init_ground_grid(self) -> None:
        """Build a ground grid on the XZ plane (Y=0) spanning -150..150 with step 30."""
        grid_extent: int = 150
        grid_step: int = 30
        idx: int = 0

        # Lines parallel to X axis
        for z in range(-grid_extent, grid_extent + 1, grid_step):
            if idx + 2 <= 200:
                self.grid_vertices[idx] = tm.vec3(float(-grid_extent), 0.0, float(z))
                self.grid_vertices[idx + 1] = tm.vec3(float(grid_extent), 0.0, float(z))
                idx += 2

        # Lines parallel to Z axis
        for x in range(-grid_extent, grid_extent + 1, grid_step):
            if idx + 2 <= 200:
                self.grid_vertices[idx] = tm.vec3(float(x), 0.0, float(-grid_extent))
                self.grid_vertices[idx + 1] = tm.vec3(float(x), 0.0, float(grid_extent))
                idx += 2

        self.num_grid_verts = idx

    # ------------------------------------------------------------------ #
    #  Geometry builder (Taichi kernel)
    # ------------------------------------------------------------------ #

    @ti.kernel
    def _build_geometry(
        self,
        # Carriage world positions (flattened)
        c0x: ti.f32, c0y: ti.f32, c0z: ti.f32,
        c1x: ti.f32, c1y: ti.f32, c1z: ti.f32,
        c2x: ti.f32, c2y: ti.f32, c2z: ti.f32,
        # Platform vertex world positions
        p0x: ti.f32, p0y: ti.f32, p0z: ti.f32,
        p1x: ti.f32, p1y: ti.f32, p1z: ti.f32,
        p2x: ti.f32, p2y: ti.f32, p2z: ti.f32,
        # Base triangle vertices
        b0x: ti.f32, b0y: ti.f32, b0z: ti.f32,
        b1x: ti.f32, b1y: ti.f32, b1z: ti.f32,
        b2x: ti.f32, b2y: ti.f32, b2z: ti.f32,
        # Geometry params
        rail_start: ti.f32,
        rail_length: ti.f32,
        is_valid: ti.i32,
    ) -> None:
        """Populate line / point / triangle buffers for one frame."""

        carriage = [
            tm.vec3(c0x, c0y, c0z),
            tm.vec3(c1x, c1y, c1z),
            tm.vec3(c2x, c2y, c2z),
        ]
        platform = [
            tm.vec3(p0x, p0y, p0z),
            tm.vec3(p1x, p1y, p1z),
            tm.vec3(p2x, p2y, p2z),
        ]
        base = [
            tm.vec3(b0x, b0y, b0z),
            tm.vec3(b1x, b1y, b1z),
            tm.vec3(b2x, b2y, b2z),
        ]

        line_idx: ti.i32 = 0
        point_idx: ti.i32 = 0

        rail_angles = [0.0, 120.0, 240.0]

        # ---- Rails (yellow) ----
        for i in ti.static(range(3)):
            angle = rail_angles[i] * tm.pi / 180.0
            cos_a = tm.cos(angle)
            sin_a = tm.sin(angle)
            start = tm.vec3(rail_start * cos_a, 0.0, rail_start * sin_a)
            end = tm.vec3(
                (rail_start + rail_length) * cos_a,
                0.0,
                (rail_start + rail_length) * sin_a,
            )
            self.line_vertices[line_idx * 2] = start
            self.line_vertices[line_idx * 2 + 1] = end
            self.line_colors[line_idx * 2] = tm.vec3(0.9, 0.9, 0.2)
            self.line_colors[line_idx * 2 + 1] = tm.vec3(0.9, 0.9, 0.2)
            line_idx += 1

        # ---- Rods (orange if valid, red otherwise) ----
        rod_color = tm.vec3(1.0, 0.5, 0.1) if is_valid else tm.vec3(1.0, 0.2, 0.2)
        for i in ti.static(range(3)):
            self.line_vertices[line_idx * 2] = carriage[i]
            self.line_vertices[line_idx * 2 + 1] = platform[i]
            self.line_colors[line_idx * 2] = rod_color
            self.line_colors[line_idx * 2 + 1] = rod_color
            line_idx += 1

        # ---- Base triangle edges (gray) ----
        base_color = tm.vec3(0.5, 0.5, 0.5)
        for i in ti.static(range(3)):
            j = (i + 1) % 3
            self.line_vertices[line_idx * 2] = base[i]
            self.line_vertices[line_idx * 2 + 1] = base[j]
            self.line_colors[line_idx * 2] = base_color
            self.line_colors[line_idx * 2 + 1] = base_color
            line_idx += 1

        # ---- Platform edges (blue if valid, red otherwise) ----
        plat_color = tm.vec3(0.2, 0.6, 1.0) if is_valid else tm.vec3(1.0, 0.3, 0.3)
        for i in ti.static(range(3)):
            j = (i + 1) % 3
            self.line_vertices[line_idx * 2] = platform[i]
            self.line_vertices[line_idx * 2 + 1] = platform[j]
            self.line_colors[line_idx * 2] = plat_color
            self.line_colors[line_idx * 2 + 1] = plat_color
            line_idx += 1

        # ---- RGB axes at origin (20 units long) ----
        axis_len: ti.f32 = 20.0
        # X axis - red
        self.line_vertices[line_idx * 2] = tm.vec3(0.0, 0.0, 0.0)
        self.line_vertices[line_idx * 2 + 1] = tm.vec3(axis_len, 0.0, 0.0)
        self.line_colors[line_idx * 2] = tm.vec3(1.0, 0.0, 0.0)
        self.line_colors[line_idx * 2 + 1] = tm.vec3(1.0, 0.0, 0.0)
        line_idx += 1
        # Y axis - green
        self.line_vertices[line_idx * 2] = tm.vec3(0.0, 0.0, 0.0)
        self.line_vertices[line_idx * 2 + 1] = tm.vec3(0.0, axis_len, 0.0)
        self.line_colors[line_idx * 2] = tm.vec3(0.0, 1.0, 0.0)
        self.line_colors[line_idx * 2 + 1] = tm.vec3(0.0, 1.0, 0.0)
        line_idx += 1
        # Z axis - blue
        self.line_vertices[line_idx * 2] = tm.vec3(0.0, 0.0, 0.0)
        self.line_vertices[line_idx * 2 + 1] = tm.vec3(0.0, 0.0, axis_len)
        self.line_colors[line_idx * 2] = tm.vec3(0.0, 0.0, 1.0)
        self.line_colors[line_idx * 2 + 1] = tm.vec3(0.0, 0.0, 1.0)
        line_idx += 1

        self.num_lines[None] = line_idx

        # ---- Points: carriages (red) ----
        for i in ti.static(range(3)):
            self.point_vertices[point_idx] = carriage[i]
            self.point_colors[point_idx] = tm.vec3(1.0, 0.2, 0.2)
            point_idx += 1

        # ---- Points: platform vertices (blue/red) ----
        pt_color = tm.vec3(0.2, 0.4, 1.0) if is_valid else tm.vec3(1.0, 0.3, 0.3)
        for i in ti.static(range(3)):
            self.point_vertices[point_idx] = platform[i]
            self.point_colors[point_idx] = pt_color
            point_idx += 1

        self.num_points[None] = point_idx

        # ---- Filled triangles ----
        for i in ti.static(range(3)):
            self.platform_triangle[i] = platform[i]
            self.base_triangle[i] = base[i]

    # ------------------------------------------------------------------ #
    #  State update
    # ------------------------------------------------------------------ #

    def update_from_state(self, state: HexapodState) -> None:
        """Extract render data from a *HexapodState* and rebuild GPU geometry."""
        self.current_state = state

        c: np.ndarray = state.carriage_world
        p: np.ndarray = state.vertices if state.vertices is not None else np.zeros((3, 3))

        # Base triangle at rail-end radius
        rail_end: float = state.geometry.rail_end
        b = np.zeros((3, 3))
        for i in range(3):
            angle = i * 2.0 * np.pi / 3.0
            b[i] = [rail_end * np.cos(angle), 0.0, rail_end * np.sin(angle)]

        self.platform_color = (0.3, 0.6, 0.9) if state.is_valid else (0.9, 0.3, 0.3)

        self._build_geometry(
            c[0, 0], c[0, 1], c[0, 2],
            c[1, 0], c[1, 1], c[1, 2],
            c[2, 0], c[2, 1], c[2, 2],
            p[0, 0], p[0, 1], p[0, 2],
            p[1, 0], p[1, 1], p[1, 2],
            p[2, 0], p[2, 1], p[2, 2],
            b[0, 0], b[0, 1], b[0, 2],
            b[1, 0], b[1, 1], b[1, 2],
            b[2, 0], b[2, 1], b[2, 2],
            state.geometry.rail_start,
            state.geometry.rail_length,
            1 if state.is_valid else 0,
        )

    # ------------------------------------------------------------------ #
    #  Render
    # ------------------------------------------------------------------ #

    def render(self) -> None:
        """Draw the current frame to the canvas."""
        # Let the user orbit / pan the camera
        self.camera.track_user_inputs(self.window, movement_speed=3.0, hold_key=ti.ui.RMB)
        self.scene.set_camera(self.camera)

        # Lighting
        self.scene.ambient_light((0.4, 0.4, 0.4))
        self.scene.point_light(pos=(100.0, 150.0, 100.0), color=(1.0, 1.0, 1.0))

        # Ground grid
        self.scene.lines(
            self.grid_vertices,
            width=1.0,
            color=(0.3, 0.3, 0.3),
            vertex_count=self.num_grid_verts,
        )

        # Base triangle mesh
        self.scene.mesh(self.base_triangle, color=(0.4, 0.4, 0.4))

        # Lines (rails, rods, edges, axes)
        n_line_verts: int = self.num_lines[None] * 2
        self.scene.lines(
            self.line_vertices,
            width=3.0,
            per_vertex_color=self.line_colors,
            vertex_count=n_line_verts,
        )

        # Points (carriages, platform vertices)
        n_points: int = self.num_points[None]
        self.scene.particles(
            self.point_vertices,
            radius=4.0,
            per_vertex_color=self.point_colors,
            index_count=n_points,
        )

        # Platform triangle mesh
        self.scene.mesh(self.platform_triangle, color=self.platform_color)

        self.canvas.scene(self.scene)

    # ------------------------------------------------------------------ #
    #  GUI panels
    # ------------------------------------------------------------------ #

    def show_gui(self) -> None:
        """Draw the control and status GUI panels."""
        # Build a temporary geometry for range queries
        geometry = HexapodGeometry(
            rail_start=self.rail_start,
            rail_length=self.rail_length,
            rod_length=self.rod_length,
            platform_radius=self.platform_radius,
        )
        height_min, height_max = geometry.compute_height_range()
        corner_min, corner_max = geometry.compute_corner_height_range()

        # ---- Control panel ----
        self.window.GUI.begin("Control", 0.02, 0.02, 0.28, 0.70)

        self.window.GUI.text("Control Mode")
        if self.window.GUI.button("Carriages"):
            self.control_mode = 0
        if self.window.GUI.button("Pose (H/P/R)"):
            self.control_mode = 1
        if self.window.GUI.button("Corners"):
            self.control_mode = 2

        self.window.GUI.text("")

        if self.control_mode == 0:
            self.window.GUI.text("=== Carriage Control ===")
            self.window.GUI.text("(Always valid: 0-100%)")
            self.carriage_a = self.window.GUI.slider_float("Rail A", self.carriage_a, 0.0, 100.0)
            self.carriage_b = self.window.GUI.slider_float("Rail B", self.carriage_b, 0.0, 100.0)
            self.carriage_c = self.window.GUI.slider_float("Rail C", self.carriage_c, 0.0, 100.0)

        elif self.control_mode == 1:
            self.window.GUI.text("=== Pose Control ===")
            self.window.GUI.text(f"Height range: {height_min:.1f} - {height_max:.1f}")
            self.target_height = max(height_min, min(height_max, self.target_height))
            self.target_height = self.window.GUI.slider_float(
                "Height", self.target_height, height_min, height_max,
            )

            tilt_min, tilt_max = geometry.compute_tilt_range(self.target_height)
            tilt_max_deg: float = math.degrees(tilt_max)
            self.window.GUI.text(f"Tilt range: +/-{tilt_max_deg:.1f} deg")
            self.target_pitch = max(-tilt_max_deg, min(tilt_max_deg, self.target_pitch))
            self.target_roll = max(-tilt_max_deg, min(tilt_max_deg, self.target_roll))
            self.target_pitch = self.window.GUI.slider_float(
                "Pitch (deg)", self.target_pitch, -tilt_max_deg, tilt_max_deg,
            )
            self.target_roll = self.window.GUI.slider_float(
                "Roll (deg)", self.target_roll, -tilt_max_deg, tilt_max_deg,
            )

        elif self.control_mode == 2:
            self.window.GUI.text("=== Corner Control ===")
            self.window.GUI.text(f"Height range: {corner_min:.1f} - {corner_max:.1f}")
            self.window.GUI.text("(Corners are coupled by rigid body)")
            self.corner_0 = max(corner_min, min(corner_max, self.corner_0))
            self.corner_1 = max(corner_min, min(corner_max, self.corner_1))
            self.corner_2 = max(corner_min, min(corner_max, self.corner_2))
            self.corner_0 = self.window.GUI.slider_float("Corner 0", self.corner_0, corner_min, corner_max)
            self.corner_1 = self.window.GUI.slider_float("Corner 1", self.corner_1, corner_min, corner_max)
            self.corner_2 = self.window.GUI.slider_float("Corner 2", self.corner_2, corner_min, corner_max)

        # ---- Geometry controls ----
        self.window.GUI.text("")
        self.window.GUI.text("=== Geometry ===")

        old_base: float = self.rail_start + self.rail_length
        self.rail_start = self.window.GUI.slider_float("Rail Start", self.rail_start, 10.0, 60.0)
        self.rail_length = self.window.GUI.slider_float("Rail Length", self.rail_length, 30.0, 100.0)
        base_size: float = self.rail_start + self.rail_length

        # Keep platform radius locked to base if it was there before
        if abs(self.platform_radius - old_base) < 1.0:
            self.platform_radius = base_size
        self.platform_radius = min(self.platform_radius, base_size)
        self.platform_radius = self.window.GUI.slider_float(
            "Platform R", self.platform_radius, 20.0, base_size,
        )
        self.window.GUI.text(f"(Base size: {base_size:.0f})")

        self.rod_length = self.window.GUI.slider_float("Rod Length", self.rod_length, 40.0, 120.0)

        self.window.GUI.end()

        # ---- Status panel ----
        self.window.GUI.begin("Status", 0.72, 0.02, 0.26, 0.50)

        if self.current_state is not None:
            state = self.current_state
            status_name: str = state.constraint_status.value.upper()

            if state.constraint_status == SolveStatus.DEFINED:
                self.window.GUI.text(f"Status: {status_name} [OK]")
            elif state.constraint_status == SolveStatus.UNDERDEFINED:
                self.window.GUI.text(f"Status: {status_name} [!]")
            elif state.constraint_status == SolveStatus.OVERDEFINED:
                self.window.GUI.text(f"Status: {status_name} [~]")
            else:
                self.window.GUI.text(f"Status: {status_name} [X]")

            self.window.GUI.text(f"Valid: {'Yes' if state.is_valid else 'NO'}")

            if not state.is_valid:
                self.window.GUI.text("")
                msg = state.solve_message[:40]
                self.window.GUI.text(f"Error: {msg}")

            self.window.GUI.text("")
            self.window.GUI.text("=== Platform State ===")

            if state.pose is not None:
                self.window.GUI.text(f"Height: {state.pose.height:.1f}")
                pitch_deg: float = math.degrees(state.pose.pitch)
                roll_deg: float = math.degrees(state.pose.roll)
                self.window.GUI.text(f"Pitch: {pitch_deg:.1f} deg")
                self.window.GUI.text(f"Roll: {roll_deg:.1f} deg")

            if state.corner_heights is not None:
                self.window.GUI.text("")
                self.window.GUI.text("Corner Heights:")
                self.window.GUI.text(f"  C0: {state.corner_heights[0]:.1f}")
                self.window.GUI.text(f"  C1: {state.corner_heights[1]:.1f}")
                self.window.GUI.text(f"  C2: {state.corner_heights[2]:.1f}")

            self.window.GUI.text("")
            self.window.GUI.text("=== Carriages ===")
            self.window.GUI.text(f"  A: {state.carriage_positions[0]:.1f}%")
            self.window.GUI.text(f"  B: {state.carriage_positions[1]:.1f}%")
            self.window.GUI.text(f"  C: {state.carriage_positions[2]:.1f}%")

            self.window.GUI.text("")
            self.window.GUI.text(f"Rod Error: {state.rod_error:.4f}")

        self.window.GUI.end()

    # ------------------------------------------------------------------ #
    #  Apply GUI values back to the hexapod
    # ------------------------------------------------------------------ #

    def apply_controls_to_hexapod(self, hexapod: HexapodUnit) -> None:
        """Push the current slider values into *hexapod*, updating geometry if needed."""
        # Recreate geometry when sliders change
        if (
            hexapod.geometry.rail_start != self.rail_start
            or hexapod.geometry.rail_length != self.rail_length
            or hexapod.geometry.platform_radius != self.platform_radius
            or hexapod.geometry.rod_length != self.rod_length
        ):
            new_geom = HexapodGeometry(
                rail_start=self.rail_start,
                rail_length=self.rail_length,
                rod_length=self.rod_length,
                platform_radius=self.platform_radius,
            )
            hexapod.geometry = new_geom
            hexapod.solver.geometry = new_geom

        # Apply control-mode inputs
        if self.control_mode == 0:
            hexapod.set_carriages(self.carriage_a, self.carriage_b, self.carriage_c)
        elif self.control_mode == 1:
            hexapod.set_pose(
                height=self.target_height,
                pitch=math.radians(self.target_pitch),
                roll=math.radians(self.target_roll),
            )
        elif self.control_mode == 2:
            hexapod.set_corners(self.corner_0, self.corner_1, self.corner_2)

    # ------------------------------------------------------------------ #
    #  Window lifecycle helpers
    # ------------------------------------------------------------------ #

    def should_close(self) -> bool:
        """Return *True* when the user has closed the window."""
        return not self.window.running

    def finish_frame(self) -> None:
        """Swap buffers and present the frame."""
        self.window.show()
