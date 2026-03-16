"""Grid visualization for multiple hexapod platforms.

Renders a triangular grid of hexapod parallel platforms with:
- Custom orbit camera (WASD/QE keys, mouse drag, scroll zoom)
- Per-platform color coding by constraint status
- Corner-to-corner connection lines and mismatch highlighting
- Ghost triangles showing available empty slots for expansion
- Global pose controls with optional rigid propagation
- Platform selection and detail inspection

Y-up coordinate system (Taichi default).
"""

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import taichi as ti
import taichi.math as tm

from hexapod_grid_sim.grid.hexapod_unit import HexapodState, HexapodUnit
from hexapod_grid_sim.physics.solver import HexapodGeometry
from hexapod_grid_sim.constraints.constraint_set import SolveStatus

# Optional win32 window management (Windows only)
try:
    import win32gui  # type: ignore[import-untyped]
    HAS_WIN32: bool = True
except ImportError:
    HAS_WIN32 = False

# Optional pynput for scroll-wheel support
try:
    from pynput import mouse as pynput_mouse  # type: ignore[import-untyped]
    HAS_PYNPUT: bool = True
except ImportError:
    HAS_PYNPUT = False


# ====================================================================== #
#  Helper: lazily import grid types so the module can be loaded even if
#  grid/ hasn't been written yet.
# ====================================================================== #

def _grid_imports():
    """Return (HexapodGrid, GridPlatform, GridPosition, Orientation) tuple."""
    # Deferred to avoid circular imports; callers destructure the result.
    from hexapod_grid_sim.grid.hexapod_unit import HexapodUnit  # noqa: F811
    # The grid module is expected to provide these types:
    from hexapod_grid_sim.grid.grid import (  # type: ignore[import-untyped]
        HexapodGrid,
        GridPlatform,
        GridPosition,
        Orientation,
    )
    return HexapodGrid, GridPlatform, GridPosition, Orientation


# ====================================================================== #
#  GridViewer
# ====================================================================== #

@ti.data_oriented
class GridViewer:
    """3D viewer for a grid of hexapod platforms.

    Camera is controlled via an orbit system (spherical coordinates around a
    look-at target) with keyboard / mouse input.
    """

    # ------------------------------------------------------------------ #
    #  Construction
    # ------------------------------------------------------------------ #

    def __init__(self, width: int = 1400, height: int = 900) -> None:
        self.width: int = width
        self.height: int = height

        # --- Taichi GGUI objects ---
        self.window = ti.ui.Window("Hexapod Grid Viewer", (width, height), vsync=True)
        self.canvas = self.window.get_canvas()
        self.scene = self.window.get_scene()
        self.camera = ti.ui.Camera()

        # Initial camera
        self.camera.position(400.0, 300.0, 400.0)
        self.camera.lookat(200.0, 50.0, 150.0)
        self.camera.projection_mode(ti.ui.ProjectionMode.Perspective)
        self.camera.fov(45)
        self.camera.z_near(1.0)
        self.camera.z_far(10000.0)

        # --- Camera orbit state ---
        self.cam_target: np.ndarray = np.array([200.0, 0.0, 150.0])
        self.cam_distance: float = 400.0
        self.cam_pitch: float = 0.6      # radians (elevation)
        self.cam_yaw: float = 0.8        # radians (azimuth)
        self.cam_min_dist: float = 50.0
        self.cam_max_dist: float = 3000.0

        # --- Line buffers ---
        self.max_lines: int = 5000
        self.line_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.max_lines * 2)
        self.line_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.max_lines * 2)
        self.num_lines = ti.field(dtype=ti.i32, shape=())

        # --- Point buffers ---
        self.max_points: int = 2000
        self.point_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.max_points)
        self.point_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.max_points)
        self.num_points = ti.field(dtype=ti.i32, shape=())

        # --- Triangle buffers ---
        self.max_triangles: int = 500
        self.triangle_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.max_triangles * 3)
        self.triangle_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.max_triangles * 3)
        self.num_triangles = ti.field(dtype=ti.i32, shape=())

        # --- Ghost triangle buffers (empty slots) ---
        self.max_ghost_tris: int = 50
        self.ghost_tri_vertices = ti.Vector.field(3, dtype=ti.f32, shape=self.max_ghost_tris * 3)
        self.ghost_tri_colors = ti.Vector.field(3, dtype=ti.f32, shape=self.max_ghost_tris * 3)
        self.num_ghost_tris = ti.field(dtype=ti.i32, shape=())
        self.ghost_positions: List[Tuple] = []
        self.show_ghosts: bool = True

        # --- Ground grid ---
        self.grid_vertices = ti.Vector.field(3, dtype=ti.f32, shape=400)
        self.num_grid_verts: int = 0
        self._init_ground_grid()

        # --- Origin axes ---
        self._origin_axes = ti.Vector.field(3, dtype=ti.f32, shape=6)
        axis_len: float = 30.0
        self._origin_axes[0] = tm.vec3(0.0, 0.1, 0.0)
        self._origin_axes[1] = tm.vec3(axis_len, 0.1, 0.0)
        self._origin_axes[2] = tm.vec3(0.0, 0.1, 0.0)
        self._origin_axes[3] = tm.vec3(0.0, axis_len, 0.0)
        self._origin_axes[4] = tm.vec3(0.0, 0.1, 0.0)
        self._origin_axes[5] = tm.vec3(0.0, 0.1, axis_len)

        # --- Platform selection ---
        self.selected_platform: Optional[Tuple[int, int]] = None  # (row, col) or None

        # --- Global controls ---
        self.global_height: float = 70.0
        self.global_pitch: float = 0.0   # degrees
        self.global_roll: float = 0.0    # degrees

        # --- Propagation ---
        self.propagation_mode: int = 0   # 0=disabled, 1=rigid
        self.anchor_index: int = 0

        # --- Display toggles ---
        self.show_connections: bool = True
        self.show_mismatches: bool = True

        # --- Cached solve results ---
        self._cached_states: Dict = {}
        self._cached_mismatches: List = []

        # --- Scroll wheel (pynput) ---
        self._scroll_delta: float = 0.0
        self._scroll_listener = None
        self._scroll_listener_started: bool = False

        # --- Mouse drag state ---
        self._mouse_prev: Optional[Tuple[float, float]] = None
        self._dragging: bool = False

        # Initialise all GPU buffers to zero to avoid garbage
        self._init_color_buffers()

    # ------------------------------------------------------------------ #
    #  Buffer initialisation
    # ------------------------------------------------------------------ #

    def _init_color_buffers(self) -> None:
        """Zero-fill all GPU geometry buffers."""
        self.line_vertices.from_numpy(np.zeros((self.max_lines * 2, 3), dtype=np.float32))
        self.line_colors.from_numpy(np.zeros((self.max_lines * 2, 3), dtype=np.float32))
        self.point_vertices.from_numpy(np.zeros((self.max_points, 3), dtype=np.float32))
        self.point_colors.from_numpy(np.zeros((self.max_points, 3), dtype=np.float32))
        self.triangle_vertices.from_numpy(np.zeros((self.max_triangles * 3, 3), dtype=np.float32))
        self.triangle_colors.from_numpy(np.zeros((self.max_triangles * 3, 3), dtype=np.float32))
        self.ghost_tri_vertices.from_numpy(np.zeros((self.max_ghost_tris * 3, 3), dtype=np.float32))
        self.ghost_tri_colors.from_numpy(np.zeros((self.max_ghost_tris * 3, 3), dtype=np.float32))

        self.num_lines[None] = 0
        self.num_points[None] = 0
        self.num_triangles[None] = 0
        self.num_ghost_tris[None] = 0

    # ------------------------------------------------------------------ #
    #  Ground grid
    # ------------------------------------------------------------------ #

    def _init_ground_grid(self) -> None:
        """Build a large ground grid on the XZ plane."""
        grid_size: int = 2000
        grid_step: int = 100
        grid_y: float = -0.5  # slightly below Y=0 to avoid z-fighting
        idx: int = 0

        for z in range(-500, grid_size + 1, grid_step):
            if idx + 2 <= 400:
                self.grid_vertices[idx] = tm.vec3(-500.0, grid_y, float(z))
                self.grid_vertices[idx + 1] = tm.vec3(float(grid_size), grid_y, float(z))
                idx += 2

        for x in range(-500, grid_size + 1, grid_step):
            if idx + 2 <= 400:
                self.grid_vertices[idx] = tm.vec3(float(x), grid_y, -500.0)
                self.grid_vertices[idx + 1] = tm.vec3(float(x), grid_y, float(grid_size))
                idx += 2

        self.num_grid_verts = idx

    # ------------------------------------------------------------------ #
    #  Camera orbit system
    # ------------------------------------------------------------------ #

    def _update_camera(self) -> None:
        """Compute camera position from spherical orbit coordinates and push
        the result to the Taichi camera object."""
        cos_p = math.cos(self.cam_pitch)
        sin_p = math.sin(self.cam_pitch)
        cos_y = math.cos(self.cam_yaw)
        sin_y = math.sin(self.cam_yaw)

        offset = np.array([
            self.cam_distance * cos_p * sin_y,
            self.cam_distance * sin_p,
            self.cam_distance * cos_p * cos_y,
        ])
        cam_pos = self.cam_target + offset

        self.camera.position(float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2]))
        self.camera.lookat(
            float(self.cam_target[0]),
            float(self.cam_target[1]),
            float(self.cam_target[2]),
        )

    def handle_input(self) -> None:
        """Process keyboard, mouse-drag, and scroll-wheel input for the orbit
        camera.

        Keys:
            W/S  - orbit pitch (up/down)
            A/D  - orbit yaw (left/right)
            Q/E  - zoom in/out
            Arrow keys - pan target on XZ plane
        Mouse:
            LMB drag - orbit
            Scroll   - zoom
        """
        dt: float = 0.016  # approximate frame time

        # --- Keyboard orbit ---
        orbit_speed: float = 1.5  # rad/s
        pan_speed: float = 200.0  # units/s
        zoom_speed: float = 200.0  # units/s

        if self.window.is_pressed("w"):
            self.cam_pitch = min(self.cam_pitch + orbit_speed * dt, math.pi * 0.49)
        if self.window.is_pressed("s"):
            self.cam_pitch = max(self.cam_pitch - orbit_speed * dt, 0.05)
        if self.window.is_pressed("a"):
            self.cam_yaw += orbit_speed * dt
        if self.window.is_pressed("d"):
            self.cam_yaw -= orbit_speed * dt
        if self.window.is_pressed("q"):
            self.cam_distance = max(self.cam_min_dist, self.cam_distance - zoom_speed * dt)
        if self.window.is_pressed("e"):
            self.cam_distance = min(self.cam_max_dist, self.cam_distance + zoom_speed * dt)

        # Arrow key panning (XZ plane)
        forward = np.array([-math.sin(self.cam_yaw), 0.0, -math.cos(self.cam_yaw)])
        right = np.array([math.cos(self.cam_yaw), 0.0, -math.sin(self.cam_yaw)])
        if self.window.is_pressed(ti.ui.UP):
            self.cam_target += forward * pan_speed * dt
        if self.window.is_pressed(ti.ui.DOWN):
            self.cam_target -= forward * pan_speed * dt
        if self.window.is_pressed(ti.ui.LEFT):
            self.cam_target -= right * pan_speed * dt
        if self.window.is_pressed(ti.ui.RIGHT):
            self.cam_target += right * pan_speed * dt

        # --- Mouse drag orbit ---
        cur = self.window.get_cursor_pos()
        if self.window.is_pressed(ti.ui.LMB):
            if self._mouse_prev is not None:
                dx = cur[0] - self._mouse_prev[0]
                dy = cur[1] - self._mouse_prev[1]
                self.cam_yaw -= dx * 3.0
                self.cam_pitch = min(
                    max(self.cam_pitch + dy * 3.0, 0.05),
                    math.pi * 0.49,
                )
            self._mouse_prev = cur
            self._dragging = True
        else:
            self._mouse_prev = None
            self._dragging = False

        # --- Scroll zoom (pynput) ---
        if HAS_PYNPUT:
            self._ensure_scroll_listener()
            delta = self._get_scroll_delta()
            if abs(delta) > 0.01:
                self.cam_distance *= (1.0 - delta * 0.1)
                self.cam_distance = max(self.cam_min_dist, min(self.cam_max_dist, self.cam_distance))

        self._update_camera()

    # ------------------------------------------------------------------ #
    #  Scroll listener helpers (pynput)
    # ------------------------------------------------------------------ #

    def _ensure_scroll_listener(self) -> None:
        """Start the pynput scroll listener (if not already running)."""
        if self._scroll_listener_started or not HAS_PYNPUT:
            return
        try:
            viewer = self

            def on_scroll(_x: int, _y: int, _dx: int, dy: int) -> None:
                viewer._scroll_delta += dy

            self._scroll_listener = pynput_mouse.Listener(on_scroll=on_scroll)
            self._scroll_listener.start()
            self._scroll_listener_started = True
        except Exception:
            pass

    def _get_scroll_delta(self) -> float:
        """Read and reset accumulated scroll delta."""
        delta = self._scroll_delta
        self._scroll_delta = 0.0
        return delta

    def _stop_scroll_listener(self) -> None:
        """Cleanly shut down the pynput listener."""
        if self._scroll_listener is not None:
            try:
                self._scroll_listener.stop()
            except Exception:
                pass
            self._scroll_listener = None
            self._scroll_listener_started = False
            self._scroll_delta = 0.0

    # ------------------------------------------------------------------ #
    #  Build geometry from grid
    # ------------------------------------------------------------------ #

    def update_from_grid(self, grid) -> None:  # grid: HexapodGrid
        """Rebuild all GPU geometry from the current state of *grid*.

        Iterates every platform, applies its world offset and rotation, and
        emits rails, rods, base edges, platform edges, connection lines, and
        ghost triangles.

        Parameters
        ----------
        grid:
            A ``HexapodGrid`` instance (imported lazily to avoid circular deps).
        """
        line_verts: List[np.ndarray] = []
        line_cols: List[np.ndarray] = []
        point_verts: List[np.ndarray] = []
        point_cols: List[np.ndarray] = []
        tri_verts: List[np.ndarray] = []
        tri_cols: List[np.ndarray] = []

        # Solve all platforms and cache
        self._cached_states = {}
        for pos, platform in grid.platforms.items():
            self._cached_states[pos] = platform.hexapod.solve()

        # Colours for constraint status
        COLOR_DEFINED = np.array([0.2, 0.8, 0.3])     # green
        COLOR_OVERDEFINED = np.array([0.9, 0.9, 0.2])  # yellow
        COLOR_CONFLICTING = np.array([1.0, 0.2, 0.2])  # red
        COLOR_UNDERDEFINED = np.array([0.5, 0.5, 0.5]) # gray

        COLOR_RAIL = np.array([0.9, 0.9, 0.2])         # yellow
        COLOR_ROD_OK = np.array([1.0, 0.5, 0.1])       # orange
        COLOR_ROD_BAD = np.array([1.0, 0.2, 0.2])      # red
        COLOR_BASE = np.array([0.5, 0.5, 0.5])         # gray
        COLOR_CARRIAGE = np.array([1.0, 0.2, 0.2])     # red
        COLOR_CONNECTION = np.array([0.0, 0.8, 0.8])   # cyan

        # ---------- per-platform geometry ----------
        for pos, platform in grid.platforms.items():
            state: Optional[HexapodState] = self._cached_states.get(pos)
            if state is None:
                continue

            wo: np.ndarray = platform.world_offset
            rot: float = platform.rotation
            cos_r, sin_r = math.cos(rot), math.sin(rot)

            def _to_world(local: np.ndarray) -> np.ndarray:
                """Rotate by platform rotation then translate."""
                rx = local[0] * cos_r - local[2] * sin_r + wo[0]
                ry = local[1] + wo[1]
                rz = local[0] * sin_r + local[2] * cos_r + wo[2]
                return np.array([rx, ry, rz])

            # Determine platform colour from status
            if state.constraint_status == SolveStatus.DEFINED:
                status_color = COLOR_DEFINED
            elif state.constraint_status == SolveStatus.OVERDEFINED:
                status_color = COLOR_OVERDEFINED
            elif state.constraint_status == SolveStatus.CONFLICTING:
                status_color = COLOR_CONFLICTING
            else:
                status_color = COLOR_UNDERDEFINED

            g: HexapodGeometry = state.geometry

            # --- Rails (yellow) ---
            for i in range(3):
                angle = i * 2.0 * math.pi / 3.0 + rot
                cos_a = math.cos(angle)
                sin_a = math.sin(angle)
                rs = np.array([g.rail_start * cos_a + wo[0], 0.0 + wo[1], g.rail_start * sin_a + wo[2]])
                re = np.array([(g.rail_start + g.rail_length) * cos_a + wo[0], 0.0 + wo[1],
                               (g.rail_start + g.rail_length) * sin_a + wo[2]])
                line_verts.extend([rs, re])
                line_cols.extend([COLOR_RAIL, COLOR_RAIL])

            # --- Carriages, rods, platform ---
            carriage_world = state.carriage_world  # (3,3)
            vertices = state.vertices if state.vertices is not None else np.zeros((3, 3))

            c_w = np.array([_to_world(carriage_world[i]) for i in range(3)])
            p_w = np.array([_to_world(vertices[i]) for i in range(3)])

            rod_color = COLOR_ROD_OK if state.is_valid else COLOR_ROD_BAD
            for i in range(3):
                # Rods
                line_verts.extend([c_w[i], p_w[i]])
                line_cols.extend([rod_color, rod_color])
                # Carriage points
                point_verts.append(c_w[i])
                point_cols.append(COLOR_CARRIAGE)

            # Base triangle edges
            rail_end = g.rail_end
            base_w = np.zeros((3, 3))
            for i in range(3):
                angle = i * 2.0 * math.pi / 3.0
                local_b = np.array([rail_end * math.cos(angle), 0.0, rail_end * math.sin(angle)])
                base_w[i] = _to_world(local_b)

            for i in range(3):
                j = (i + 1) % 3
                line_verts.extend([base_w[i], base_w[j]])
                line_cols.extend([COLOR_BASE, COLOR_BASE])

            # Platform edges (coloured by status)
            plat_edge_color = status_color if state.is_valid else COLOR_ROD_BAD
            for i in range(3):
                j = (i + 1) % 3
                line_verts.extend([p_w[i], p_w[j]])
                line_cols.extend([plat_edge_color, plat_edge_color])

            # Platform vertex points
            pt_color = status_color if state.is_valid else COLOR_ROD_BAD
            for i in range(3):
                point_verts.append(p_w[i])
                point_cols.append(pt_color)

            # Platform filled triangle
            for i in range(3):
                tri_verts.append(p_w[i])
                tri_cols.append(status_color * 0.6)  # slightly darker fill

            # Base filled triangle
            for i in range(3):
                tri_verts.append(base_w[i])
                tri_cols.append(COLOR_BASE * 0.5)

            # Highlight selected platform
            is_selected = (
                self.selected_platform is not None
                and self.selected_platform[0] == pos.row
                and self.selected_platform[1] == pos.col
            )
            if is_selected:
                sel_color = np.array([1.0, 1.0, 0.0])  # bright yellow
                for i in range(3):
                    j = (i + 1) % 3
                    # Slightly offset above platform
                    a = p_w[i].copy(); a[1] += 2.0
                    b = p_w[j].copy(); b[1] += 2.0
                    line_verts.extend([a, b])
                    line_cols.extend([sel_color, sel_color])

        # ---------- Connection lines ----------
        if self.show_connections:
            for conn in grid.connections:
                pa = grid.platforms.get(conn.pos_a)
                pb = grid.platforms.get(conn.pos_b)
                if pa is None or pb is None:
                    continue

                wp_a = pa.get_corner_world_position(conn.corner_a)
                wp_b = pb.get_corner_world_position(conn.corner_b)
                if wp_a is not None and wp_b is not None:
                    # Height mismatch indicator
                    h_diff = abs(wp_a[1] - wp_b[1])
                    if h_diff > 0.1 and self.show_mismatches:
                        conn_col = np.array([1.0, 0.3, 0.3])  # red for mismatch
                    else:
                        conn_col = COLOR_CONNECTION

                    # Raise lines slightly
                    a = wp_a.copy(); a[1] += 1.0
                    b = wp_b.copy(); b[1] += 1.0
                    line_verts.extend([a, b])
                    line_cols.extend([conn_col, conn_col])

        # ---------- Ghost triangles for empty slots ----------
        if self.show_ghosts and hasattr(grid, "get_ghost_triangles"):
            ghosts = grid.get_ghost_triangles()
            self.ghost_positions = []
            ghost_tri_v: List[np.ndarray] = []
            ghost_tri_c: List[np.ndarray] = []
            ghost_color = np.array([0.0, 0.15, 0.2])  # dark teal

            for gpos, world_pos, rotation, orientation in ghosts:
                self.ghost_positions.append((gpos, world_pos, rotation))
                R = grid.geometry.platform_radius
                cos_gr, sin_gr = math.cos(rotation), math.sin(rotation)
                for ci in range(3):
                    ca = math.radians(ci * 120)
                    lx = R * math.cos(ca)
                    lz = R * math.sin(ca)
                    wx = lx * cos_gr - lz * sin_gr + world_pos[0]
                    wz = lx * sin_gr + lz * cos_gr + world_pos[2]
                    ghost_tri_v.append(np.array([wx, 1.0, wz]))
                    ghost_tri_c.append(ghost_color)

            if ghost_tri_v:
                gtv = np.array(ghost_tri_v, dtype=np.float32)
                gtc = np.array(ghost_tri_c, dtype=np.float32)
                n = min(len(gtv), self.max_ghost_tris * 3)
                self.ghost_tri_vertices.from_numpy(gtv[:n])
                self.ghost_tri_colors.from_numpy(gtc[:n])
                self.num_ghost_tris[None] = n // 3
            else:
                self.num_ghost_tris[None] = 0
        else:
            self.num_ghost_tris[None] = 0

        # ---------- Origin axes ----------
        axis_len = 30.0
        for axis_idx, (color, end) in enumerate([
            (np.array([1.0, 0.0, 0.0]), np.array([axis_len, 0.1, 0.0])),
            (np.array([0.0, 1.0, 0.0]), np.array([0.0, axis_len, 0.0])),
            (np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.1, axis_len])),
        ]):
            line_verts.extend([np.array([0.0, 0.1, 0.0]), end])
            line_cols.extend([color, color])

        # ---------- Upload to GPU ----------
        self._upload_lines(line_verts, line_cols)
        self._upload_points(point_verts, point_cols)
        self._upload_triangles(tri_verts, tri_cols)

    # ------------------------------------------------------------------ #
    #  GPU upload helpers
    # ------------------------------------------------------------------ #

    def _upload_lines(self, verts: List[np.ndarray], cols: List[np.ndarray]) -> None:
        if verts:
            lv = np.array(verts, dtype=np.float32)
            lc = np.array(cols, dtype=np.float32)
            n = min(len(lv), self.max_lines * 2)
            self.line_vertices.from_numpy(lv[:n])
            self.line_colors.from_numpy(lc[:n])
            self.num_lines[None] = n // 2
        else:
            self.num_lines[None] = 0

    def _upload_points(self, verts: List[np.ndarray], cols: List[np.ndarray]) -> None:
        if verts:
            pv = np.array(verts, dtype=np.float32)
            pc = np.array(cols, dtype=np.float32)
            n = min(len(pv), self.max_points)
            self.point_vertices.from_numpy(pv[:n])
            self.point_colors.from_numpy(pc[:n])
            self.num_points[None] = n
        else:
            self.num_points[None] = 0

    def _upload_triangles(self, verts: List[np.ndarray], cols: List[np.ndarray]) -> None:
        if verts:
            tv = np.array(verts, dtype=np.float32)
            tc = np.array(cols, dtype=np.float32)
            n = min(len(tv), self.max_triangles * 3)
            self.triangle_vertices.from_numpy(tv[:n])
            self.triangle_colors.from_numpy(tc[:n])
            self.num_triangles[None] = n // 3
        else:
            self.num_triangles[None] = 0

    # ------------------------------------------------------------------ #
    #  Render
    # ------------------------------------------------------------------ #

    def render(self) -> None:
        """Draw the current frame."""
        self.scene.set_camera(self.camera)

        # Lighting
        self.scene.ambient_light((0.4, 0.4, 0.4))
        self.scene.point_light(pos=(300.0, 400.0, 300.0), color=(1.0, 1.0, 1.0))

        # Ground grid
        self.scene.lines(
            self.grid_vertices,
            width=1.0,
            color=(0.25, 0.25, 0.25),
            vertex_count=self.num_grid_verts,
        )

        # Origin axes
        self.scene.lines(self._origin_axes, width=2.0, color=(1.0, 1.0, 1.0), vertex_count=6)

        # Filled triangles (base + platform meshes)
        n_tri_verts: int = self.num_triangles[None] * 3
        if n_tri_verts > 0:
            self.scene.mesh(
                self.triangle_vertices,
                per_vertex_color=self.triangle_colors,
                vertex_count=n_tri_verts,
            )

        # Ghost triangles
        n_ghost_verts: int = self.num_ghost_tris[None] * 3
        if n_ghost_verts > 0:
            self.scene.mesh(
                self.ghost_tri_vertices,
                per_vertex_color=self.ghost_tri_colors,
                vertex_count=n_ghost_verts,
            )

        # Lines (rails, rods, edges, connections, axes)
        n_line_verts: int = self.num_lines[None] * 2
        if n_line_verts > 0:
            self.scene.lines(
                self.line_vertices,
                width=2.0,
                per_vertex_color=self.line_colors,
                vertex_count=n_line_verts,
            )

        # Points (carriages, platform vertices)
        n_points: int = self.num_points[None]
        if n_points > 0:
            self.scene.particles(
                self.point_vertices,
                radius=3.0,
                per_vertex_color=self.point_colors,
                index_count=n_points,
            )

        self.canvas.scene(self.scene)

    # ------------------------------------------------------------------ #
    #  GUI
    # ------------------------------------------------------------------ #

    def show_gui(self, grid) -> None:  # grid: HexapodGrid
        """Draw the master control panel and selected-platform details.

        Parameters
        ----------
        grid:
            A ``HexapodGrid`` instance.
        """

        # ---- Master control panel ----
        self.window.GUI.begin("Grid Control", 0.02, 0.02, 0.24, 0.70)

        # --- Global pose ---
        self.window.GUI.text("=== Global Pose ===")

        geometry: HexapodGeometry = grid.geometry
        h_min, h_max = geometry.compute_height_range()
        self.global_height = max(h_min, min(h_max, self.global_height))
        self.global_height = self.window.GUI.slider_float("Height", self.global_height, h_min, h_max)

        tilt_min, tilt_max = geometry.compute_tilt_range(self.global_height)
        tilt_max_deg: float = math.degrees(tilt_max)
        self.global_pitch = max(-tilt_max_deg, min(tilt_max_deg, self.global_pitch))
        self.global_roll = max(-tilt_max_deg, min(tilt_max_deg, self.global_roll))
        self.global_pitch = self.window.GUI.slider_float("Pitch (deg)", self.global_pitch, -tilt_max_deg, tilt_max_deg)
        self.global_roll = self.window.GUI.slider_float("Roll (deg)", self.global_roll, -tilt_max_deg, tilt_max_deg)

        if self.window.GUI.button("Apply to All"):
            self._apply_global_pose(grid)

        # --- Propagation ---
        self.window.GUI.text("")
        self.window.GUI.text("=== Propagation ===")
        if self.window.GUI.button("Disabled" if self.propagation_mode == 0 else "Rigid"):
            self.propagation_mode = 1 - self.propagation_mode

        mode_label = "Disabled" if self.propagation_mode == 0 else "Rigid"
        self.window.GUI.text(f"Mode: {mode_label}")

        # Anchor selection
        platform_keys = list(grid.platforms.keys())
        if platform_keys:
            n_platforms = len(platform_keys)
            self.anchor_index = min(self.anchor_index, n_platforms - 1)
            self.anchor_index = int(
                self.window.GUI.slider_float("Anchor Idx", float(self.anchor_index), 0.0, float(n_platforms - 1))
            )
            anchor_pos = platform_keys[self.anchor_index]
            self.window.GUI.text(f"Anchor: ({anchor_pos.row}, {anchor_pos.col})")

        # --- Info ---
        self.window.GUI.text("")
        self.window.GUI.text("=== Grid Info ===")
        self.window.GUI.text(f"Platforms: {len(grid.platforms)}")
        self.window.GUI.text(f"Connections: {len(grid.connections)}")

        # Display toggles
        self.window.GUI.text("")
        self.window.GUI.text("=== Display ===")
        if self.window.GUI.button("Connections: " + ("ON" if self.show_connections else "OFF")):
            self.show_connections = not self.show_connections
        if self.window.GUI.button("Ghosts: " + ("ON" if self.show_ghosts else "OFF")):
            self.show_ghosts = not self.show_ghosts

        self.window.GUI.end()

        # ---- Selected platform panel ----
        self.window.GUI.begin("Selected Platform", 0.74, 0.02, 0.24, 0.50)

        if self.selected_platform is not None:
            row, col = self.selected_platform
            self.window.GUI.text(f"Platform ({row}, {col})")

            # Find state
            from hexapod_grid_sim.grid.grid import GridPosition  # type: ignore[import-untyped]
            sel_pos = GridPosition(row, col)
            state = self._cached_states.get(sel_pos)

            if state is not None:
                status_name = state.constraint_status.value.upper()
                self.window.GUI.text(f"Status: {status_name}")
                self.window.GUI.text(f"Valid: {'Yes' if state.is_valid else 'NO'}")

                if state.pose is not None:
                    self.window.GUI.text("")
                    self.window.GUI.text(f"Height: {state.pose.height:.1f}")
                    self.window.GUI.text(f"Pitch: {math.degrees(state.pose.pitch):.1f} deg")
                    self.window.GUI.text(f"Roll: {math.degrees(state.pose.roll):.1f} deg")

                if state.corner_heights is not None:
                    self.window.GUI.text("")
                    self.window.GUI.text("Corner Heights:")
                    for ci in range(3):
                        self.window.GUI.text(f"  C{ci}: {state.corner_heights[ci]:.1f}")

                self.window.GUI.text("")
                self.window.GUI.text("Carriages:")
                for ci in range(3):
                    self.window.GUI.text(f"  Rail {ci}: {state.carriage_positions[ci]:.1f}%")

                self.window.GUI.text("")
                self.window.GUI.text(f"Rod Error: {state.rod_error:.4f}")

                # Neighbor info
                platform = grid.platforms.get(sel_pos)
                if platform is not None and platform.neighbors:
                    self.window.GUI.text("")
                    self.window.GUI.text(f"Neighbors: {len(platform.neighbors)}")
                    for corner_idx, (npos, ncorner) in platform.neighbors.items():
                        self.window.GUI.text(f"  C{corner_idx} -> ({npos.row},{npos.col}).C{ncorner}")
            else:
                self.window.GUI.text("(no solve data)")
        else:
            self.window.GUI.text("Click a platform to select")

        self.window.GUI.end()

    # ------------------------------------------------------------------ #
    #  Global pose application
    # ------------------------------------------------------------------ #

    def _apply_global_pose(self, grid) -> None:
        """Set every platform in *grid* to the current global height/pitch/roll."""
        pitch_rad = math.radians(self.global_pitch)
        roll_rad = math.radians(self.global_roll)
        for _pos, platform in grid.platforms.items():
            platform.hexapod.set_pose(
                height=self.global_height,
                pitch=pitch_rad,
                roll=roll_rad,
            )

    # ------------------------------------------------------------------ #
    #  Platform selection via mouse click
    # ------------------------------------------------------------------ #

    def handle_click(self, grid) -> None:
        """Check for mouse clicks and select or add platforms accordingly.

        Left-click on an existing platform to select it.  Left-click on a
        ghost triangle to add a new platform at that slot.
        """
        if self._dragging:
            return  # Don't process clicks during drag

        if not self.window.is_pressed(ti.ui.LMB):
            return

        mouse = self.window.get_cursor_pos()
        hit = self._find_platform_at_screen_pos(mouse[0], mouse[1], grid)
        if hit is not None:
            self.selected_platform = hit
            return

        # Check ghost triangles
        ghost = self._find_ghost_at_screen_pos(mouse[0], mouse[1], grid)
        if ghost is not None and hasattr(grid, "add_platform"):
            grid.add_platform(ghost.row, ghost.col)

    def _find_platform_at_screen_pos(
        self, mx: float, my: float, grid
    ) -> Optional[Tuple[int, int]]:
        """Ray-cast against all platform positions to find the closest one
        under the cursor.  Returns (row, col) or None."""
        cam_pos, ray_dir = self._screen_to_ray(mx, my)
        if cam_pos is None:
            return None

        # Intersect with Y=platform_height plane (approximate)
        avg_height = self.global_height
        if abs(ray_dir[1]) < 1e-4:
            return None
        t = (avg_height - cam_pos[1]) / ray_dir[1]
        if t < 0:
            return None
        hit = cam_pos + ray_dir * t

        best = None
        best_dist = float("inf")
        R = grid.geometry.platform_radius * 0.8

        for pos, platform in grid.platforms.items():
            cx = platform.world_offset[0]
            cz = platform.world_offset[2]
            dist = math.sqrt((hit[0] - cx) ** 2 + (hit[2] - cz) ** 2)
            if dist < R and dist < best_dist:
                best = (pos.row, pos.col)
                best_dist = dist

        return best

    def _find_ghost_at_screen_pos(self, mx: float, my: float, grid):
        """Find a ghost triangle under the cursor.  Returns a GridPosition or None."""
        if not self.ghost_positions:
            return None

        cam_pos, ray_dir = self._screen_to_ray(mx, my)
        if cam_pos is None:
            return None

        if abs(ray_dir[1]) < 1e-4:
            return None
        t = (1.0 - cam_pos[1]) / ray_dir[1]
        if t < 0:
            return None
        hit = cam_pos + ray_dir * t

        best_pos = None
        best_dist = float("inf")
        R = grid.geometry.platform_radius * 0.9

        for gpos, world_pos, _rot in self.ghost_positions:
            dist = math.sqrt((hit[0] - world_pos[0]) ** 2 + (hit[2] - world_pos[2]) ** 2)
            if dist < R and dist < best_dist:
                best_dist = dist
                best_pos = gpos

        return best_pos

    def _screen_to_ray(self, mx: float, my: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Convert normalised screen coords (0..1) to a world-space ray
        (origin, direction)."""
        cos_p = math.cos(self.cam_pitch)
        sin_p = math.sin(self.cam_pitch)
        cos_y = math.cos(self.cam_yaw)
        sin_y = math.sin(self.cam_yaw)

        offset = np.array([
            self.cam_distance * cos_p * sin_y,
            self.cam_distance * sin_p,
            self.cam_distance * cos_p * cos_y,
        ])
        cam_pos = self.cam_target + offset
        cam_dir = self.cam_target - cam_pos
        cam_dir = cam_dir / np.linalg.norm(cam_dir)

        world_up = np.array([0.0, 1.0, 0.0])
        right = np.cross(cam_dir, world_up)
        r_len = np.linalg.norm(right)
        if r_len < 1e-4:
            right = np.array([1.0, 0.0, 0.0])
        else:
            right = right / r_len
        up = np.cross(right, cam_dir)
        up = up / np.linalg.norm(up)

        fov_rad = math.radians(45.0)
        aspect = self.width / self.height
        half_h = math.tan(fov_rad / 2.0)
        half_w = half_h * aspect

        ndc_x = (mx - 0.5) * 2.0
        ndc_y = (0.5 - my) * 2.0

        ray = cam_dir + right * (ndc_x * half_w) - up * (ndc_y * half_h)
        ray = ray / np.linalg.norm(ray)
        return cam_pos, ray

    # ------------------------------------------------------------------ #
    #  Window lifecycle
    # ------------------------------------------------------------------ #

    def should_close(self) -> bool:
        """Return *True* when the user has closed the window."""
        return not self.window.running

    def finish_frame(self) -> None:
        """Swap buffers and present the current frame."""
        self.window.show()

    def cleanup(self) -> None:
        """Release resources (scroll listener, etc.)."""
        self._stop_scroll_listener()
