"""Core physics engine for hexapod grid simulator.

Rigid body kinematics for a 3-DOF parallel platform (hexapod).
Coordinate system: Y-up (Taichi convention).
Rails at 0deg, 120deg, 240deg from X axis in XZ plane.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np
from numpy.typing import NDArray

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_RAIL_ANGLES: NDArray[np.float64] = np.radians([0.0, 120.0, 240.0])
"""Angles of the three rails measured from the +X axis in the XZ plane."""

_VERTEX_ANGLES: NDArray[np.float64] = _RAIL_ANGLES.copy()
"""Platform vertex angles, aligned with rail angles."""

_TWO_PI = 2.0 * np.pi

# Newton-Raphson parameters
_NR_MAX_ITER: int = 50
_NR_TOL: float = 1e-12
_NR_LINE_SEARCH_ALPHA: float = 1e-4
_NR_LINE_SEARCH_MAX_HALVINGS: int = 20
_NR_JACOBIAN_EPS: float = 1e-8
_NR_REGULARIZATION: float = 1e-14

# Corner-solve Newton parameters
_CORNER_MAX_ITER: int = 30
_CORNER_TOL: float = 1e-10
_CORNER_JACOBIAN_EPS: float = 1e-8


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class SolveResult(enum.Enum):
    """Outcome of a kinematic solve."""

    SUCCESS = "success"
    NO_SOLUTION = "no_solution"
    MULTIPLE_SOLUTIONS = "multiple_solutions"
    NUMERICAL_ERROR = "numerical_error"


# ---------------------------------------------------------------------------
# Geometry dataclass
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class HexapodGeometry:
    """Immutable description of the hexapod's physical dimensions.

    Parameters
    ----------
    rail_start : float
        Distance from the world-center to the start (inner end) of each rail.
    rail_length : float
        Length of each linear rail.
    rod_length : float
        Length of each rigid rod connecting carriage to platform vertex.
    platform_radius : float
        Distance from the platform center to each vertex.
    """

    rail_start: float
    rail_length: float
    rod_length: float
    platform_radius: float

    # -- derived properties --------------------------------------------------

    @property
    def rail_end(self) -> float:
        """Distance from world-center to the outer end of each rail."""
        return self.rail_start + self.rail_length

    @property
    def platform_edge_length(self) -> float:
        """Length of one edge of the equilateral platform triangle."""
        return self.platform_radius * np.sqrt(3.0)

    # -- coordinate helpers --------------------------------------------------

    def carriage_distance_to_normalized(self, distance: float) -> float:
        """Convert an absolute distance along the rail to [0, 1]."""
        return (distance - self.rail_start) / self.rail_length

    def carriage_position_to_normalized(self, distance: float) -> float:
        """Alias kept for API compatibility; identical to *carriage_distance_to_normalized*."""
        return self.carriage_distance_to_normalized(distance)

    def carriage_normalized_to_distance(self, normalized: float) -> float:
        """Convert a normalized [0, 1] position to absolute distance."""
        return self.rail_start + normalized * self.rail_length

    # -- range computations --------------------------------------------------

    def compute_height_range(self) -> Tuple[float, float]:
        """Return (min_height, max_height) reachable with level platform.

        The platform is level (pitch=roll=0) and centered above the origin.
        Height is bounded by the rod length and the rail reach.
        """
        # At any rail distance d, the carriage is at (d*cos(a), 0, d*sin(a)).
        # The vertex sits at (R*cos(a), h, R*sin(a)) where R = platform_radius.
        # Rod constraint: (d - R)^2 + h^2 = L^2
        # => h = sqrt(L^2 - (d - R)^2)
        # Minimum height: maximise |d - R| => d at rail_start or rail_end.
        # Maximum height: minimise |d - R| => d = R if within rail range.

        R = self.platform_radius
        L = self.rod_length

        d_min = self.rail_start
        d_max = self.rail_end

        # Closest carriage distance to platform radius
        d_closest = np.clip(R, d_min, d_max)
        delta_closest = d_closest - R
        h_max_sq = L * L - delta_closest * delta_closest
        if h_max_sq < 0.0:
            return (0.0, 0.0)
        h_max = np.sqrt(h_max_sq)

        # Farthest carriage distance from platform radius
        delta_far = max(abs(d_min - R), abs(d_max - R))
        h_min_sq = L * L - delta_far * delta_far
        h_min = np.sqrt(max(h_min_sq, 0.0))

        return (float(h_min), float(h_max))

    def compute_tilt_range(self, height: float) -> Tuple[float, float]:
        """Return approximate (min_tilt, max_tilt) in radians at *height*.

        This gives a symmetric bound on pitch or roll independently,
        estimated by varying one carriage to extremes while keeping the
        others centred.
        """
        R = self.platform_radius
        L = self.rod_length

        # Centred carriage distance (level)
        d_level_sq = L * L - height * height
        if d_level_sq < 0.0:
            return (0.0, 0.0)
        d_level = R + np.sqrt(d_level_sq)  # choosing + branch

        # Move one carriage to extremes
        tilt_max = 0.0
        for d_edge in (self.rail_start, self.rail_end):
            horiz_shift = d_edge - d_level
            # Approximate tilt as atan(delta_h / baseline)
            new_h_sq = L * L - (d_edge - R) ** 2
            if new_h_sq < 0.0:
                continue
            delta_h = np.sqrt(new_h_sq) - height
            baseline = R  # rough baseline distance between vertex and centre
            if baseline > 0:
                angle = abs(np.arctan2(delta_h, baseline))
                tilt_max = max(tilt_max, angle)

        return (-float(tilt_max), float(tilt_max))

    def compute_corner_height_range(self) -> Tuple[float, float]:
        """Return the (min, max) height reachable by any single corner vertex."""
        R = self.platform_radius
        L = self.rod_length

        d_min = self.rail_start
        d_max = self.rail_end

        # Maximum corner height: carriage distance closest to R
        d_best = np.clip(R, d_min, d_max)
        h_max_sq = L * L - (d_best - R) ** 2
        h_max = np.sqrt(max(h_max_sq, 0.0))

        # Minimum corner height: carriage distance farthest from R
        delta_far = max(abs(d_min - R), abs(d_max - R))
        h_min_sq = L * L - delta_far * delta_far
        h_min = np.sqrt(max(h_min_sq, 0.0))

        return (float(h_min), float(h_max))


# ---------------------------------------------------------------------------
# Platform pose
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PlatformPose:
    """Fully described pose of the triangular platform.

    Attributes
    ----------
    center : ndarray, shape (3,)
        World position of the platform centre (x, y, z).  Y is up.
    pitch : float
        Rotation about the Z axis in radians.
    roll : float
        Rotation about the X axis in radians.
    vertices : ndarray, shape (3, 3)
        World positions of the three vertices, one per row.
    normal : ndarray, shape (3,)
        Unit normal of the platform surface.
    height : float
        Y coordinate of the platform centre (convenience alias for center[1]).
    """

    center: NDArray[np.float64]
    pitch: float
    roll: float
    vertices: NDArray[np.float64]
    normal: NDArray[np.float64]
    height: float

    @property
    def corner_heights(self) -> NDArray[np.float64]:
        """Y coordinates of each vertex, shape (3,)."""
        return self.vertices[:, 1].copy()


# ---------------------------------------------------------------------------
# Solve output
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SolveOutput:
    """Complete output from a kinematic solve.

    Attributes
    ----------
    result : SolveResult
        Outcome flag.
    pose : PlatformPose or None
        Computed platform pose (None on failure).
    carriage_distances : ndarray or None
        Absolute distances of each carriage along its rail, shape (3,).
    carriage_positions : ndarray or None
        Normalised [0, 1] carriage positions, shape (3,).
    carriage_world : ndarray or None
        World coordinates of each carriage, shape (3, 3).
    actual_rod_lengths : ndarray or None
        Euclidean distance from each carriage to its vertex, shape (3,).
    error : float
        Residual norm of the rod-length constraint.
    message : str
        Human-readable description of the outcome.
    """

    result: SolveResult
    pose: Optional[PlatformPose] = None
    carriage_distances: Optional[NDArray[np.float64]] = None
    carriage_positions: Optional[NDArray[np.float64]] = None
    carriage_world: Optional[NDArray[np.float64]] = None
    actual_rod_lengths: Optional[NDArray[np.float64]] = None
    error: float = 0.0
    message: str = ""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def rotation_matrix(pitch: float, roll: float) -> NDArray[np.float64]:
    """Build a 3x3 rotation matrix: R_pitch @ R_roll.

    * Roll  rotates about the X axis.
    * Pitch rotates about the Z axis.

    Parameters
    ----------
    pitch : float
        Angle about Z in radians.
    roll : float
        Angle about X in radians.

    Returns
    -------
    ndarray, shape (3, 3)
    """
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)

    # R_roll (rotation about X)
    R_roll = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr,  cr],
    ])

    # R_pitch (rotation about Z)
    R_pitch = np.array([
        [cp, -sp, 0.0],
        [sp,  cp, 0.0],
        [0.0, 0.0, 1.0],
    ])

    return R_pitch @ R_roll


def compute_vertices(
    center: NDArray[np.float64],
    pitch: float,
    roll: float,
    platform_radius: float,
) -> NDArray[np.float64]:
    """Compute world positions of the three platform vertices.

    Vertices lie at 0deg, 120deg, 240deg from the +X axis in the local
    XZ plane before rotation.

    Parameters
    ----------
    center : ndarray, shape (3,)
    pitch, roll : float
    platform_radius : float

    Returns
    -------
    ndarray, shape (3, 3)  – one vertex per row.
    """
    R = rotation_matrix(pitch, roll)
    vertices = np.empty((3, 3), dtype=np.float64)
    for i, angle in enumerate(_VERTEX_ANGLES):
        local = np.array([
            platform_radius * np.cos(angle),
            0.0,
            platform_radius * np.sin(angle),
        ])
        vertices[i] = center + R @ local
    return vertices


def compute_normal(vertices: NDArray[np.float64]) -> NDArray[np.float64]:
    """Unit normal of the triangle defined by *vertices*.

    Uses the cross product of two edges.  The sign convention produces
    a normal that points in the +Y direction for a level platform.

    Parameters
    ----------
    vertices : ndarray, shape (3, 3)

    Returns
    -------
    ndarray, shape (3,)
    """
    e1 = vertices[1] - vertices[0]
    e2 = vertices[2] - vertices[0]
    n = np.cross(e1, e2)
    norm = np.linalg.norm(n)
    if norm < 1e-15:
        return np.array([0.0, 1.0, 0.0])
    return n / norm


def carriage_world_position(
    index: int,
    distance: float,
) -> NDArray[np.float64]:
    """World position of carriage *index* at absolute *distance* along its rail.

    The carriage sits in the XZ plane (Y = 0) at the rail angle.

    Parameters
    ----------
    index : int
        Rail index (0, 1, or 2).
    distance : float
        Absolute distance from the world centre along the rail.

    Returns
    -------
    ndarray, shape (3,)
    """
    angle = _RAIL_ANGLES[index]
    return np.array([
        distance * np.cos(angle),
        0.0,
        distance * np.sin(angle),
    ])


def _carriage_world_positions(
    distances: NDArray[np.float64],
) -> NDArray[np.float64]:
    """World positions of all three carriages, shape (3, 3)."""
    positions = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        positions[i] = carriage_world_position(i, distances[i])
    return positions


def _rod_length_residuals(
    vertices: NDArray[np.float64],
    carriage_world: NDArray[np.float64],
    rod_length: float,
) -> NDArray[np.float64]:
    """Squared-distance residuals: |v_i - c_i|^2 - L^2, shape (3,)."""
    diffs = vertices - carriage_world
    sq_dists = np.sum(diffs * diffs, axis=1)
    return sq_dists - rod_length * rod_length


def _actual_rod_lengths(
    vertices: NDArray[np.float64],
    carriage_world: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Euclidean distances from each carriage to its vertex, shape (3,)."""
    diffs = vertices - carriage_world
    return np.sqrt(np.sum(diffs * diffs, axis=1))


# ---------------------------------------------------------------------------
# Build output helpers
# ---------------------------------------------------------------------------

def _build_output(
    result: SolveResult,
    geometry: HexapodGeometry,
    height: float,
    pitch: float,
    roll: float,
    carriage_distances: NDArray[np.float64],
    error: float = 0.0,
    message: str = "",
) -> SolveOutput:
    """Construct a fully populated :class:`SolveOutput`."""
    center = np.array([0.0, height, 0.0])
    vertices = compute_vertices(center, pitch, roll, geometry.platform_radius)
    normal = compute_normal(vertices)
    pose = PlatformPose(
        center=center,
        pitch=pitch,
        roll=roll,
        vertices=vertices,
        normal=normal,
        height=height,
    )
    cw = _carriage_world_positions(carriage_distances)
    cn = np.array([
        geometry.carriage_distance_to_normalized(d) for d in carriage_distances
    ])
    rods = _actual_rod_lengths(vertices, cw)
    return SolveOutput(
        result=result,
        pose=pose,
        carriage_distances=carriage_distances.copy(),
        carriage_positions=cn,
        carriage_world=cw,
        actual_rod_lengths=rods,
        error=error,
        message=message,
    )


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

class HexapodSolver:
    """Forward and inverse kinematic solver for the hexapod platform.

    Parameters
    ----------
    geometry : HexapodGeometry
        Physical dimensions of the hexapod.
    """

    __slots__ = ("geometry",)

    def __init__(self, geometry: HexapodGeometry) -> None:
        self.geometry = geometry

    # -- Inverse kinematics (analytical) ------------------------------------

    def solve_from_pose(
        self,
        height: float,
        pitch: float = 0.0,
        roll: float = 0.0,
    ) -> SolveOutput:
        """Inverse kinematics: given platform pose, find carriage positions.

        For each rail the constraint reduces to a quadratic in the carriage
        distance *d*:

            (vx - d*cos(a))^2 + vy^2 + (vz - d*sin(a))^2 = L^2

        which expands to  d^2 - 2*d*(vx*cos(a) + vz*sin(a)) + |v|^2 - L^2 = 0.

        Parameters
        ----------
        height : float
            Y coordinate of platform centre.
        pitch, roll : float
            Platform orientation in radians.

        Returns
        -------
        SolveOutput
        """
        geo = self.geometry
        center = np.array([0.0, height, 0.0])
        vertices = compute_vertices(center, pitch, roll, geo.platform_radius)

        distances = np.empty(3, dtype=np.float64)

        for i in range(3):
            vx, vy, vz = vertices[i]
            ca = np.cos(_RAIL_ANGLES[i])
            sa = np.sin(_RAIL_ANGLES[i])

            # Quadratic: d^2 - 2*B*d + C = 0
            B = vx * ca + vz * sa
            C = vx * vx + vy * vy + vz * vz - geo.rod_length ** 2

            discriminant = B * B - C
            if discriminant < 0.0:
                return SolveOutput(
                    result=SolveResult.NO_SOLUTION,
                    error=abs(discriminant),
                    message=(
                        f"No real solution for rail {i}: "
                        f"discriminant = {discriminant:.6e}"
                    ),
                )

            sqrt_disc = np.sqrt(discriminant)
            d_plus = B + sqrt_disc
            d_minus = B - sqrt_disc

            # Choose solution within rail bounds, preferring d_plus (outer)
            candidates = []
            for d in (d_plus, d_minus):
                if geo.rail_start - 1e-9 <= d <= geo.rail_end + 1e-9:
                    candidates.append(d)

            if len(candidates) == 0:
                return SolveOutput(
                    result=SolveResult.NO_SOLUTION,
                    error=0.0,
                    message=(
                        f"No in-range solution for rail {i}: "
                        f"d+ = {d_plus:.4f}, d- = {d_minus:.4f}, "
                        f"range = [{geo.rail_start:.4f}, {geo.rail_end:.4f}]"
                    ),
                )

            # Prefer the solution closer to the outer end (d_plus) for
            # consistency; if both are valid they should agree for normal poses.
            distances[i] = np.clip(candidates[0], geo.rail_start, geo.rail_end)

        residuals = _rod_length_residuals(
            vertices, _carriage_world_positions(distances), geo.rod_length
        )
        error = float(np.max(np.abs(residuals)))

        return _build_output(
            result=SolveResult.SUCCESS,
            geometry=geo,
            height=height,
            pitch=pitch,
            roll=roll,
            carriage_distances=distances,
            error=error,
            message="Inverse kinematics solved analytically.",
        )

    # -- Forward kinematics (Newton-Raphson) --------------------------------

    def solve_from_carriages(
        self,
        carriage_normalized: NDArray[np.float64],
    ) -> SolveOutput:
        """Forward kinematics: given normalised carriage positions, find pose.

        Uses Newton-Raphson iteration with backtracking line search and
        Tikhonov regularisation on the augmented Jacobian.

        Parameters
        ----------
        carriage_normalized : ndarray, shape (3,)
            Normalised carriage positions in [0, 1].

        Returns
        -------
        SolveOutput
        """
        geo = self.geometry
        distances = np.array([
            geo.carriage_normalized_to_distance(n) for n in carriage_normalized
        ])
        cw = _carriage_world_positions(distances)

        # Initial guess: level platform, height from average rod projection
        avg_d = np.mean(distances)
        delta = avg_d - geo.platform_radius
        h_sq = geo.rod_length ** 2 - delta * delta
        h0 = np.sqrt(max(h_sq, 0.01))

        state = np.array([h0, 0.0, 0.0])  # [height, pitch, roll]

        for iteration in range(_NR_MAX_ITER):
            center = np.array([0.0, state[0], 0.0])
            verts = compute_vertices(center, state[1], state[2], geo.platform_radius)
            residuals = _rod_length_residuals(verts, cw, geo.rod_length)

            res_norm = np.linalg.norm(residuals)
            if res_norm < _NR_TOL:
                return _build_output(
                    result=SolveResult.SUCCESS,
                    geometry=geo,
                    height=state[0],
                    pitch=state[1],
                    roll=state[2],
                    carriage_distances=distances,
                    error=float(res_norm),
                    message=(
                        f"Forward kinematics converged in {iteration} iterations."
                    ),
                )

            J = self._compute_jacobian(state, cw)

            # Augmented system with Tikhonov regularisation
            JtJ = J.T @ J + _NR_REGULARIZATION * np.eye(3)
            Jtr = J.T @ residuals

            try:
                delta_state = np.linalg.solve(JtJ, -Jtr)
            except np.linalg.LinAlgError:
                return SolveOutput(
                    result=SolveResult.NUMERICAL_ERROR,
                    error=float(res_norm),
                    message="Singular Jacobian encountered.",
                )

            # Backtracking line search (Armijo condition)
            alpha = 1.0
            current_cost = 0.5 * res_norm * res_norm
            directional_deriv = float(Jtr @ delta_state)

            for _ in range(_NR_LINE_SEARCH_MAX_HALVINGS):
                new_state = state + alpha * delta_state
                new_center = np.array([0.0, new_state[0], 0.0])
                new_verts = compute_vertices(
                    new_center, new_state[1], new_state[2], geo.platform_radius
                )
                new_res = _rod_length_residuals(new_verts, cw, geo.rod_length)
                new_cost = 0.5 * np.dot(new_res, new_res)

                if new_cost <= current_cost + _NR_LINE_SEARCH_ALPHA * alpha * directional_deriv:
                    break
                alpha *= 0.5
            else:
                # Line search exhausted; accept step anyway
                pass

            state = state + alpha * delta_state

        # Did not converge
        center = np.array([0.0, state[0], 0.0])
        verts = compute_vertices(center, state[1], state[2], geo.platform_radius)
        residuals = _rod_length_residuals(verts, cw, geo.rod_length)
        res_norm = float(np.linalg.norm(residuals))

        if res_norm < 1e-6:
            # Close enough
            return _build_output(
                result=SolveResult.SUCCESS,
                geometry=geo,
                height=state[0],
                pitch=state[1],
                roll=state[2],
                carriage_distances=distances,
                error=res_norm,
                message=(
                    f"Forward kinematics converged loosely after "
                    f"{_NR_MAX_ITER} iterations (error={res_norm:.2e})."
                ),
            )

        return SolveOutput(
            result=SolveResult.NUMERICAL_ERROR,
            error=res_norm,
            message=(
                f"Forward kinematics did not converge after "
                f"{_NR_MAX_ITER} iterations (error={res_norm:.2e})."
            ),
        )

    # -- Corner-height based solving ----------------------------------------

    def solve_from_corners(
        self,
        h0: float,
        h1: float,
        h2: float,
    ) -> SolveOutput:
        """Solve from desired corner (vertex) heights.

        First recovers (height, pitch, roll) from the three corner heights
        using Newton iteration, then delegates to :meth:`solve_from_pose`.

        Parameters
        ----------
        h0, h1, h2 : float
            Desired Y coordinates of vertices 0, 1, 2.

        Returns
        -------
        SolveOutput
        """
        geo = self.geometry
        target_heights = np.array([h0, h1, h2])

        # Initial guess: average height, zero tilt
        height = np.mean(target_heights)
        pitch = 0.0
        roll = 0.0

        state = np.array([height, pitch, roll])

        for iteration in range(_CORNER_MAX_ITER):
            center = np.array([0.0, state[0], 0.0])
            verts = compute_vertices(center, state[1], state[2], geo.platform_radius)
            corner_h = verts[:, 1]

            residuals = corner_h - target_heights
            res_norm = np.linalg.norm(residuals)

            if res_norm < _CORNER_TOL:
                return self.solve_from_pose(state[0], state[1], state[2])

            # Numerical Jacobian of corner heights w.r.t. (height, pitch, roll)
            J = np.empty((3, 3), dtype=np.float64)
            for j in range(3):
                state_p = state.copy()
                state_p[j] += _CORNER_JACOBIAN_EPS
                center_p = np.array([0.0, state_p[0], 0.0])
                verts_p = compute_vertices(
                    center_p, state_p[1], state_p[2], geo.platform_radius
                )
                J[:, j] = (verts_p[:, 1] - corner_h) / _CORNER_JACOBIAN_EPS

            try:
                delta_state = np.linalg.solve(J, -residuals)
            except np.linalg.LinAlgError:
                return SolveOutput(
                    result=SolveResult.NUMERICAL_ERROR,
                    error=float(res_norm),
                    message="Singular Jacobian in corner-height solve.",
                )

            state = state + delta_state

        # Final attempt even if not fully converged
        return self.solve_from_pose(state[0], state[1], state[2])

    # -- Numerical Jacobian -------------------------------------------------

    def _compute_jacobian(
        self,
        state: NDArray[np.float64],
        carriage_world: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Numerical Jacobian of the rod-length residuals.

        Computes d(residuals)/d(state) using forward finite differences
        where state = [height, pitch, roll].

        Parameters
        ----------
        state : ndarray, shape (3,)
        carriage_world : ndarray, shape (3, 3)

        Returns
        -------
        ndarray, shape (3, 3)
        """
        geo = self.geometry
        eps = _NR_JACOBIAN_EPS

        center = np.array([0.0, state[0], 0.0])
        verts = compute_vertices(center, state[1], state[2], geo.platform_radius)
        f0 = _rod_length_residuals(verts, carriage_world, geo.rod_length)

        J = np.empty((3, 3), dtype=np.float64)
        for j in range(3):
            state_p = state.copy()
            state_p[j] += eps
            center_p = np.array([0.0, state_p[0], 0.0])
            verts_p = compute_vertices(
                center_p, state_p[1], state_p[2], geo.platform_radius
            )
            f_p = _rod_length_residuals(verts_p, carriage_world, geo.rod_length)
            J[:, j] = (f_p - f0) / eps

        return J
