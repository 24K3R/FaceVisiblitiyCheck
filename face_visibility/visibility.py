"""Directional visible-face extraction for explicit surface elements.

The module assumes the caller has already discretized explicit surfaces into face
vertices. Visibility is evaluated from an orthographic camera looking along a
specified direction. The implementation combines front-face culling with a small
software z-buffer so that faces hidden behind nearer geometry are removed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, floor, inf, sqrt
from typing import Iterable, Sequence

Vec3 = tuple[float, float, float]
ProjectedPoint = tuple[float, float, float]


@dataclass(frozen=True)
class Face:
    """A discretized surface element.

    Attributes:
        id: Stable identifier returned in the visibility result.
        vertices: Three or more 3-D points ordered around the face boundary.
        payload: Optional user data, for example the original parametric cell
            indices returned by an existing meshing/discretization method.
    """

    id: object
    vertices: Sequence[Sequence[float]]
    payload: object | None = None


@dataclass(frozen=True)
class VisibilityResult:
    """Visible face ids and the corresponding face objects."""

    visible_ids: list[object]
    visible_faces: list[Face]


def _as_vec3(vector: Sequence[float], name: str) -> Vec3:
    if len(vector) != 3:
        raise ValueError(f"{name} must be a 3-D vector")
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _dot(left: Vec3, right: Vec3) -> float:
    return left[0] * right[0] + left[1] * right[1] + left[2] * right[2]


def _cross(left: Vec3, right: Vec3) -> Vec3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(vector: Sequence[float]) -> Vec3:
    x, y, z = _as_vec3(vector, "vector")
    norm = sqrt(x * x + y * y + z * z)
    if norm == 0.0:
        raise ValueError("view_direction must be a non-zero 3-D vector")
    return (x / norm, y / norm, z / norm)


def _camera_basis(view_direction: Sequence[float]) -> tuple[Vec3, Vec3, Vec3]:
    """Return right, up, and forward vectors for an orthographic camera."""

    forward = _normalize(view_direction)
    helper: Vec3 = (0.0, 0.0, 1.0)
    if abs(_dot(forward, helper)) > 0.95:
        helper = (0.0, 1.0, 0.0)
    right = _normalize(_cross(helper, forward))
    up = _cross(forward, right)
    return right, up, forward


def _face_normal(vertices: Sequence[Sequence[float]]) -> Vec3:
    """Compute a robust polygon normal with Newell's method."""

    nx = ny = nz = 0.0
    points = [_as_vec3(vertex, "face vertex") for vertex in vertices]
    for index, current in enumerate(points):
        nxt = points[(index + 1) % len(points)]
        nx += (current[1] - nxt[1]) * (current[2] + nxt[2])
        ny += (current[2] - nxt[2]) * (current[0] + nxt[0])
        nz += (current[0] - nxt[0]) * (current[1] + nxt[1])
    return _normalize((nx, ny, nz))


def is_front_facing(
    vertices: Sequence[Sequence[float]],
    view_direction: Sequence[float],
    *,
    two_sided: bool = False,
    eps: float = 1e-12,
) -> bool:
    """Return whether a face can face an observer looking along ``view_direction``.

    ``view_direction`` points from the observer toward the model. A one-sided
    face is front-facing when its outward normal points back toward the observer,
    i.e. ``dot(normal, view_direction) < 0``. Set ``two_sided`` for sheets whose
    orientation is unknown or intentionally visible from both sides.
    """

    if two_sided:
        return True
    normal = _face_normal(vertices)
    forward = _normalize(view_direction)
    return _dot(normal, forward) < -eps


def _triangulate_projected(poly: Sequence[ProjectedPoint]) -> Iterable[tuple[ProjectedPoint, ProjectedPoint, ProjectedPoint]]:
    """Fan-triangulate a projected convex or nearly convex polygon."""

    anchor = poly[0]
    for index in range(1, len(poly) - 1):
        yield anchor, poly[index], poly[index + 1]


def _rasterize_triangle(
    triangle: tuple[ProjectedPoint, ProjectedPoint, ProjectedPoint],
    width: int,
    height: int,
) -> Iterable[tuple[int, int, float]]:
    """Yield covered pixel coordinates and interpolated depth for a triangle."""

    p0, p1, p2 = triangle
    min_x = max(0, int(floor(min(p0[0], p1[0], p2[0]))))
    max_x = min(width - 1, int(ceil(max(p0[0], p1[0], p2[0]))))
    min_y = max(0, int(floor(min(p0[1], p1[1], p2[1]))))
    max_y = min(height - 1, int(ceil(max(p0[1], p1[1], p2[1]))))
    area = (p1[1] - p2[1]) * (p0[0] - p2[0]) + (p2[0] - p1[0]) * (p0[1] - p2[1])
    if abs(area) < 1e-12:
        return

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            sample_x = x + 0.5
            sample_y = y + 0.5
            w0 = ((p1[1] - p2[1]) * (sample_x - p2[0]) + (p2[0] - p1[0]) * (sample_y - p2[1])) / area
            w1 = ((p2[1] - p0[1]) * (sample_x - p2[0]) + (p0[0] - p2[0]) * (sample_y - p2[1])) / area
            w2 = 1.0 - w0 - w1
            if w0 >= -1e-9 and w1 >= -1e-9 and w2 >= -1e-9:
                depth = w0 * p0[2] + w1 * p1[2] + w2 * p2[2]
                yield x, y, depth


def extract_visible_faces(
    faces: Sequence[Face],
    view_direction: Sequence[float],
    *,
    resolution: int = 512,
    two_sided: bool = False,
    coverage_threshold: int = 1,
    depth_eps: float = 1e-9,
) -> VisibilityResult:
    """Extract faces visible from an orthographic view direction.

    Algorithm:
        1. Remove back-facing faces by normal/view-direction dot product.
        2. Project remaining face vertices into a camera plane perpendicular to
           ``view_direction``.
        3. Rasterize projected triangles into a z-buffer. Smaller depth means
           closer to the observer because depth is ``dot(point, view_direction)``.
        4. Return faces that win at least ``coverage_threshold`` pixels.

    This is appropriate for explicit surfaces discretized into triangles/quads or
    small convex polygons. Increase ``resolution`` for tiny elements or for more
    accurate silhouette decisions.
    """

    if resolution <= 0:
        raise ValueError("resolution must be positive")
    if coverage_threshold <= 0:
        raise ValueError("coverage_threshold must be positive")

    right, up, forward = _camera_basis(view_direction)
    candidates: list[tuple[Face, list[ProjectedPoint]]] = []
    all_x: list[float] = []
    all_y: list[float] = []

    for face in faces:
        if len(face.vertices) < 3:
            raise ValueError(f"face {face.id!r} must contain at least three 3-D vertices")
        points = [_as_vec3(vertex, "face vertex") for vertex in face.vertices]
        if not is_front_facing(points, forward, two_sided=two_sided):
            continue
        projected = [(_dot(point, right), _dot(point, up), _dot(point, forward)) for point in points]
        candidates.append((face, projected))
        all_x.extend(point[0] for point in projected)
        all_y.extend(point[1] for point in projected)

    if not candidates:
        return VisibilityResult([], [])

    min_x, max_x = min(all_x), max(all_x)
    min_y, max_y = min(all_y), max(all_y)
    span_x = max(max_x - min_x, 1e-12)
    span_y = max(max_y - min_y, 1e-12)
    scale = (resolution - 1) / max(span_x, span_y)
    offset_x = ((resolution - 1) - span_x * scale) / 2.0
    offset_y = ((resolution - 1) - span_y * scale) / 2.0

    depth_buffer = [[inf for _ in range(resolution)] for _ in range(resolution)]
    owner_buffer = [[-1 for _ in range(resolution)] for _ in range(resolution)]

    for face_index, (_, projected) in enumerate(candidates):
        pixel_poly = [
            ((x - min_x) * scale + offset_x, (y - min_y) * scale + offset_y, depth)
            for x, y, depth in projected
        ]
        for triangle in _triangulate_projected(pixel_poly):
            for x, y, depth in _rasterize_triangle(triangle, resolution, resolution):
                if depth < depth_buffer[y][x] - depth_eps:
                    depth_buffer[y][x] = depth
                    owner_buffer[y][x] = face_index

    coverage = [0 for _ in candidates]
    for row in owner_buffer:
        for owner in row:
            if owner >= 0:
                coverage[owner] += 1

    visible_indices = [index for index, count in enumerate(coverage) if count >= coverage_threshold]
    visible_faces = [candidates[index][0] for index in visible_indices]
    return VisibilityResult([face.id for face in visible_faces], visible_faces)
