"""Triangle topology handling for PDGCN HDF5 files."""

from __future__ import annotations

import numpy as np


def read_or_recover_triangles(h5_file):
    """Return triangle connectivity and a short source label."""

    if "mesh/triangles" in h5_file:
        triangles = np.asarray(h5_file["mesh/triangles"][()], dtype=np.int64)
        return _validate_triangles(triangles), "mesh/triangles"
    if "edge_index" not in h5_file:
        raise KeyError("Missing mesh/triangles and edge_index; cannot build FEniCS mesh.")
    triangles = recover_triangles_from_edge_index(h5_file["edge_index"][()])
    return triangles, "edge_index_3cycles"


def recover_triangles_from_edge_index(edge_index) -> np.ndarray:
    edge_index = np.asarray(edge_index, dtype=np.int64)
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError(f"edge_index must have shape [2, E], got {edge_index.shape}.")

    undirected = {
        (int(min(a, b)), int(max(a, b)))
        for a, b in edge_index.T
        if int(a) != int(b)
    }
    if not undirected:
        raise ValueError("edge_index contains no usable edges.")

    num_nodes = max(max(edge) for edge in undirected) + 1
    adjacency = [set() for _ in range(num_nodes)]
    for a, b in undirected:
        adjacency[a].add(b)
        adjacency[b].add(a)

    triangles = []
    for i in range(num_nodes):
        greater_neighbors = {j for j in adjacency[i] if j > i}
        for j in sorted(greater_neighbors):
            common = greater_neighbors.intersection(k for k in adjacency[j] if k > j)
            for k in sorted(common):
                triangles.append((i, j, k))

    if not triangles:
        raise ValueError("No triangular 3-cycles were found in edge_index.")
    return np.asarray(triangles, dtype=np.int64)


def orient_triangles_xy(triangles, xy, *, area_tolerance=1.0e-24) -> np.ndarray:
    triangles = _validate_triangles(triangles)
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError(f"xy must have shape [N, 2], got {xy.shape}.")

    oriented = []
    for tri in triangles:
        pts = xy[tri]
        area2 = _signed_area2(pts)
        if abs(area2) <= area_tolerance:
            continue
        if area2 < 0.0:
            tri = np.asarray([tri[0], tri[2], tri[1]], dtype=np.int64)
        oriented.append(tri)

    if not oriented:
        raise ValueError("All triangles are degenerate in the first-frame xy projection.")
    return np.asarray(oriented, dtype=np.int64)


def orient_triangles_3d(triangles, xyz, normal, *, area_tolerance=1.0e-24) -> np.ndarray:
    triangles = _validate_triangles(triangles)
    xyz = np.asarray(xyz, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f"xyz must have shape [N, 3], got {xyz.shape}.")
    if normal.shape != xyz.shape:
        raise ValueError(f"normal must match xyz shape {xyz.shape}, got {normal.shape}.")

    oriented = []
    for tri in triangles:
        pts = xyz[tri]
        geo_normal = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        area2 = float(np.linalg.norm(geo_normal))
        if area2 <= area_tolerance:
            continue

        reference_normal = np.mean(normal[tri], axis=0)
        if float(np.linalg.norm(reference_normal)) <= 1.0e-15:
            reference_normal = geo_normal
        if float(np.dot(geo_normal, reference_normal)) < 0.0:
            tri = np.asarray([tri[0], tri[2], tri[1]], dtype=np.int64)
        oriented.append(tri)

    if not oriented:
        raise ValueError("All triangles are degenerate in the first-frame 3D surface.")
    return np.asarray(oriented, dtype=np.int64)


def validate_triangle_indices(triangles, num_nodes: int) -> None:
    triangles = _validate_triangles(triangles)
    if triangles.size == 0:
        raise ValueError("mesh/triangles must not be empty.")
    min_idx = int(triangles.min())
    max_idx = int(triangles.max())
    if min_idx < 0 or max_idx >= int(num_nodes):
        raise ValueError(
            f"Triangle node indices must be within [0, {num_nodes - 1}], got [{min_idx}, {max_idx}]."
        )


def _validate_triangles(triangles) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError(f"mesh/triangles must have shape [M, 3], got {triangles.shape}.")
    return triangles


def _signed_area2(points: np.ndarray) -> float:
    return float(
        (points[1, 0] - points[0, 0]) * (points[2, 1] - points[0, 1])
        - (points[2, 0] - points[0, 0]) * (points[1, 1] - points[0, 1])
    )
