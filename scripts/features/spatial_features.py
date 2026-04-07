import math
from typing import Dict, Sequence

import numpy as np
from scipy.spatial import Delaunay, cKDTree
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

from .common import safe_iqr


def compute_spatial_features(points: np.ndarray,
                             img_shape_um,
                             grid_k: int = 10,
                             knn_values: Sequence[int] = (1, 3, 5)) -> Dict[str, float]:
    out: Dict[str, float] = {}
    pts = np.asarray(points, dtype=np.float64)
    out.update(nearest_neighbor_stats(pts, img_shape_um))
    out.update(grid_dispersion(pts, img_shape_um, K=grid_k))
    out.update(knn_distance_features(pts, ks=knn_values))
    out.update(delaunay_features(pts))
    out.update(proximity_graph_features(pts))
    return out


def nearest_neighbor_stats(points: np.ndarray,
                           img_shape_um) -> Dict[str, float]:
    h_um, w_um = img_shape_um[:2]
    img_area_um2 = float(h_um * w_um)
    out = {
        "density_per_mm2": float("nan"),
        "nn_mean_um": float("nan"),
        "nn_sd_um": float("nan"),
        "clark_evans_R": float("nan"),
        "clark_evans_z": float("nan"),
    }
    if img_area_um2 > 0:
        img_area_mm2 = img_area_um2 / 1_000_000.0
        out["density_per_mm2"] = (float(points.shape[0]) / img_area_mm2) if img_area_mm2 > 0 else float("nan")
    if points.shape[0] < 2:
        return out
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    nn_um = dists[:, 1]
    out["nn_mean_um"] = float(np.mean(nn_um))
    out["nn_sd_um"] = float(np.std(nn_um, ddof=1)) if nn_um.size > 1 else 0.0

    lam = float(points.shape[0]) / img_area_um2 if img_area_um2 > 0 else float("nan")
    if lam > 0:
        re = 1.0 / (2.0 * math.sqrt(lam))
        ro = out["nn_mean_um"]
        out["clark_evans_R"] = float(ro / re) if re > 0 else float("nan")
        var_ro = (4.0 - math.pi) / (4.0 * math.pi * lam * float(points.shape[0]))
        if var_ro > 0:
            out["clark_evans_z"] = float((ro - re) / math.sqrt(var_ro))
    return out


def grid_dispersion(points: np.ndarray, img_shape, K: int = 10) -> Dict[str, float]:
    h, w = img_shape[:2]
    if points.shape[0] == 0:
        return {"grid_vmr": float("nan"), "grid_cv": float("nan")}
    rows = np.clip((points[:, 0] * K / max(h, 1)).astype(int), 0, K - 1)
    cols = np.clip((points[:, 1] * K / max(w, 1)).astype(int), 0, K - 1)
    counts = np.zeros((K, K), dtype=np.float64)
    np.add.at(counts, (rows, cols), 1.0)
    vals = counts.ravel()
    mean = float(np.mean(vals))
    if mean <= 0:
        return {"grid_vmr": float("nan"), "grid_cv": float("nan")}
    var = float(np.var(vals, ddof=1)) if vals.size > 1 else 0.0
    return {
        "grid_vmr": float(var / mean),
        "grid_cv": float(math.sqrt(var) / mean),
    }


def knn_distance_features(points: np.ndarray, ks: Sequence[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if points.shape[0] < 2:
        for k in ks:
            out[f"knn_k{k}_mean_um"] = float("nan")
        return out
    max_k = max(int(k) for k in ks)
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=min(points.shape[0], max_k + 1))
    for k in ks:
        kk = int(k)
        if dists.shape[1] <= kk:
            out[f"knn_k{kk}_mean_um"] = float("nan")
            continue
        out[f"knn_k{kk}_mean_um"] = float(np.mean(dists[:, kk]))
    return out


def delaunay_features(points: np.ndarray) -> Dict[str, float]:
    out = {
        "delaunay_edge_mean_um": float("nan"),
        "delaunay_edge_sd_um": float("nan"),
        "delaunay_edge_iqr_um": float("nan"),
        "delaunay_triangle_area_mean_um2": float("nan"),
        "delaunay_triangle_area_sd_um2": float("nan"),
        "mst_edge_mean_um": float("nan"),
        "mst_edge_sd_um": float("nan"),
        "mst_edge_iqr_um": float("nan"),
    }
    if points.shape[0] < 3:
        return out
    try:
        tri = Delaunay(points[:, ::-1])
    except Exception:
        return out
    edges = set()
    triangle_areas = []
    for simplex in tri.simplices:
        a, b, c = simplex
        edges.add(tuple(sorted((int(a), int(b)))))
        edges.add(tuple(sorted((int(a), int(c)))))
        edges.add(tuple(sorted((int(b), int(c)))))
        pa, pb, pc = points[a], points[b], points[c]
        area = 0.5 * abs(
            (pb[1] - pa[1]) * (pc[0] - pa[0]) -
            (pb[0] - pa[0]) * (pc[1] - pa[1])
        )
        triangle_areas.append(float(area))
    if edges:
        edge_lengths = np.array([
            math.dist(tuple(points[i]), tuple(points[j]))
            for i, j in sorted(edges)
        ], dtype=np.float64)
        out["delaunay_edge_mean_um"] = float(np.mean(edge_lengths))
        out["delaunay_edge_sd_um"] = float(np.std(edge_lengths, ddof=1)) if edge_lengths.size > 1 else 0.0
        out["delaunay_edge_iqr_um"] = safe_iqr(edge_lengths)
        mst = _mst_lengths(points.shape[0], sorted(edges), edge_lengths)
        if mst.size:
            out["mst_edge_mean_um"] = float(np.mean(mst))
            out["mst_edge_sd_um"] = float(np.std(mst, ddof=1)) if mst.size > 1 else 0.0
            out["mst_edge_iqr_um"] = safe_iqr(mst)
    if triangle_areas:
        tri_arr = np.asarray(triangle_areas, dtype=np.float64)
        out["delaunay_triangle_area_mean_um2"] = float(np.mean(tri_arr))
        out["delaunay_triangle_area_sd_um2"] = float(np.std(tri_arr, ddof=1)) if tri_arr.size > 1 else 0.0
    return out


def proximity_graph_features(points: np.ndarray) -> Dict[str, float]:
    out = {
        "graph_radius_um": float("nan"),
        "graph_mean_degree": float("nan"),
        "graph_sd_degree": float("nan"),
        "graph_components": float("nan"),
        "graph_largest_component_frac": float("nan"),
        "graph_singleton_frac": float("nan"),
    }
    if points.shape[0] < 2:
        return out
    tree = cKDTree(points)
    dists, _ = tree.query(points, k=2)
    radius_um = max(float(np.mean(dists[:, 1])) * 1.5, 1.0)
    neighbor_lists = tree.query_ball_point(points, r=radius_um)
    degrees = np.array([max(0, len(neigh) - 1) for neigh in neighbor_lists], dtype=np.float64)
    rows = []
    cols = []
    for i, neigh in enumerate(neighbor_lists):
        for j in neigh:
            if j > i:
                rows.append(i)
                cols.append(j)
                rows.append(j)
                cols.append(i)
    out["graph_radius_um"] = float(radius_um)
    out["graph_mean_degree"] = float(np.mean(degrees))
    out["graph_sd_degree"] = float(np.std(degrees, ddof=1)) if degrees.size > 1 else 0.0
    if rows:
        data = np.ones(len(rows), dtype=np.float64)
        graph = coo_matrix((data, (rows, cols)), shape=(points.shape[0], points.shape[0]))
        n_components, labels = connected_components(graph, directed=False, return_labels=True)
        counts = np.bincount(labels)
        out["graph_components"] = float(n_components)
        out["graph_largest_component_frac"] = float(np.max(counts) / points.shape[0])
        out["graph_singleton_frac"] = float(np.sum(counts == 1) / points.shape[0])
    else:
        out["graph_components"] = float(points.shape[0])
        out["graph_largest_component_frac"] = 1.0 / float(points.shape[0])
        out["graph_singleton_frac"] = 1.0
    return out


def _mst_lengths(n_points: int, edges, edge_lengths: np.ndarray) -> np.ndarray:
    if not edges:
        return np.array([], dtype=np.float64)
    rows = []
    cols = []
    data = []
    for (i, j), w in zip(edges, edge_lengths):
        rows.extend([i, j])
        cols.extend([j, i])
        data.extend([float(w), float(w)])
    graph = coo_matrix((np.asarray(data), (rows, cols)), shape=(n_points, n_points))
    mst = minimum_spanning_tree(graph)
    vals = np.asarray(mst.data, dtype=np.float64)
    return vals[np.isfinite(vals)]
