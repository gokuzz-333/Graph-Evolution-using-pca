"""Feature extraction, preprocessing, sampling and classical PCA."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import networkx as nx
from .pca_engine import CustomPCA, StandardScaler

SUMMARY_FEATURES = ["mean_out_degree", "mean_in_degree"]


def _degrees(A):
    return np.asarray(A.sum(axis=1)).ravel(), np.asarray(A.sum(axis=0)).ravel()


def snapshot_features(snapshots, node_count: int) -> pd.DataFrame:
    """One row per snapshot with graph-level features.

    ``edge_count`` = distinct directed edges (nnz of A). In ``binary`` mode degrees
    count distinct neighbours; in ``count`` mode they count emails.
    """
    if node_count < 2:
        raise ValueError("Need at least 2 nodes.")
    possible = node_count * (node_count - 1)
    rows = []
    for s in snapshots:
        if s.A.shape != (node_count, node_count):
            raise ValueError(f"Snapshot {s.label} has shape {s.A.shape}, expected {(node_count,) * 2}.")
        out_deg, in_deg = _degrees(s.A)
        directed_graph = nx.from_scipy_sparse_array(s.A, create_using=nx.DiGraph)
        # NetworkX uses the directed Fagiolo clustering definition here.
        directed_clustering = (float(nx.average_clustering(directed_graph))
                               if s.A.nnz else 0.0)
        rows.append({
            "snapshot": s.label, "start_day": s.start_day,
            "edge_count": s.n_edges, "density": s.n_edges / possible,
            "active_senders": int(np.count_nonzero(out_deg)),
            "active_receivers": int(np.count_nonzero(in_deg)),
            "mean_out_degree": float(out_deg.mean()), "mean_in_degree": float(in_deg.mean()),
            "max_out_degree": float(out_deg.max()), "max_in_degree": float(in_deg.max()),
            "isolated_nodes": int(np.count_nonzero((out_deg == 0) & (in_deg == 0))),
            "directed_average_clustering_fagiolo": directed_clustering,
        })
    return pd.DataFrame(rows)


def node_feature_matrix(snapshots, node_ids: np.ndarray):
    """Node x feature matrix.

    Columns: ``<label>_out_degree, <label>_in_degree`` for every snapshot, followed by
    two timeline summaries ``mean_out_degree`` and ``mean_in_degree`` (mean over ALL
    snapshots, empty ones included). Row ``i`` belongs to dataset node ``node_ids[i]``.
    """
    n = len(node_ids)
    if not snapshots:
        raise ValueError("No snapshots given.")
    cols, names = [], []
    out_sum, in_sum = np.zeros(n), np.zeros(n)
    for s in snapshots:
        if s.A.shape != (n, n):
            raise ValueError(f"Snapshot {s.label} shape {s.A.shape} does not match {n} nodes.")
        out_deg, in_deg = _degrees(s.A)
        cols += [out_deg, in_deg]
        names += [f"{s.label}_out_degree", f"{s.label}_in_degree"]
        out_sum += out_deg
        in_sum += in_deg
    cols += [out_sum / len(snapshots), in_sum / len(snapshots)]
    names += SUMMARY_FEATURES
    X = np.column_stack(cols)
    if not np.isfinite(X).all():
        raise ValueError("Node feature matrix contains non-finite values.")
    return X, names


def node_feature_frame(X: np.ndarray, names: list[str], node_ids: np.ndarray) -> pd.DataFrame:
    df = pd.DataFrame(X, columns=names)
    df.insert(0, "node_index", np.arange(len(node_ids)))
    df.insert(1, "node_id", node_ids)
    return df


def summary_feature_matrix(X: np.ndarray, names: list[str]) -> np.ndarray:
    """The two timeline-summary columns (mean out-degree, mean in-degree)."""
    return X[:, [names.index(c) for c in SUMMARY_FEATURES]]


def preprocess_summary_features(X2: np.ndarray, log_transform: bool = True) -> np.ndarray:
    """Shared first step for the fair comparison: optional ``log1p`` (degrees are heavy tailed)."""
    X2 = np.asarray(X2, dtype=float)
    if X2.ndim != 2 or X2.shape[1] != 2:
        raise ValueError(f"Expected an (m, 2) feature array, got {X2.shape}.")
    if not np.isfinite(X2).all():
        raise ValueError("Features contain non-finite values.")
    if (X2 < 0).any():
        raise ValueError("Degree features must be non-negative.")
    return np.log1p(X2) if log_transform else X2


def select_sample(X2: np.ndarray, sample_size: int, seed: int) -> np.ndarray:
    """Reproducible random sample (sorted row indices) from nodes with any activity."""
    if sample_size < 3:
        raise ValueError("sample_size must be at least 3.")
    eligible = np.flatnonzero(X2.sum(axis=1) > 0)
    if len(eligible) < 3:
        raise ValueError("Fewer than 3 active nodes; cannot draw a sample.")
    k = min(sample_size, len(eligible))
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(eligible, size=k, replace=False))


@dataclass
class PCAResult:
    scores: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    components: np.ndarray          # (n_components, n_features) loadings
    feature_names: list[str]
    covariance_matrix: np.ndarray
    covariance_matrix_type: str


def run_classical_pca(X: np.ndarray, n_components: int = 2, feature_names=None) -> PCAResult:
    """Standardize columns, then run the explicit covariance/eigen PCA steps."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError("X must be 2-D.")
    if not np.isfinite(X).all():
        raise ValueError("X contains non-finite values.")
    if X.shape[0] < 2 or X.shape[1] < 1:
        raise ValueError("Not enough data for PCA (need >=2 rows and >=1 column).")
    k = min(n_components, X.shape[0], X.shape[1])
    if k < n_components:
        raise ValueError(f"Requested {n_components} components but data only supports {k}.")
    Z = StandardScaler().fit_transform(X)       # constant columns become all zeros
    model = CustomPCA(n_components=k).fit(Z)
    scores = model.transform(Z)
    names = list(feature_names) if feature_names is not None else [f"f{i}" for i in range(X.shape[1])]
    return PCAResult(scores, model.explained_variance_, model.explained_variance_ratio_,
                     model.components_, names, model.covariance_matrix_,
                     model.eigen_matrix_type_)
