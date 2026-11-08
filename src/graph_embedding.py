"""PCA scores for full-node adjacency and directed-Laplacian matrices."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from .pca_engine import CustomPCA


@dataclass
class GraphPCAResult:
    scores: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray
    explained_variance_ratio: np.ndarray
    cumulative_explained_variance: float
    reconstruction_error: float
    runtime_seconds: float


def _frobenius_norm(matrix):
    """Return the square-root sum of squared entries for a dense or sparse matrix."""
    if sparse.issparse(matrix):
        return float(np.sqrt(matrix.multiply(matrix).sum()))
    return float(np.linalg.norm(matrix, ord="fro"))


def fit_graph_pca(matrix, n_components=2):
    """Run centered PCA with nodes as rows and matrix columns as graph features.

    PCA centers every feature column before fitting. The full 986-node matrices
    are small enough to convert from sparse storage to a dense array here.
    Features are not standardized, preserving the numeric scale of A or L_out.
    """
    matrix = sparse.csr_matrix(matrix, dtype=float)
    if matrix.shape[0] < 2 or matrix.shape[1] < n_components:
        raise ValueError("PCA needs at least two rows and enough feature columns.")
    if matrix.nnz and not np.isfinite(matrix.data).all():
        raise ValueError("Graph matrix entries must be finite.")

    start = time.perf_counter()
    values = matrix.toarray()
    model = CustomPCA(n_components=n_components).fit(values)
    scores = model.transform(values)
    runtime = time.perf_counter() - start

    # Compare reconstruction with the centered input PCA actually analyzes.
    centered = values - model.mean_
    reconstructed = scores @ model.components_
    centered_norm = float(np.linalg.norm(centered, ord="fro"))
    residual_norm = float(np.linalg.norm(centered - reconstructed, ord="fro"))
    reconstruction_error = residual_norm / centered_norm if centered_norm else 0.0

    return GraphPCAResult(
        scores, model.components_, model.explained_variance_,
        model.explained_variance_ratio_,
        float(model.explained_variance_ratio_.sum()), reconstruction_error, runtime)


def compare_graph_representations(adjacency, laplacian, n_components=2):
    """Apply PCA separately to A, L_out and their combined node-feature matrix."""
    adjacency = sparse.csr_matrix(adjacency, dtype=float)
    laplacian = sparse.csr_matrix(laplacian, dtype=float)
    if adjacency.shape != laplacian.shape or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency and Laplacian must have matching square shapes.")
    if adjacency.nnz and (adjacency.data < 0).any():
        raise ValueError("Adjacency entries must be non-negative.")
    a_norm = _frobenius_norm(adjacency)
    l_norm = _frobenius_norm(laplacian)
    if a_norm == 0 or l_norm == 0:
        raise ValueError("Cannot compare graph matrices when either matrix is empty.")

    # Each row represents a node; columns describe links to all nodes.
    # Normalize each block first so A and L_out have equal total input energy.
    combined = sparse.hstack((adjacency / a_norm, laplacian / l_norm), format="csr")
    matrices = {"adjacency": adjacency, "laplacian": laplacian, "combined": combined}
    results = {name: fit_graph_pca(mat, n_components) for name, mat in matrices.items()}
    rows = []
    for name, result in results.items():
        row = {
            "representation": name,
            "number_of_components": n_components,
            "cumulative_explained_variance": result.cumulative_explained_variance,
            "relative_reconstruction_error": result.reconstruction_error,
            "runtime_seconds": result.runtime_seconds,
            "matrix_rows": matrices[name].shape[0],
            "matrix_columns": matrices[name].shape[1],
            "matrix_nnz": matrices[name].nnz,
        }
        for component_index, (eigenvalue, ratio) in enumerate(
                zip(result.explained_variance, result.explained_variance_ratio), start=1):
            row[f"PC{component_index}_eigenvalue"] = float(eigenvalue)
            row[f"PC{component_index}_explained_variance_ratio"] = float(ratio)
            row[f"PC{component_index}_cumulative_variance_ratio"] = float(
                result.explained_variance_ratio[:component_index].sum())
        rows.append(row)
    return results, pd.DataFrame(rows)


def sweep_graph_reconstruction(adjacency, laplacian, component_counts):
    """Measure variance retained and reconstruction error for several PCA sizes.

    Fit each representation once at the largest requested size, then reuse its
    leading components for every smaller size. This keeps the sweep consistent
    and avoids repeating the expensive eigen-decomposition for each k.
    """
    adjacency = sparse.csr_matrix(adjacency, dtype=float)
    laplacian = sparse.csr_matrix(laplacian, dtype=float)
    if adjacency.shape != laplacian.shape or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("Adjacency and Laplacian must have matching square shapes.")
    a_norm = _frobenius_norm(adjacency)
    l_norm = _frobenius_norm(laplacian)
    if a_norm == 0 or l_norm == 0:
        raise ValueError("Cannot sweep graph matrices when either matrix is empty.")

    counts = sorted(set(int(k) for k in component_counts))
    if not counts or counts[0] < 1:
        raise ValueError("component_counts must contain positive integers.")

    combined = sparse.hstack((adjacency / a_norm, laplacian / l_norm), format="csr")
    matrices = {"adjacency": adjacency, "laplacian": laplacian, "combined": combined}
    rows = []
    for representation, matrix in matrices.items():
        maximum = min(matrix.shape)
        if counts[-1] > maximum:
            raise ValueError(f"Requested {counts[-1]} components for {representation}, "
                             f"but the input supports at most {maximum}.")

        values = matrix.toarray()
        model = CustomPCA(n_components=counts[-1]).fit(values)
        scores = model.transform(values)
        centered = values - model.mean_
        centered_norm = float(np.linalg.norm(centered, ord="fro"))

        for count in counts:
            reconstructed = scores[:, :count] @ model.components_[:count]
            residual_norm = float(np.linalg.norm(centered - reconstructed, ord="fro"))
            error = residual_norm / centered_norm if centered_norm else 0.0
            retained = float(model.explained_variance_ratio_[:count].sum())
            rows.append({
                "representation": representation,
                "n_components": count,
                "cumulative_explained_variance_ratio": retained,
                "relative_reconstruction_error": error,
            })
    return pd.DataFrame(rows)
