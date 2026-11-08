"""End-to-end pipeline: data -> snapshots -> features -> PCA -> quantum-kernel PCA -> outputs."""
from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import spearmanr

from . import features as F
from . import temporal_graph as T
from . import visualization as V
from . import graph_embedding as G
from . import project_config as C
from .quantum_kpca import QiskitNotAvailableError, run_quantum_kernel_pca, INSTALL_HINT

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class RunConfig:
    interval: str = C.SNAPSHOT_MODE
    snapshot_count: int = C.SNAPSHOT_COUNT
    snapshot_boundaries_days: list[float] | None = C.SNAPSHOT_BOUNDARIES_DAYS
    edge_mode: str = C.EDGE_MODE
    data_file: str | None = None          # None -> download SNAP file to data/
    output_dir: str = str(ROOT / C.OUTPUT_DIR)  # results/ and figures/ are created inside
    skip_quantum: bool = False
    quantum_samples: int = C.QUANTUM_SAMPLES
    seed: int = C.RANDOM_SEED
    log_transform: bool = C.LOG_TRANSFORM
    graph_representation: str = C.GRAPH_REPRESENTATION
    n_components: int = C.N_COMPONENTS
    graph_component_sweep: tuple[int, ...] = C.GRAPH_COMPONENT_SWEEP
    quantum_reps: int = C.QUANTUM_REPS
    quantum_entanglement: str = C.QUANTUM_ENTANGLEMENT
    subset_nodes: int = C.VISUALIZATION_SUBSET_NODES
    adjacency_nodes: int = C.ADJACENCY_VISUALIZATION_NODES


def validate_config(cfg: RunConfig) -> None:
    if cfg.interval not in (*T.INTERVAL_DAYS, T.EQUAL_INTERVAL):
        raise ValueError(f"interval must be one of {(*T.INTERVAL_DAYS, T.EQUAL_INTERVAL)}")
    if cfg.snapshot_count < 1 or cfg.n_components < 2:
        raise ValueError("snapshot_count must be positive and n_components must be at least 2")
    if cfg.graph_representation not in {"adjacency", "laplacian", "combined"}:
        raise ValueError("graph_representation must be adjacency, laplacian, or combined")
    if not cfg.graph_component_sweep or any(k < 1 for k in cfg.graph_component_sweep):
        raise ValueError("graph_component_sweep must contain positive component counts")
    if cfg.edge_mode not in T.EDGE_MODES:
        raise ValueError(f"edge_mode must be one of {T.EDGE_MODES}")
    if cfg.quantum_samples < 3:
        raise ValueError("quantum_samples must be >= 3")
    if cfg.subset_nodes < 3 or cfg.adjacency_nodes < 3:
        raise ValueError("subset sizes must be >= 3")
    if cfg.subset_nodes > 30:
        raise ValueError("subset_nodes > 30 makes the graph figure unreadable")


def compare_embeddings(classical: np.ndarray, quantum: np.ndarray) -> dict:
    """Rotation/sign-invariant similarity between two 2-D embeddings of the same nodes.

    * ``distance_rank_correlation``: Spearman correlation of all pairwise distances.
    * ``abs_spearman_pc1`` / ``pc2``: |Spearman| of matching components (signs are arbitrary).
    """
    def pdist(S):
        d = np.linalg.norm(S[:, None, :] - S[None, :, :], axis=-1)
        return d[np.triu_indices(len(S), 1)]
    return {
        "distance_rank_correlation": float(spearmanr(pdist(classical), pdist(quantum))[0]),
        "abs_spearman_pc1": float(abs(spearmanr(classical[:, 0], quantum[:, 0])[0])),
        "abs_spearman_pc2": float(abs(spearmanr(classical[:, 1], quantum[:, 1])[0])),
    }


def _print_pca_results(title: str, eigenvalues, variance_ratios,
                      reconstruction_error: float | None = None,
                      component_prefix: str = "PC",
                      omit_negligible: bool = False) -> None:
    """Print one PCA run's component results in a readable, consistent format."""
    print(f"\n{title}")
    print("-" * len(title))
    eigenvalues = np.asarray(eigenvalues, dtype=float)
    variance_ratios = np.asarray(variance_ratios, dtype=float)
    threshold = (float(np.max(np.abs(eigenvalues))) * 1e-8
                 if omit_negligible and eigenvalues.size else 0.0)
    cumulative = 0.0
    printed = 0
    for index, (eigenvalue, ratio) in enumerate(zip(eigenvalues, variance_ratios), start=1):
        if omit_negligible and eigenvalue <= threshold:
            continue
        cumulative += float(ratio)
        printed += 1
        print(f"{component_prefix}{index}: eigenvalue={float(eigenvalue):.6f} | "
              f"explained={float(ratio) * 100:.2f}% | "
              f"cumulative={cumulative * 100:.2f}%")
    if omit_negligible:
        omitted = len(eigenvalues) - printed
        print(f"Omitted {omitted} zero/negligible components "
              f"(eigenvalue <= {threshold:.2e}).")
    if reconstruction_error is not None:
        print(f"Relative reconstruction error: {reconstruction_error:.4f} "
              f"({reconstruction_error * 100:.2f}%)")


def run_pipeline(cfg: RunConfig) -> dict:
    validate_config(cfg)
    out = Path(cfg.output_dir)
    results, figures = out / "results", out / "figures"
    temporal_dir = results / "01_temporal_graph"
    feature_pca_dir = results / "02_feature_pca"
    full_pca_dir = feature_pca_dir / "full_node_temporal"
    sample_pca_dir = feature_pca_dir / "matched_sample"
    snapshot_pca_dir = feature_pca_dir / "snapshot_level"
    quantum_dir = results / "03_quantum_kernel_pca"
    matrix_pca_dir = results / "04_graph_matrix_pca"
    comparison_dir = results / "05_classical_quantum_comparison"
    for directory in (results, figures, temporal_dir, full_pca_dir, sample_pca_dir,
                      snapshot_pca_dir, quantum_dir, matrix_pca_dir, comparison_dir):
        directory.mkdir(parents=True, exist_ok=True)

    # 1. Data ------------------------------------------------------------------ #
    if cfg.data_file is None:
        path = T.download_dataset(ROOT / "data" / "email-Eu-core-temporal.txt.gz")
    else:
        path = Path(cfg.data_file)
    df = T.read_interactions(path)
    snapshots, mapping = T.build_snapshots(df, cfg.interval, cfg.edge_mode,
                                           snapshot_count=cfg.snapshot_count,
                                           snapshot_boundaries_days=cfg.snapshot_boundaries_days)
    node_ids = T.node_ids_from_mapping(mapping)
    n = len(node_ids)
    n_self = int((df["sender"] == df["receiver"]).sum())
    print(f"Interactions: {len(df):,} (self-loops dropped: {n_self:,}) | nodes: {n} | "
          f"snapshots: {len(snapshots)} | {cfg.interval}/{cfg.edge_mode}")

    pd.DataFrame({"node_index": np.arange(n), "node_id": node_ids}).to_csv(
        temporal_dir / "node_id_to_matrix_index.csv", index=False)
    summary = T.summarize_snapshots(snapshots, n)
    summary.to_csv(temporal_dir / "snapshot_graph_statistics.csv", index=False)
    feat = F.snapshot_features(snapshots, n)
    feat.to_csv(temporal_dir / "snapshot_graph_features.csv", index=False)

    # Graph-level temporal PCA describes the snapshot feature table (one row per snapshot).
    graph_feature_columns = [c for c in feat.select_dtypes(include=[np.number]).columns
                             if c != "start_day"]
    graph_pca_start = time.perf_counter()
    graph_pca = F.run_classical_pca(
        feat[graph_feature_columns].to_numpy(),
        min(cfg.n_components, len(feat), len(graph_feature_columns)),
        graph_feature_columns)
    graph_pca_runtime = time.perf_counter() - graph_pca_start
    pd.DataFrame({"snapshot": feat["snapshot"], "PC1": graph_pca.scores[:, 0],
                  "PC2": graph_pca.scores[:, 1]}).to_csv(
                      snapshot_pca_dir / "snapshot_level_pca_scores.csv", index=False)
    pd.DataFrame({"component": [f"PC{i}" for i in range(1, len(graph_pca.explained_variance) + 1)],
                  "eigenvalue": graph_pca.explained_variance,
                  "explained_variance_ratio": graph_pca.explained_variance_ratio,
                  "cumulative_explained_variance_ratio": np.cumsum(
                      graph_pca.explained_variance_ratio)}).to_csv(
                      snapshot_pca_dir / "snapshot_level_pca_variance.csv", index=False)

    # Use the global union graph, while keeping all rows in the dataset node order.
    global_A = snapshots[0].A.copy().tocsr()
    for snap in snapshots[1:]:
        global_A = global_A + snap.A
    global_A = global_A.tocsr()
    global_A.sum_duplicates()
    if cfg.edge_mode == "binary":
        global_A.data[:] = 1.0
    global_A.eliminate_zeros()
    global_L = T.directed_out_laplacian(global_A)
    sparse.save_npz(temporal_dir / "global_adjacency_matrix.npz", global_A)
    sparse.save_npz(temporal_dir / "global_directed_out_laplacian_matrix.npz", global_L)
    graph_results, graph_metrics = G.compare_graph_representations(
        global_A, global_L, cfg.n_components)
    graph_metrics.to_csv(matrix_pca_dir / "graph_matrix_pca_metrics.csv", index=False)
    graph_sweep = G.sweep_graph_reconstruction(
        global_A, global_L, cfg.graph_component_sweep)
    graph_sweep.to_csv(matrix_pca_dir / "graph_matrix_reconstruction_sweep.csv", index=False)
    for representation, embedding in graph_results.items():
        score_frame = pd.DataFrame({"node_index": np.arange(n), "node_id": node_ids})
        for j in range(embedding.scores.shape[1]):
            score_frame[f"PC{j + 1}_score"] = embedding.scores[:, j]
        score_frame.to_csv(matrix_pca_dir / f"{representation}_matrix_pca_scores.csv", index=False)
        if representation == "combined":
            feature_blocks = ["normalized_adjacency"] * n + ["normalized_directed_out_laplacian"] * n
            feature_node_ids = np.concatenate((node_ids, node_ids))
            feature_indices = np.concatenate((np.arange(n), np.arange(n)))
        else:
            feature_blocks = [representation] * n
            feature_node_ids = node_ids
            feature_indices = np.arange(n)
        component_frame = pd.DataFrame({
            "feature_block": feature_blocks,
            "feature_index": feature_indices,
            "feature_node_id": feature_node_ids,
        })
        for j in range(embedding.components.shape[0]):
            component_frame[f"PC{j + 1}_loading"] = embedding.components[j]
        component_frame.to_csv(
            matrix_pca_dir / f"{representation}_matrix_pca_components.csv", index=False)

    # 2. Node features + full-data classical PCA ------------------------------- #
    X, names = F.node_feature_matrix(snapshots, node_ids)
    F.node_feature_frame(X, names, node_ids).to_csv(
        full_pca_dir / "temporal_node_degree_features.csv", index=False)
    pca_start = time.perf_counter()
    full = F.run_classical_pca(X, cfg.n_components, names)
    full_pca_runtime = time.perf_counter() - pca_start
    if full.covariance_matrix_type != "feature_covariance":
        raise RuntimeError("Full temporal-feature PCA should produce a feature covariance matrix.")
    covariance_frame = pd.DataFrame(full.covariance_matrix, index=names, columns=names)
    covariance_frame.to_csv(
        full_pca_dir / "full_temporal_feature_covariance_matrix.csv", index_label="feature")
    X2_raw = F.summary_feature_matrix(X, names)
    full_score_columns = {
        f"PC{i + 1}": full.scores[:, i]
        for i in range(full.scores.shape[1])
    }
    full_df = pd.DataFrame({"node_index": np.arange(n), "node_id": node_ids,
                            **full_score_columns,
                            "mean_out_degree": X2_raw[:, 0], "mean_in_degree": X2_raw[:, 1]})
    full_df.to_csv(full_pca_dir / "full_temporal_pca_scores.csv", index=False)
    pd.DataFrame({"component": [f"PC{i}" for i in range(1, len(full.explained_variance) + 1)],
                  "eigenvalue": full.explained_variance,
                  "explained_variance_ratio": full.explained_variance_ratio,
                  "cumulative_explained_variance_ratio": np.cumsum(
                      full.explained_variance_ratio)}).to_csv(
        full_pca_dir / "full_temporal_pca_variance.csv", index=False)
    loading_columns = [f"PC{i + 1}_loading" for i in range(full.components.shape[0])]
    load = pd.DataFrame(full.components.T, columns=loading_columns)
    load.insert(0, "feature", names)
    load.to_csv(full_pca_dir / "full_temporal_pca_loadings.csv", index=False)

    # 3. Sample-matched comparison data ---------------------------------------- #
    X2 = F.preprocess_summary_features(X2_raw, cfg.log_transform)
    idx = F.select_sample(X2_raw, cfg.quantum_samples, cfg.seed)
    X2s = X2[idx]
    pca_start = time.perf_counter()
    # This branch has only two input features, so it cannot produce more than
    # two ordinary PCA components even when the full-data run requests more.
    sample_components = min(cfg.n_components, X2s.shape[1])
    sample_cls = F.run_classical_pca(X2s, sample_components, F.SUMMARY_FEATURES)
    sample_pca_runtime = time.perf_counter() - pca_start
    color = np.log1p(X2_raw[idx].sum(axis=1))

    # 4. Quantum-kernel PCA ------------------------------------------------------ #
    quantum, q_status, metrics = None, "completed", {}
    quantum_runtime = None
    if cfg.skip_quantum:
        q_status = "skipped (--skip-quantum)"
    else:
        try:
            quantum_start = time.perf_counter()
            quantum = run_quantum_kernel_pca(X2s, cfg.n_components, cfg.quantum_reps,
                                             cfg.quantum_entanglement)
            quantum_runtime = time.perf_counter() - quantum_start
        except QiskitNotAvailableError as exc:
            q_status = f"skipped: {exc}"
            print(f"\nQuantum branch skipped: {exc}\n")
    base = pd.DataFrame({"node_index": idx, "node_id": node_ids[idx],
                         "mean_out_degree": X2_raw[idx, 0], "mean_in_degree": X2_raw[idx, 1]})
    cmp_df = base.assign(classical_PC1=sample_cls.scores[:, 0], classical_PC2=sample_cls.scores[:, 1])
    cmp_df.to_csv(sample_pca_dir / "matched_sample_classical_pca_scores.csv", index=False)
    if quantum is not None:
        qdf = base.assign(QPC1=quantum.scores[:, 0], QPC2=quantum.scores[:, 1])
        qdf.to_csv(quantum_dir / "quantum_kernel_pca_scores.csv", index=False)
        cmp_df = cmp_df.assign(quantum_QPC1=quantum.scores[:, 0], quantum_QPC2=quantum.scores[:, 1])
        cmp_df.to_csv(comparison_dir / "matched_classical_vs_quantum_scores.csv", index=False)
        quantum_ratios = quantum.explained_fraction
        pd.DataFrame({"component": [f"QPC{i}" for i in range(1, len(quantum.eigenvalues) + 1)],
                      "eigenvalue": quantum.eigenvalues,
                      "fraction_of_eigenvalue_sum": quantum_ratios,
                      "cumulative_fraction_of_eigenvalue_sum": np.cumsum(quantum_ratios)}).to_csv(
            quantum_dir / "quantum_kernel_pca_eigenvalues.csv", index=False)
        kernel_frame = pd.DataFrame(
            quantum.kernel,
            index=pd.Index(node_ids[idx], name="row_node_id"),
            columns=[f"node_id_{node_id}" for node_id in node_ids[idx]])
        kernel_frame.to_csv(
            quantum_dir / "quantum_fidelity_kernel_matrix.csv")
        metrics = compare_embeddings(sample_cls.scores, quantum.scores)
        metrics["quantum_kernel_pca_runtime_seconds"] = float(quantum_runtime)

    # 5. Figures ----------------------------------------------------------------- #
    iv, em = cfg.interval, cfg.edge_mode
    figs = {
        "01_edges_over_time.png": V.plot_edges(feat, figures / "01_edges_over_time.png", iv, em),
        "02_density_over_time.png": V.plot_density(feat, figures / "02_density_over_time.png", iv, em),
        "03_active_senders_receivers.png": V.plot_active_nodes(feat, figures / "03_active_senders_receivers.png", iv, em),
        "04_mean_degrees.png": V.plot_mean_degrees(feat, figures / "04_mean_degrees.png", iv, em),
        "12_clustering_coefficient.png": V.plot_clustering(
            feat, figures / "12_clustering_coefficient.png", iv, em),
    }
    for representation, embedding in graph_results.items():
        figs[f"{representation}_embedding.png"] = V.plot_graph_embedding(
            embedding.scores, node_ids,
            f"PCA scores from {representation.title()} matrix ({n} nodes)",
            figures / f"{representation}_embedding.png")
    figs["graph_matrix_reconstruction_sweep.png"] = V.plot_graph_reconstruction_sweep(
        graph_sweep, figures / "graph_matrix_reconstruction_sweep.png")
    figs["snapshot_feature_pca.png"] = V.plot_embedding(
        graph_pca.scores, "PC1 (snapshot-feature PCA)", "PC2 (snapshot-feature PCA)",
        f"Graph-level PCA across {len(snapshots)} snapshots",
        figures / "snapshot_feature_pca.png")
    cvar = full.explained_variance_ratio
    figs["05_classical_pca_full.png"] = V.plot_embedding(
        full.scores, f"PC1 ({cvar[0]*100:.1f}% of variance)", f"PC2 ({cvar[1]*100:.1f}% of variance)",
        f"Classical PCA, all {n} nodes (temporal degree features)", figures / "05_classical_pca_full.png",
        color=np.log1p(X2_raw.sum(axis=1)), color_label="log(1 + mean total degree)")
    sv = sample_cls.explained_variance_ratio
    figs["06_classical_pca_sample.png"] = V.plot_embedding(
        sample_cls.scores, f"PC1 ({sv[0]*100:.1f}% of variance)", f"PC2 ({sv[1]*100:.1f}% of variance)",
        f"Classical PCA, sample-matched ({len(idx)} nodes, 2 features)", figures / "06_classical_pca_sample.png",
        color=color, color_label="log(1 + mean total degree)")
    if quantum is not None:
        qf = quantum.explained_fraction
        figs["07_quantum_kernel_pca.png"] = V.plot_embedding(
            quantum.scores, f"Kernel-PC1 ({qf[0]*100:.1f}% of kernel eigenvalue sum)",
            f"Kernel-PC2 ({qf[1]*100:.1f}% of kernel eigenvalue sum)",
            f"Quantum-kernel PCA ({len(idx)} nodes, 2 qubits, simulated)", figures / "07_quantum_kernel_pca.png",
            color=color, color_label="log(1 + mean total degree)",
            annotate_note="Statevector simulation; not quantum hardware")
        figs["11_comparison_side_by_side.png"] = V.plot_side_by_side(
            sample_cls.scores, quantum.scores, sv, qf, color, figures / "11_comparison_side_by_side.png", len(idx))

    busiest = max(snapshots, key=lambda s: s.n_edges)
    sub_big = V.top_degree_subset(busiest.A, cfg.adjacency_nodes)
    sub_small = V.top_degree_subset(busiest.A, cfg.subset_nodes)
    A_big = busiest.A[sub_big][:, sub_big]
    A_small = busiest.A[sub_small][:, sub_small].tocsr()
    L_small = T.directed_out_laplacian(A_small)         # Laplacian of the induced subgraph
    T.validate_laplacian(L_small, A_small)
    note = f"snapshot {busiest.label} (busiest)"
    figs["08_adjacency_subset.png"] = V.plot_adjacency_subset(
        A_big, node_ids[sub_big], f"Adjacency matrix: top-{len(sub_big)} nodes by degree\n{note}",
        figures / "08_adjacency_subset.png", cfg.edge_mode)
    figs["09_directed_graph_subset.png"] = V.plot_directed_graph_subset(
        A_small, node_ids[sub_small], f"Directed graph: top-{len(sub_small)} nodes by degree\n{note}",
        figures / "09_directed_graph_subset.png")
    figs["10_laplacian_subset.png"] = V.plot_laplacian_subset(
        L_small.tocsr(), node_ids[sub_small],
        f"Outgoing Laplacian L_out = D_out - A of the induced subgraph\n{len(sub_small)} nodes, {note}",
        figures / "10_laplacian_subset.png")
    pd.DataFrame({"node_index": sub_small, "node_id": node_ids[sub_small]}).to_csv(
        temporal_dir / "nodes_in_visualization_subgraph.csv", index=False)

    # 6. Metadata ------------------------------------------------------------------ #
    try:
        import qiskit
        qv = qiskit.__version__
    except ImportError:
        qv = None
    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "data_file": str(path), "dataset_name": Path(path).name,
        "is_snap_dataset": Path(path).name.startswith("email-Eu-core-temporal"),
        "n_interactions": int(len(df)), "n_self_loops_dropped": n_self,
        "exact_duplicate_rows": int(df.duplicated(["sender", "receiver", "timestamp"]).sum()),
        "timestamp_unit": "seconds since collection start (relative, not Unix time)",
        "timestamp_min": int(df["timestamp"].min()), "timestamp_max": int(df["timestamp"].max()),
        "observed_time_span_days": float((df["timestamp"].max() - df["timestamp"].min()) /
                                         T.SECONDS_PER_DAY),
        "n_nodes": n, "n_snapshots": len(snapshots),
        "node_id_min": int(node_ids.min()), "node_id_max": int(node_ids.max()),
        "node_ids_contiguous": bool(int(node_ids.max()) - int(node_ids.min()) + 1 == n),
        "n_empty_snapshots": int(sum(s.n_interactions == 0 for s in snapshots)),
        "time_span_days": float(snapshots[-1].end_day),
        "window_days": (float(np.mean(np.diff(cfg.snapshot_boundaries_days)))
                        if cfg.snapshot_boundaries_days is not None else
                        (float(snapshots[-1].end_day / len(snapshots))
                         if cfg.interval == T.EQUAL_INTERVAL else T.INTERVAL_DAYS[cfg.interval])),
        "n_node_features": int(X.shape[1]),
        "config": asdict(cfg), "sample_size_used": int(len(idx)),
        "graph_matrix_nodes": n,
        "graph_matrix_representation": cfg.graph_representation,
        "graph_matrix_method": "centered PCA on full-node adjacency, directed Laplacian, and normalized combined matrices",
        "graph_matrix_files": [
            "01_temporal_graph/global_adjacency_matrix.npz",
            "01_temporal_graph/global_directed_out_laplacian_matrix.npz"],
        "organized_results": {
            "temporal_graph": "results/01_temporal_graph/",
            "feature_pca": "results/02_feature_pca/",
            "quantum_kernel_pca": "results/03_quantum_kernel_pca/",
            "graph_matrix_pca": "results/04_graph_matrix_pca/",
            "classical_quantum_comparison": "results/05_classical_quantum_comparison/",
            "run_metadata": "results/run_metadata.json"},
        "full_temporal_feature_covariance_matrix":
            "02_feature_pca/full_node_temporal/full_temporal_feature_covariance_matrix.csv",
        "graph_matrix_metrics": graph_metrics.to_dict(orient="records"),
        "graph_matrix_reconstruction_sweep": graph_sweep.to_dict(orient="records"),
        "graph_matrix_runtime_seconds": {
            name: float(result.runtime_seconds) for name, result in graph_results.items()},
        "graph_matrix_pca_explained_variance_ratio": {
            name: [float(v) for v in result.explained_variance_ratio]
            for name, result in graph_results.items()},
        "graph_matrix_pca_cumulative_explained_variance": {
            name: float(result.cumulative_explained_variance)
            for name, result in graph_results.items()},
        "graph_matrix_reconstruction_error": {
            name: float(result.reconstruction_error) for name, result in graph_results.items()},
        "graph_matrix_density": float(global_A.nnz / (n * (n - 1))),
        "graph_matrix_edges": int(global_A.nnz),
        "snapshot_feature_pca_variance_ratio": [float(v) for v in graph_pca.explained_variance_ratio],
        "quantum_runtime_seconds": quantum_runtime,
        "classical_full_node_temporal_feature_pca_runtime_seconds": float(full_pca_runtime),
        "classical_sample_matched_pca_runtime_seconds": float(sample_pca_runtime),
        "snapshot_feature_pca_runtime_seconds": float(graph_pca_runtime),
        "sample_selection": "uniform random without replacement among nodes with any activity; "
                            "np.random.default_rng(seed); indices sorted",
        "comparison_features": F.SUMMARY_FEATURES,
        "full_pca_variance_ratio": [float(v) for v in full.explained_variance_ratio],
        "sample_pca_variance_ratio": [float(v) for v in sample_cls.explained_variance_ratio],
        "quantum_status": q_status,
        "quantum_eigenvalue_fraction_top2": ([float(v) for v in quantum.explained_fraction[:2]]
                                             if quantum is not None else None),
        "comparison_metrics": metrics,
        "busiest_snapshot": busiest.label,
        "visualization_subset": "top-degree nodes in busiest snapshot (biased toward hubs)",
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "qiskit": qv},
        "install_hint_quantum": INSTALL_HINT,
        "figures": sorted(figs),
    }
    (results / "run_metadata.json").write_text(json.dumps(meta, indent=2))
    print("\nFINAL PCA RESULTS (printed in experiment order)")
    print("=" * 48)
    print("Full temporal-feature covariance matrix saved to: "
          f"{full_pca_dir / 'full_temporal_feature_covariance_matrix.csv'}")
    _print_pca_results("1. Snapshot-level graph-feature PCA",
                       graph_pca.explained_variance,
                       graph_pca.explained_variance_ratio)
    _print_pca_results("2. Full-node temporal-feature classical PCA",
                       full.explained_variance, full.explained_variance_ratio)
    _print_pca_results("3. Matched-sample classical PCA",
                       sample_cls.explained_variance,
                       sample_cls.explained_variance_ratio)
    if quantum is not None:
        _print_pca_results("4. Simulated quantum-kernel PCA (centered fidelity kernel)",
                           quantum.eigenvalues, quantum.explained_fraction,
                           component_prefix="KPC", omit_negligible=True)
        if metrics:
            print("Classical-versus-quantum comparison metrics:")
            for name, value in metrics.items():
                print(f"  {name}: {value:.4f}")
    else:
        print(f"\n4. Simulated quantum-kernel PCA: {q_status}")
    for number, representation in enumerate(("adjacency", "laplacian", "combined"), start=5):
        embedding = graph_results[representation]
        _print_pca_results(f"{number}. {representation.title()} matrix PCA",
                           embedding.explained_variance,
                           embedding.explained_variance_ratio,
                           embedding.reconstruction_error)
    print("\n8. Graph-matrix reconstruction sweep")
    print("Representation      PCs   Variance retained   Reconstruction error")
    print("-" * 70)
    for row in graph_sweep.itertuples(index=False):
        print(f"{row.representation:<19} {row.n_components:>3}   "
              f"{row.cumulative_explained_variance_ratio * 100:>7.2f}%             "
              f"{row.relative_reconstruction_error * 100:>7.2f}%")
    print(f"\nQuantum branch status: {q_status}")
    print(f"Outputs written to {results} and {figures}")
    return meta
