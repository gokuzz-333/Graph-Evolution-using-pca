# Results file guide

Run `python run_project.py` from the project folder. The pipeline creates this layout on every run. `run_metadata.json` remains at the top level and records the input file, parameters, versions, timing, and run summary.

## `01_temporal_graph/` — processed graph data

| File | Contents |
|---|---|
| `node_id_to_matrix_index.csv` | Maps each original anonymized node ID to the stable row/column index used in matrices. |
| `snapshot_graph_statistics.csv` | One row per time window: node/interaction/edge counts, density, matrix nonzero counts, and window boundaries. |
| `snapshot_graph_features.csv` | One row per time window: graph-level features such as edge count, density, degrees, active senders/receivers, and clustering. |
| `global_adjacency_matrix.npz` | Sparse adjacency matrix combined across snapshots. Rows are senders; columns are receivers. |
| `global_directed_out_laplacian_matrix.npz` | Sparse directed outgoing Laplacian, `L_out = D_out - A`. |
| `nodes_in_visualization_subgraph.csv` | Original IDs and matrix indices for nodes shown in the small network illustration. |

## `02_feature_pca/` — PCA on extracted graph features

### `full_node_temporal/`

- `temporal_node_degree_features.csv`: one row per node; in-degree and out-degree per snapshot plus their two temporal means.
- `full_temporal_feature_covariance_matrix.csv`: 12 × 12 sample covariance matrix of the standardized temporal features. Feature names label both axes.
- `full_temporal_pca_scores.csv`: each node's PC coordinates, original ID, and mean degrees.
- `full_temporal_pca_variance.csv`: eigenvalues and explained-variance ratios for the displayed PCs.
- `full_temporal_pca_loadings.csv`: feature weights for PC1 and PC2 (loadings, not node coordinates).

### `matched_sample/`

- `matched_sample_classical_pca_scores.csv`: classical PCA coordinates for the selected active-node sample used in the quantum comparison.

### `snapshot_level/`

- `snapshot_level_pca_scores.csv`: PCA coordinates for graph-level observations, one per snapshot.
- `snapshot_level_pca_variance.csv`: explained-variance ratios for this snapshot-level experiment.

## `03_quantum_kernel_pca/` — simulated quantum-kernel experiment

When Qiskit is available and the quantum branch is not skipped, this contains:

- `quantum_kernel_pca_scores.csv`: QPC coordinates for the sampled nodes.
- `quantum_kernel_pca_eigenvalues.csv`: centred fidelity-kernel eigenvalues and their shares of the eigenvalue sum.
- `quantum_fidelity_kernel_matrix.csv`: pairwise state-fidelity similarities for the sample.

## `04_graph_matrix_pca/` — PCA directly on graph matrices

- `adjacency_matrix_pca_scores.csv`, `laplacian_matrix_pca_scores.csv`, and `combined_matrix_pca_scores.csv`: one row per node with its retained PC scores.
- Matching `adjacency_matrix_pca_components.csv`, `laplacian_matrix_pca_components.csv`, and `combined_matrix_pca_components.csv`: component directions over matrix columns.
- `graph_matrix_pca_metrics.csv`: one row per representation, with clearly separated PC eigenvalues, explained-variance ratios, cumulative variance, reconstruction error, dimensions, and runtime.
- `graph_matrix_reconstruction_sweep.csv`: retained variance and reconstruction error for each tested component count.

## `05_classical_quantum_comparison/`

- `matched_classical_vs_quantum_scores.csv`: aligned classical and quantum-kernel coordinates for the same sampled nodes, plus their input degree features.

Plots remain in the sibling `figures/` folder. The PowerPoint is kept separately in `presentation_output/`.
