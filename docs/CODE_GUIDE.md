# Code guide

## Workflow

The pipeline reads the SNAP event file, validates rows, assigns stable indices to nodes, and splits event times into snapshots. Each snapshot produces a SciPy CSR directed adjacency matrix and outgoing Laplacian. The pipeline extracts graph statistics, creates three full-node graph-matrix embeddings, runs two separate classical PCA analyses, and optionally runs quantum-kernel PCA on a reproducible sample. CSV, JSON, and sparse matrix outputs go to results/; PNG plots go to figures/.

## Important files

- run_project.py: command-line entry point and run options.
- src/project_config.py: shared beginner-editable defaults.
- src/temporal_graph.py: parsing, global node mapping, snapshot boundaries, adjacency, and Laplacian checks.
- src/pca_engine.py: beginner-readable NumPy StandardScaler and PCA steps: center, covariance/Gram matrix, eigen-decomposition, component sorting, and projection.
- src/features.py: graph statistics, directed clustering, temporal degree features, sampling, and classical PCA using pca_engine.py.
- src/graph_embedding.py: centered custom PCA for adjacency, Laplacian, and combined matrices.
- src/quantum_kpca.py: Qiskit statevectors, fidelity kernel, kernel centering, and classical eigendecomposition.
- src/pipeline.py: connects methods, writes outputs, measures runtimes, and creates plots.
- src/visualization.py: all matplotlib plots. Some graph and matrix illustrations use labeled subsets only.
- make_presentation.py: builds the 15-slide review deck from the current CSV/JSON results and PNG figures. It writes `Hybrid_Classical_Quantum_PCA_Improved_Final.pptx` by default so earlier presentations remain intact.
- make_results_presentation.mjs: alternate 16-slide editable deck built with Codex's bundled Artifact Tool.
- tests/: focused tests for parsing, direction, boundaries, Laplacian, features, quantum kernel, and pipeline behavior.

## Main functions and equations

An event row has sender, receiver, timestamp. It creates A[i,j], where i is sender and j is receiver. Binary mode stores 1 if at least one interaction occurred for that ordered pair in a snapshot. Count mode stores the number of interaction rows. Exact duplicate rows remain repeated interactions; binary mode collapses them during graph construction. Self-loops do not enter A or L, but their nodes stay in the mapping.

Outgoing degree is the row sum of A. The outgoing directed Laplacian is L_out = diag(A @ 1) − A. Each Laplacian row sums to zero. Sparse CSR matrices and saved NPZ files preserve direction without dense graph-matrix allocations.

The global graph is the union of edges from every snapshot in binary mode, or the aggregate interaction weights in count mode. Each node is one row of A and L_out; each column is a connection to one node. graph_embedding.py centers columns across nodes and applies custom PCA without standardizing them. When the feature count is no greater than the node count, pca_engine.py diagonalizes the feature covariance matrix. For wider matrices, it diagonalizes the smaller node-by-node Gram matrix and recovers the equivalent feature directions. For the combined representation, it divides A and L_out by their own Frobenius norms, then joins the 986-column matrices side by side. Component scores and PCA directions are saved.

PCA explained-variance ratios measure how much variation in the centered node connection patterns is captured by each component. Relative reconstruction error compares the centered matrix with the low-rank reconstruction from the selected components.

features.py standardizes each input column, then pca_engine.py centers it, forms the covariance matrix, uses numpy.linalg.eigh to find directions and eigenvalues, sorts them from largest to smallest, and projects each node onto the selected directions. This describes which combinations of in-degree and out-degree histories vary across nodes. Snapshot-level PCA uses the same steps separately: each snapshot is one observation, with graph-level statistics as features. StandardScaler uses population standard deviation (ddof=0) to preserve the previous pipeline's preprocessing.

The quantum branch logs two mean-degree features and rescales them to [0, pi]. A two-qubit ZZFeatureMap creates statevectors. The fidelity kernel is K[i,j] = |<psi_i|psi_j>| squared. The code centers K and performs a classical eigen-decomposition. This is quantum-kernel PCA, not the density-matrix-exponentiation qPCA algorithm.

Directed average clustering uses NetworkX directed clustering, based on the Fagiolo directed-triangle coefficient. It averages local coefficients over every node. Nodes with no directed triangles count as zero; an empty snapshot is explicitly assigned zero.

## Install and run

In PowerShell from the project folder:

~~~~powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python run_project.py
python -m pytest -q
~~~~

The supplied dataset is already at data/email-Eu-core-temporal.txt.gz. If absent, the pipeline attempts to download it from SNAP. If that fails, download it manually and pass --data-file PATH.

The current deck builder uses the Codex Artifact Tool runtime:

~~~~powershell
New-Item -ItemType Directory -Force .codex_build | Out-Null
if (-not (Test-Path .codex_build\node_modules)) {
  New-Item -ItemType Junction .codex_build\node_modules -Target "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules" | Out-Null
}
Copy-Item make_results_presentation.mjs .codex_build\make_results_presentation.mjs -Force
$env:RUNTIME_NODE_MODULES = "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules"
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe" .codex_build\make_results_presentation.mjs
& "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe" .codex_build\finalize.mjs
~~~~

The finalizer checks the draft and saves results/Hybrid_Classical_Quantum_PCA_Results.pptx. It archives the previously generated deck under .codex_build/previous_decks/ when rebuilding.

## Change common settings

Edit src/project_config.py:

- Snapshot count: SNAPSHOT_COUNT. Default is 5.
- Snapshot mode: SNAPSHOT_MODE can be equal, week, or month. Week and month use fixed 7-day and 30-day intervals.
- Custom boundaries: SNAPSHOT_BOUNDARIES_DAYS can contain increasing elapsed-day values from the first event, for example [0, 100, 300, 600, 804]. The values must cover the last event.
- Edge handling: EDGE_MODE is binary or count.
- Highlighted graph-matrix method in metadata: GRAPH_REPRESENTATION can be adjacency, laplacian, or combined. All three always run.
- Component count: N_COMPONENTS. Two makes the standard plots; rerun tests and pipeline after changing it.
- Output folder: OUTPUT_DIR or the command-line --output-dir PATH.
- Optional weekly analysis: run python run_project.py --interval week.
- Quantum limits: QUANTUM_SAMPLES, QUANTUM_REPS, and QUANTUM_ENTANGLEMENT.
- Random seed: RANDOM_SEED controls reproducible sample selection. The custom covariance/eigen PCA is deterministic.

Example with custom boundaries:

~~~~powershell
python run_project.py --interval equal --snapshot-boundaries 0,100,300,600,804 --edge-mode binary
~~~~

## Outputs

`results/README.md` is the index for all saved analysis files. `results/run_metadata.json` records the run settings and overall status. Processed snapshots, graph statistics, node-ID mapping, and sparse matrices are grouped in `results/01_temporal_graph/`. Feature-PCA inputs and outputs are in `results/02_feature_pca/`, with separate folders for full-node, matched-sample, and snapshot-level experiments. The full-node temporal-feature folder includes the labelled 12 × 12 covariance matrix of the standardized features. Quantum outputs are in `results/03_quantum_kernel_pca/`; adjacency/Laplacian PCA scores, component directions, metrics, and the reconstruction sweep are in `results/04_graph_matrix_pca/`; aligned classical-versus-quantum scores are in `results/05_classical_quantum_comparison/`. Plot images remain in `figures/`.

## Rebuild the 15-slide review deck

After running the analysis pipeline, run `python make_presentation.py`. The generator reads current values from `results/run_metadata.json`, `results/01_temporal_graph/snapshot_graph_features.csv`, `results/02_feature_pca/full_node_temporal/full_temporal_pca_loadings.csv`, and `results/04_graph_matrix_pca/graph_matrix_pca_metrics.csv`. It uses the PNGs in `figures/` and stops with a clear error if a required figure is missing. To choose another output name, use `python make_presentation.py --out My_Review.pptx`. The original deck is never overwritten by default.

For edits, shared colors and font-size defaults are near the top of the script. The numbered blocks inside `main()` define slide content; `d.image(...)` calls control figure placement and the coordinate arguments are inches. Values shown on slides come from the saved run outputs, so edit labels or layout there without changing the analysis calculations.

## Safe code changes to practise for a viva

Run examples from the project folder in PowerShell. Each example uses a separate output folder so it does not overwrite the verified `results/` and `figures/`.

### Change weekly snapshots to fixed 30-day windows

- **Where:** `run_project.py`, option `--interval`; supported by `src/temporal_graph.py` through `INTERVAL_DAYS`.
- **Change:** replace `--interval week` with `--interval month`. These are fixed 7-day or 30-day windows measured from the first event, not calendar weeks/months.
- **Expected effect:** fewer, wider snapshots; each node’s per-window degree columns and graph-level trend rows will change. The number of windows is data-dependent for fixed-width intervals.
- **Try and check:**

~~~~powershell
python run_project.py --data-file data\email-Eu-core-temporal.txt.gz --interval month --edge-mode binary --skip-quantum --output-dir .\demo_monthly
~~~~

Check `demo_monthly\results\01_temporal_graph\snapshot_graph_statistics.csv` for 30-day boundaries and `snapshot_graph_features.csv` for per-window summaries. The `--skip-quantum` option keeps this quick and avoids spending time on statevector calculations.

### Switch binary edges to email-count weights

- **Where:** `run_project.py`, option `--edge-mode`; `src/temporal_graph.py:build_adjacency` implements the rule.
- **Change:** replace `--edge-mode binary` with `--edge-mode count`.
- **Expected effect:** `A[i,j]` becomes the number of email rows from i to j in a window, so degrees become email counts. Direction and sparse storage remain. `edge_count` still counts distinct nonzero ordered pairs.
- **Try and check:**

~~~~powershell
python run_project.py --data-file data\email-Eu-core-temporal.txt.gz --interval equal --snapshots 5 --edge-mode count --skip-quantum --output-dir .\demo_count
~~~~

Check `demo_count\results\01_temporal_graph\snapshot_graph_statistics.csv`, `snapshot_graph_features.csv`, and `demo_count\results\02_feature_pca\full_node_temporal\temporal_node_degree_features.csv`; run `python -m pytest tests\test_temporal_graph.py tests\test_pipeline.py -q` for the binary/count and pipeline checks.

### Change the number of PCA components

- **Safe direct experiment:** `src/features.py:run_classical_pca(X, n_components=...)` supports more components when the matrix has enough rows and columns. For example, a 20-by-5 test matrix can return three components. Verify the returned shapes: `scores` should be 20-by-3 and `components` should be 3-by-5.
- **Quick check:** `python -c "import numpy as np; from src.features import run_classical_pca; r=run_classical_pca(np.random.default_rng(7).normal(size=(20,5)), 3); print(r.scores.shape, r.components.shape)"` should print `(20, 3) (3, 5)`.
- **Integrated pipeline limit:** the current presentation and comparison pipeline intentionally exports and plots PC1/PC2, and the sample classical branch has only two input features. Changing `N_COMPONENTS` alone is not a safe way to request a three-axis project comparison. A full change would also need generalised CSV exports, plots, and the sample branch design. Keep the integrated run at two unless those outputs are updated together.
- **Check:** `python -m pytest tests\test_features.py -q` verifies the existing PCA validation and shape behavior. Do not remove centering, standardisation, or the orthogonality implied by PCA.

### Modify node features

- **Where:** `src/features.py:node_feature_matrix()` builds the full node-history matrix; `SUMMARY_FEATURES` and `summary_feature_matrix()` select the two sample-comparison features.
- **Safe example:** to remove T05 incoming degree from the *full node-history PCA only*, make a separate matrix in `src/pipeline.py` after the full matrix is built:

- **Implementation points:** create `full_columns`, `X_full`, and `full_names` from `X`/`names`; pass those to the full-data `F.run_classical_pca()` call. Use `full_names` when writing `full_temporal_pca_loadings.csv`, and add a `full_pca_feature_count` metadata value for the deck. Keep `X` and `names` unchanged for `F.summary_feature_matrix(X, names)`, so the sample-matched branch still receives its two expected summary features. The deck should use `full_pca_feature_count` rather than `n_node_features` for the full-PCA dimension.
- **Expected effect:** the full PCA scores, directions and variance ratios change because one temporal feature is omitted. Graph-level `edge_count` and `density` should not be inserted into this node-level matrix; they are separate snapshot summaries.
- **Check:** run `python -m pytest tests\test_features.py -q`, then rerun the pipeline to regenerate `results/02_feature_pca/full_node_temporal/temporal_node_degree_features.csv`, PCA outputs, and figures. To change quantum inputs, also update the exact two-feature check and feature-map qubit count in `src/quantum_kpca.py` and matching tests; do not just append a third column.

### Change the quantum circuit repetitions

- **Where:** `src/project_config.py:QUANTUM_REPS` (default `1`); passed by `src/pipeline.py` to `run_quantum_kernel_pca()` and then to Qiskit’s ZZ feature map.
- **Change:** set `QUANTUM_REPS = 2` for a small comparison.
- **Expected effect:** the feature-map circuit repeats its encoding/entangling layer; statevectors and fidelity similarities can change. It remains a classical statevector simulation.
- **Check:** run `python -m pytest tests\test_quantum.py -q`, then run `python run_project.py --data-file data\email-Eu-core-temporal.txt.gz --output-dir .\demo_reps` and confirm the completed status and updated config in `demo_reps\results\run_metadata.json`.

After a change, inspect the output metadata and CSVs before regenerating the presentation. Keep the original verified result folder intact until the changed run is understood.

## Common errors

- Missing data: place the SNAP .txt.gz in data/ or pass --data-file PATH.
- Missing Qiskit: install the optional dependency or pass --skip-quantum. The run records the skipped status.
- Invalid custom boundaries: values must increase, begin at or before elapsed day zero, and include the final event.
- Too few components: requested components must fit the sample and feature counts. Two is the default.
- Empty graph matrix: graph-matrix PCA requires a nonempty adjacency and Laplacian.
- Crowded graph illustration: only the drawing shows a subset; all matrix analysis retains the full node set.
- After changing settings: rerun python -m pytest -q and python run_project.py so outputs stay synchronized.
