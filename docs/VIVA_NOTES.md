# Viva notes

## Project summary

This project studies how email communication changes over time using the real SNAP Email-Eu-core-temporal file. Each event is a directed sender-to-receiver interaction at a relative time in seconds. Five equal, non-overlapping snapshots cover the observed 803.93-day range. Every snapshot keeps the same 986 node IDs.

For each snapshot, I build sparse adjacency A and the outgoing directed Laplacian L_out. I reduce the full global matrices in three ways: A only, L_out only, and both together. I also run centered classical PCA on temporal degree features, which is a separate experiment. Finally, Qiskit defines a fidelity kernel for 60 sampled active nodes. The quantum circuit is statevector-simulated on a classical computer and the kernel is diagonalized classically. I call that quantum-kernel PCA and make no quantum-advantage claim.

## Verified results

- Dataset: 986 nodes and 332,334 timestamped event rows.
- Original node IDs range from 0 to 1004 with gaps; all 986 IDs are unique and remain in one sorted global mapping.
- Time range: timestamp 0 to 69,459,254 seconds, equal to 803.9266 elapsed days.
- Exact duplicate event rows: 4,998. Self-loops: zero.
- Five snapshot interaction counts: 89,932; 95,251; 106,383; 38,344; and 2,424.
- Distinct directed edges per snapshot: 12,108; 12,386; 13,959; 8,136; and 1,355.
- Global union graph: 24,929 directed edges, matching SNAP's published static edge count.
- Standardized temporal node-feature PCA: PC1 explains 81.2%, PC2 explains 5.9%.
- Snapshot-level graph-statistic PCA: the first two components explain 96.8% and 2.6%. It uses only five observations, so interpret cautiously.
- Graph-matrix PCA explained variance for PC1 + PC2: 16.3% for adjacency, 9.3% for L_out, and 9.8% for the combined input. The two-component views capture a limited share of connection-pattern variation.
- Directed average clustering is 0.263 in T01, peaks at 0.286 in T03, and falls to 0.078 in T05.
- Quantum-kernel PCA completed for 60 nodes. The first two kernel components account for 24.3% and 20.3% of the centered-kernel eigenvalue sum.
- Sample classical PCA took about 0.0012 seconds; quantum-kernel PCA took about 0.82 seconds in this run. The timings depend on machine and load.
- Pairwise-distance rank correlation of the classical and quantum sample embeddings was 0.042. This is descriptive, not evidence of improvement.

## Algorithms and equations

### Temporal graph

An event row is (sender, receiver, time). A snapshot collects events within a chronological interval. One sorted global mapping assigns every original node ID a stable matrix index. A node with no edges in one snapshot still has a zero row and column.

### Directed adjacency

A[i,j] = 1 in binary mode when i sent at least one email to j in that snapshot. In count mode, A[i,j] is the number of interaction rows for that ordered pair. The graph stays directed because A[i,j] can differ from A[j,i].

### Outgoing Laplacian

~~~~
D_out[i,i] = sum_j A[i,j]
L_out = D_out − A
~~~~

The diagonal is outgoing degree. Each row of L_out sums to zero. The outgoing convention is one choice for directed graphs.

### PCA on adjacency and Laplacian matrices

Each node is one row of a 986-row matrix, and each column records its connection value to one mapped node. PCA centers every column across the nodes, but does not standardize it. The adjacency analysis uses A; the Laplacian analysis uses L_out, including negative off-diagonal entries. The combined 986 × 1,972 input places A and L_out side by side after scaling each block to unit Frobenius norm. The custom implementation diagonalizes the smaller matrix: feature covariance when there are fewer columns than rows, or the node-by-node Gram matrix for wide inputs. Both yield the same non-zero PCA eigenvalues and feature directions. PC1 + PC2 explain 16.3% for A, 9.3% for L_out, and 9.8% for the combined matrix. These scores summarize variation in connection patterns; they are distinct from the separate PCA on temporal degree features.

### Classical PCA

The node-feature analysis uses the custom NumPy PCA in `src/pca_engine.py`. It standardizes each column to zero mean and unit population standard deviation, centers the resulting matrix, computes its covariance, uses `numpy.linalg.eigh`, sorts eigenvectors by descending eigenvalue, and projects each node to get scores. The input columns are snapshot out-degree and in-degree histories. This PCA does not take adjacency or Laplacian matrices as input.

#### Exact node features in this run

The full node-feature PCA has 986 rows and 12 columns. With T = 5 snapshots, the width is 2T + 2 = 12:

- T01_out_degree, T01_in_degree, …, T05_out_degree, T05_in_degree: each node’s outgoing and incoming degree in that one window.
- mean_out_degree and mean_in_degree: each node’s average corresponding degree over all five windows, including zero-activity windows.

For the sample-matched comparison, the code uses only mean_out_degree and mean_in_degree. A node is eligible if the sum of these two nonnegative values is above zero. The code selects 60 eligible nodes uniformly without replacement with seed 7 and sorts the selected row indices. Both methods receive log1p of the same two columns. Classical PCA then standardizes the two columns. For the quantum feature map, each column is separately min–max scaled to [0, π].

Graph-level values such as edge count, density, active sender/receiver counts and clustering describe a snapshot as a whole; they are not columns in the 986-row node PCA. A separate snapshot-level PCA uses selected graph-statistic columns, with five snapshots as its observations.

#### What PC1 and PC2 mean here

After standardisation, let Z be the 986 × 12 matrix. The code centers Z, forms its feature covariance matrix, and uses `numpy.linalg.eigh` to find eigenvectors. It sorts them by descending eigenvalue. PC1 chooses a unit direction that maximises Var(Zv); PC2 maximises the remaining projected variance subject to v₂ being perpendicular to v₁. A loading is an entry of v, the weight assigned to one feature. A score is a node’s coordinate after projecting its standardized feature row onto v.

Latest verified full-data result: PC1 explains 81.2%, PC2 explains 5.9%, and together they explain 87.1% of standardized variance. PC1 has positive loadings across the degree-history columns, which fits a broad activity interpretation (the sign of any PCA direction is arbitrary). T05_out_degree has the largest absolute PC2 loading, +0.815, so the second direction is strongly associated with outgoing activity in the final window. In the scatter, most nodes cluster near the origin and higher log mean-degree colours extend along the positive PC1 tail; there is no clear visual grouping along PC2.

Loadings describe each direction in feature space; scores locate nodes in the reduced space. The quantum branch instead reports kernel-PC scores from eigenvectors of the centred fidelity kernel. Those coordinates do not have the same direct feature-wise loading interpretation as classical PCA.

### Quantum-kernel PCA

The two mean-degree features are log-transformed and scaled to [0, pi]. A two-qubit ZZFeatureMap encodes them as quantum states. The fidelity kernel is K[i,j] = |<psi_i | psi_j>| squared. The code centers K and calculates its eigenvectors and eigenvalues on a classical computer. No quantum hardware was used.

## Likely viva questions

**Why study a temporal network?**  
A static graph loses when interactions occurred. Snapshots make network change measurable over time.

**Why is the graph directed?**  
An email from A to B does not imply B emailed A. The sender determines the matrix row, and the receiver determines the column.

**What does sparse mean?**  
Most possible sender-receiver pairs have no edge in one time window. CSR storage keeps only nonzero matrix entries.

**Why keep every node in every snapshot?**  
It preserves a consistent row and column order. A temporarily inactive person remains represented by zeros instead of disappearing.

**What does the outgoing Laplacian represent?**  
It combines outgoing degree with outgoing edge relationships. Here it is L_out = D_out − A and every row sums to zero.

**Does the Laplacian really affect dimensionality reduction?**  
Yes. The pipeline runs PCA on L_out separately and includes its normalized columns in the combined PCA input.

**How does graph-matrix PCA differ from temporal-feature PCA?**  
Graph-matrix PCA treats each node's links to all nodes as features. Temporal-feature PCA uses only each node's in-degree and out-degree histories across snapshots. Both center their inputs; the temporal-feature PCA additionally standardizes its feature columns.

**Why use two components when matrix retention is low?**  
Two dimensions make the embeddings easy to plot. The explained-variance ratios show that PC1 and PC2 capture only part of the variation in the graph matrices.

**How is directed clustering defined?**  
NetworkX averages directed local triangle coefficients using directed in/out degree and reciprocal links. Nodes with zero clustering contribute zero.

**Did the quantum method outperform classical PCA?**  
There is no evidence for that. The quantum-kernel experiment ran in a classical simulator, took longer on this small sample, and uses a different feature space.

**How is the classical-versus-quantum comparison fair?**  
Both methods use the same 60 sampled nodes and the same two log-transformed mean-degree features. They scale the features differently, so we compare distances descriptively.

**What is the key limitation?**  
Two-component graph-matrix views preserve only a minority of matrix energy. The quantum experiment uses only two features and 60 sampled nodes.

**Can the experiment be repeated?**  
Yes. Seed 7 fixes the active-node sampling. Custom covariance/eigen PCA is deterministic; wide graph-matrix inputs use the equivalent smaller node-by-node Gram matrix. The run metadata records settings, software versions, metrics, and output paths.

**What quantum computing method did you use?**  
This is simulated quantum-kernel PCA, not the hardware qPCA algorithm. Qiskit’s two-qubit ZZFeatureMap uses the two log-transformed mean-degree features after scaling each to [0, π]. The verified settings are `reps=1` and `entanglement="linear"`. Qiskit `Statevector` simulates the bound circuits and returns a state for each row. The squared overlap gives Kᵢⱼ = |⟨ψᵢ|ψⱼ⟩|². The code then centres K with HKH and uses classical `numpy.linalg.eigh`; the output coordinates are eigenvectors multiplied by square-root eigenvalues. Neither the kernel eigendecomposition nor a speed-up is quantum.

**What did you learn from representing email as a temporal graph?**  
The event row has a sender, receiver and relative timestamp, so A[sender, receiver] preserves who sent to whom in each time window. A single node map keeps a person in the same row across snapshots, even when their degrees are zero in a particular window. Binary mode records distinct ordered pairs, so repeated emails within a window do not increase an edge weight.

**What did you learn about sparse matrices and the directed Laplacian?**  
The network has 986² possible ordered pairs but only a small fraction are present, so CSR stores nonzero entries instead of allocating a dense matrix. The chosen outgoing convention is L_out = D_out − A, with D_out from row sums; consequently every row of L_out sums to zero. Direction and the outgoing-degree convention matter, so this should not be casually replaced with an undirected Laplacian.

**What do the actual PCA components show?**  
For the full node-level experiment, PC1 explains 81.2%, PC2 5.9%, cumulative 87.1%. PC1 mostly represents broad degree activity across time. PC2 has its strongest absolute weight on T05_out_degree (+0.815), so late-window outgoing activity contributes strongly. In the scatter, low-activity nodes cluster close to zero and higher-activity nodes stretch rightward on PC1. The plot is descriptive, not a community label or causal result.

**What did the comparison teach you?**  
The circuit feature map defines a similarity, but classical centring and eigen-decomposition still create the final kernel-PCA coordinates. On this 60-node sample, the pairwise-distance rank correlation is 0.042, suggesting little agreement in distance ordering with the classical sample embedding. The two methods also use different coordinate spaces and variance measures, so this is not evidence of quantum advantage.

## Contribution and personal reflection

The repository supports these project-level contributions: temporal email-data preparation; stable node indexing; sparse directed adjacency and outgoing Laplacian construction; graph and temporal node-feature extraction; classical PCA; integration of Qiskit’s simulated ZZ fidelity kernel with classical kernel-PCA post-processing; saved plots, metrics and an editable presentation. These are project capabilities visible in the source and outputs. The repository does not establish who personally authored each part, so edit the following statement before presenting:

> My personal contribution was [the code or tasks I personally wrote or changed]. I tested it by [the exact test or run] and checked [the relevant output]. I can explain [the parts I understand and can modify].

Technical reflection grounded in the implementation: preserving one node order across all snapshots keeps rows comparable; sparse CSR makes the directed matrices practical; standardising the 12 degree-history columns makes PCA depend on co-variation rather than raw degree scale; and the quantum feature map only supplies the fidelity kernel while the kernel-PCA embedding is classical. The code checks matrix shapes and finite feature values, and quantum tests check kernel properties. The project test suite most recently recorded 37 passing tests and a completed full-data run. Do not claim you personally handled a challenge or wrote a function unless that matches your actual work. A concrete next improvement is to compare fixed 30-day windows and repeat the 60-node comparison over multiple seeds or larger samples.

For live, low-risk changes and exact commands, see `docs/CODE_GUIDE.md`.
