"""Central defaults for the reproducible project run."""

WEEKLY_ANALYSIS = False
SNAPSHOT_MODE = "week" if WEEKLY_ANALYSIS else "equal"  # equal, week, or month
SNAPSHOT_COUNT = 5
SNAPSHOT_BOUNDARIES_DAYS = None  # e.g. [0, 100, 250, 400, 600, 804]
EDGE_MODE = "binary"          # binary or count
GRAPH_REPRESENTATION = "combined"  # adjacency, laplacian, or combined
N_COMPONENTS = 4
# Graph-matrix compression experiment; this is separate from the other PCA branches.
GRAPH_COMPONENT_SWEEP = (2, 5, 10, 20, 30, 50)
OUTPUT_DIR = "."
QUANTUM_SAMPLES = 60
QUANTUM_REPS = 1
QUANTUM_ENTANGLEMENT = "linear"
RANDOM_SEED = 7
LOG_TRANSFORM = True
VISUALIZATION_SUBSET_NODES = 15
ADJACENCY_VISUALIZATION_NODES = 40
