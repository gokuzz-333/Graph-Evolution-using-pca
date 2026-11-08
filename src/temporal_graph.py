"""Load temporal email interactions and build sparse directed graph snapshots.

Conventions (used consistently everywhere in this project)
----------------------------------------------------------
* Dataset rows are ``sender receiver timestamp``. Timestamps in the SNAP
  Email-Eu-core-temporal file are **seconds since the start of the data
  collection**, not Unix epoch times. We therefore bucket time as
  ``floor((t - t_min) / window)``: a "week" is 7 days and a "month" is a fixed
  30-day window (calendar months are meaningless for relative timestamps).
* ``A[i, j]`` = interaction from sender ``i`` to receiver ``j``
  (binary: 1 if at least one email; count: number of emails).
* Self-loops (sender == receiver) are **dropped** from every snapshot, but the
  nodes that appear in them still keep their place in the node mapping.
* Empty time windows are **kept** as all-zero snapshots so the time axis has no
  gaps (set ``include_empty=False`` to drop them).
* Outgoing directed Laplacian: ``L_out = D_out - A`` with
  ``D_out = diag(A @ 1)`` (row sums). Every row of ``L_out`` sums to zero.
"""
from __future__ import annotations

import gzip
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from scipy import sparse
from scipy.sparse import csr_matrix, diags

DATA_URL = "https://snap.stanford.edu/data/email-Eu-core-temporal.txt.gz"
DEFAULT_FILE = Path("data/email-Eu-core-temporal.txt.gz")
SECONDS_PER_DAY = 86_400
INTERVAL_DAYS = {"week": 7, "month": 30}
INTERVAL_PREFIX = {"week": "W", "month": "M"}
EQUAL_INTERVAL = "equal"
EDGE_MODES = ("binary", "count")


class DataError(ValueError):
    """Raised when the dataset file or its contents are invalid."""


@dataclass(frozen=True)
class Snapshot:
    """One time window of the temporal graph."""
    index: int
    label: str
    start_day: float          # days since first timestamp in the dataset
    end_day: float
    A: csr_matrix             # directed adjacency (n x n)
    L: csr_matrix             # outgoing directed Laplacian (n x n)
    n_interactions: int       # raw emails in the window (self-loops removed)

    @property
    def n_edges(self) -> int:
        """Number of distinct directed edges (non-zeros of A)."""
        return int(self.A.nnz)


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def download_dataset(path: Path = DEFAULT_FILE) -> Path:
    """Download the SNAP file if it is not already on disk."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 0:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading dataset from {DATA_URL}")
    try:
        response = requests.get(DATA_URL, timeout=90)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise DataError(
            f"Could not download the dataset ({exc}). Download "
            f"'email-Eu-core-temporal.txt.gz' manually from "
            f"https://snap.stanford.edu/data/email-Eu-core-temporal.html and place it at {path}"
        ) from exc
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(response.content)
    tmp.replace(path)
    return path


def read_interactions(path: Path) -> pd.DataFrame:
    """Read ``sender receiver timestamp`` rows; validate and return int64 columns."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    opener = gzip.open if str(path).endswith(".gz") else open
    try:
        with opener(path, "rt", encoding="utf-8") as f:
            raw = pd.read_csv(f, sep=r"\s+", comment="#", header=None,
                              usecols=[0, 1, 2], names=["sender", "receiver", "timestamp"],
                              dtype=np.float64)
    except (OSError, EOFError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise DataError(f"Could not parse {path} as 'sender receiver timestamp' rows: {exc}") from exc
    if raw.empty:
        raise DataError("No interaction rows found. Check the dataset file.")
    if raw.isna().any().any():
        bad = int(raw.isna().any(axis=1).sum())
        raise DataError(f"{bad} row(s) have missing values; expected 3 numeric columns per row.")
    values = raw.to_numpy()
    if not np.isfinite(values).all():
        raise DataError("Dataset contains non-finite values (inf/nan).")
    if (values < 0).any():
        raise DataError("Dataset contains negative node IDs or timestamps.")
    if not np.array_equal(values, np.floor(values)):
        raise DataError("Node IDs and timestamps must be integers.")
    return raw.astype("int64")


def _check_frame(df: pd.DataFrame) -> None:
    missing = {"sender", "receiver", "timestamp"} - set(df.columns)
    if missing:
        raise DataError(f"Interaction table is missing columns: {sorted(missing)}")
    if df.empty:
        raise DataError("Interaction table is empty.")


def make_node_mapping(df: pd.DataFrame) -> dict[int, int]:
    """Map original dataset node IDs -> contiguous indices 0..n-1 (sorted by ID)."""
    _check_frame(df)
    ids = np.union1d(df["sender"].to_numpy(), df["receiver"].to_numpy())
    return {int(node): i for i, node in enumerate(ids)}


def node_ids_from_mapping(mapping: dict[int, int]) -> np.ndarray:
    """Return an array ``ids`` with ``ids[index] = original_id``."""
    ids = np.empty(len(mapping), dtype=np.int64)
    for original, index in mapping.items():
        ids[index] = original
    return ids


# --------------------------------------------------------------------------- #
# Matrices
# --------------------------------------------------------------------------- #
def validate_adjacency(A: csr_matrix) -> None:
    if not sparse.issparse(A):
        raise TypeError("Adjacency matrix must be a SciPy sparse matrix.")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"Adjacency matrix must be square, got shape {A.shape}.")
    if A.nnz and not np.isfinite(A.data).all():
        raise ValueError("Adjacency matrix contains non-finite values.")
    if A.nnz and (A.data < 0).any():
        raise ValueError("Adjacency matrix contains negative weights.")


def build_adjacency(rows: np.ndarray, cols: np.ndarray, n: int, edge_mode: str) -> csr_matrix:
    """Directed adjacency in CSR. Duplicate (i, j) pairs are summed, then
    optionally binarised."""
    if edge_mode not in EDGE_MODES:
        raise ValueError(f"edge_mode must be one of {EDGE_MODES}, got {edge_mode!r}")
    if len(rows) == 0:
        return csr_matrix((n, n), dtype=np.float64)
    A = sparse.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n)).tocsr()
    A.sum_duplicates()
    if edge_mode == "binary":
        A.data[:] = 1.0
    A.eliminate_zeros()
    return A


def directed_out_laplacian(A: csr_matrix) -> csr_matrix:
    """Outgoing directed Laplacian ``L_out = D_out - A`` (sparse CSR)."""
    A = csr_matrix(A, dtype=np.float64)
    validate_adjacency(A)
    n = A.shape[0]
    out_degree = np.asarray(A.sum(axis=1)).ravel()
    L = (diags(out_degree, 0, shape=(n, n), format="csr") - A).tocsr()
    L.eliminate_zeros()
    return L


def validate_laplacian(L: csr_matrix, A: csr_matrix, atol: float = 1e-9) -> None:
    """Check shape, finiteness and the defining properties of ``L_out``."""
    if L.shape != A.shape:
        raise ValueError(f"Laplacian shape {L.shape} != adjacency shape {A.shape}.")
    if L.nnz and not np.isfinite(L.data).all():
        raise ValueError("Laplacian contains non-finite values.")
    row_sums = np.asarray(L.sum(axis=1)).ravel()
    if not np.allclose(row_sums, 0.0, atol=atol):
        raise ValueError("Row sums of L_out must be zero (L_out @ 1 = 0).")
    out_degree = np.asarray(A.sum(axis=1)).ravel()
    if not np.allclose(L.diagonal(), out_degree - A.diagonal(), atol=atol):
        raise ValueError("Diagonal of L_out must equal out-degree (minus self-loop weight).")
    off = L - diags(L.diagonal(), 0, shape=L.shape, format="csr")
    if off.nnz and (off.data > atol).any():
        raise ValueError("Off-diagonal entries of L_out must be <= 0.")


# --------------------------------------------------------------------------- #
# Snapshots
# --------------------------------------------------------------------------- #
def build_snapshots(df: pd.DataFrame, interval: str = "month", edge_mode: str = "binary",
                    include_empty: bool = True, snapshot_count: int = 5,
                    snapshot_boundaries_days=None):
    """Split interactions into windows and build ``Snapshot`` objects.

    Returns ``(snapshots, mapping)`` where ``mapping`` maps original node ID -> index.
    """
    if interval not in (*INTERVAL_DAYS, EQUAL_INTERVAL):
        raise ValueError(f"interval must be one of {(*INTERVAL_DAYS, EQUAL_INTERVAL)}, got {interval!r}")
    if snapshot_count < 1:
        raise ValueError("snapshot_count must be at least 1")
    if edge_mode not in EDGE_MODES:
        raise ValueError(f"edge_mode must be one of {EDGE_MODES}, got {edge_mode!r}")
    _check_frame(df)

    mapping = make_node_mapping(df)          # built from ALL rows (incl. self-loops)
    node_ids = node_ids_from_mapping(mapping)
    n = len(node_ids)

    t = df["timestamp"].to_numpy(dtype=np.int64)
    custom_boundaries = None
    if snapshot_boundaries_days is not None:
        custom_boundaries = np.asarray(snapshot_boundaries_days, dtype=float)
        elapsed_days = (t - t.min()) / SECONDS_PER_DAY
        if (custom_boundaries.ndim != 1 or len(custom_boundaries) < 2
                or not np.isfinite(custom_boundaries).all()
                or not np.all(np.diff(custom_boundaries) > 0)
                or custom_boundaries[0] > 0
                or custom_boundaries[-1] < elapsed_days.max()):
            raise ValueError("Custom boundaries must increase, begin at or before day 0, and cover the data.")
        period = np.searchsorted(custom_boundaries[1:], elapsed_days, side="right")
        n_periods = len(custom_boundaries) - 1
        period = np.minimum(period, n_periods - 1)
        window_days = float(np.mean(np.diff(custom_boundaries)))
    elif interval == EQUAL_INTERVAL:
        # Equal, half-open intervals cover the complete observed time span.
        elapsed = t - t.min()
        span = int(t.max() - t.min()) + 1
        period = np.minimum((elapsed * snapshot_count) // span, snapshot_count - 1)
        n_periods = snapshot_count
        window_days = span / SECONDS_PER_DAY / n_periods
    else:
        window_days = INTERVAL_DAYS[interval]
        period = (t - t.min()) // (window_days * SECONDS_PER_DAY)
        n_periods = int(period.max()) + 1

    keep = df["sender"].to_numpy() != df["receiver"].to_numpy()   # drop self-loops
    if not keep.any():
        raise DataError("All interactions are self-loops; no directed edges to build.")
    rows_all = np.searchsorted(node_ids, df["sender"].to_numpy()[keep])
    cols_all = np.searchsorted(node_ids, df["receiver"].to_numpy()[keep])
    period = period[keep]

    order = np.argsort(period, kind="stable")
    period, rows_all, cols_all = period[order], rows_all[order], cols_all[order]
    bounds = np.searchsorted(period, np.arange(n_periods + 1))

    snapshots: list[Snapshot] = []
    for p in range(n_periods):
        lo, hi = bounds[p], bounds[p + 1]
        if hi == lo and not include_empty:
            continue
        A = build_adjacency(rows_all[lo:hi], cols_all[lo:hi], n, edge_mode)
        L = directed_out_laplacian(A)
        validate_laplacian(L, A)
        snapshots.append(Snapshot(
            index=len(snapshots), label=(f"T{p + 1:02d}" if (interval == EQUAL_INTERVAL
                                                                or custom_boundaries is not None)
                                         else f"{INTERVAL_PREFIX[interval]}{p:03d}"),
            start_day=(float(custom_boundaries[p]) if custom_boundaries is not None
                       else float(p * window_days)),
            end_day=(float(custom_boundaries[p + 1]) if custom_boundaries is not None
                     else float((p + 1) * window_days)),
            A=A, L=L, n_interactions=int(hi - lo)))
    if not snapshots:
        raise DataError("No snapshots were produced.")
    return snapshots, mapping


def summarize_snapshots(snapshots, node_count: int) -> pd.DataFrame:
    """Per-snapshot sizes; density = distinct edges / (n (n-1))."""
    if node_count < 2:
        raise ValueError("Need at least 2 nodes.")
    possible = node_count * (node_count - 1)
    rows = []
    for s in snapshots:
        rows.append({
            "snapshot": s.label, "start_day": s.start_day, "end_day": s.end_day,
            "nodes": node_count, "interactions": s.n_interactions,
            "distinct_directed_edges": s.n_edges, "total_edge_weight": float(s.A.sum()),
            "possible_directed_edges_no_self_loops": possible,
            "density": s.n_edges / possible,
            "adjacency_nnz": int(s.A.nnz), "laplacian_nnz": int(s.L.nnz),
            "is_empty": s.n_interactions == 0,
        })
    return pd.DataFrame(rows)
