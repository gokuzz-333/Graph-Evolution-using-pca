import gzip
import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from src import temporal_graph as T
from synthetic import make_synthetic

DAY = 86400


def frame(rows):
    return pd.DataFrame(rows, columns=["sender", "receiver", "timestamp"]).astype("int64")


def test_parse_ok_and_comments(tmp_path):
    p = tmp_path / "d.txt.gz"
    with gzip.open(p, "wt") as f:
        f.write("# comment\n5 7 0\n7 5 100\n")
    df = T.read_interactions(p)
    assert df.shape == (2, 3) and df.dtypes.eq("int64").all()


@pytest.mark.parametrize("content", ["", "1 2\n", "1 2 x\n", "1 2 -5\n", "1 2 3.5\n"])
def test_parse_rejects_bad_files(tmp_path, content):
    p = tmp_path / "bad.txt"
    p.write_text(content)
    with pytest.raises(T.DataError):
        T.read_interactions(p)


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        T.read_interactions("nope.txt")


def test_node_mapping_roundtrip():
    df = frame([[10, 30, 0], [30, 20, 5]])
    m = T.make_node_mapping(df)
    assert m == {10: 0, 20: 1, 30: 2}
    assert T.node_ids_from_mapping(m).tolist() == [10, 20, 30]


def test_snapshots_duplicates_selfloops_binary_vs_count():
    df = frame([[1, 2, 0], [1, 2, 10], [1, 2, 20], [2, 3, 30], [3, 3, 40], [1, 3, 31 * DAY]])
    snaps_b, m = T.build_snapshots(df, "month", "binary")
    snaps_c, _ = T.build_snapshots(df, "month", "count")
    assert len(snaps_b) == 2 and len(m) == 3
    i, j, k = m[1], m[2], m[3]
    assert snaps_b[0].A[i, j] == 1 and snaps_c[0].A[i, j] == 3     # duplicates
    assert snaps_b[0].A.nnz == snaps_c[0].A.nnz == 2               # self-loop 3->3 dropped
    assert snaps_b[0].A[k, k] == 0
    assert snaps_b[0].n_interactions == 4                          # self-loop not counted
    assert snaps_b[1].A[i, k] == 1
    assert sparse.isspmatrix_csr(snaps_b[0].A) and sparse.isspmatrix_csr(snaps_b[0].L)


def test_direction_preserved():
    df = frame([[1, 2, 0], [3, 2, 1]])
    (s,), m = T.build_snapshots(df, "week", "binary")
    assert s.A[m[1], m[2]] == 1 and s.A[m[2], m[1]] == 0


def test_window_boundaries_and_empty_snapshots():
    df = frame([[1, 2, 0], [2, 1, 7 * DAY - 1], [1, 2, 3 * 7 * DAY]])
    snaps, _ = T.build_snapshots(df, "week", "binary")
    assert [s.n_interactions for s in snaps] == [2, 0, 0, 1]
    assert snaps[1].A.nnz == 0 and snaps[1].L.nnz == 0
    snaps2, _ = T.build_snapshots(df, "week", "binary", include_empty=False)
    assert len(snaps2) == 2


def test_laplacian_properties():
    snaps, _ = T.build_snapshots(make_synthetic_df(), "month", "count")
    for s in snaps:
        T.validate_laplacian(s.L, s.A)
        assert np.allclose(np.asarray(s.L.sum(axis=1)).ravel(), 0)
        assert np.allclose(s.L.diagonal(), np.asarray(s.A.sum(axis=1)).ravel())


def test_laplacian_small_example():
    A = sparse.csr_matrix(np.array([[0, 2, 1], [0, 0, 1], [0, 0, 0]], float))
    L = T.directed_out_laplacian(A).toarray()
    assert np.array_equal(L, [[3, -2, -1], [0, 1, -1], [0, 0, 0]])


def test_laplacian_validation_errors():
    with pytest.raises(ValueError):
        T.directed_out_laplacian(sparse.csr_matrix(np.ones((2, 3))))
    with pytest.raises(ValueError):
        T.directed_out_laplacian(sparse.csr_matrix(np.array([[0, -1], [0, 0]], float)))
    with pytest.raises(ValueError):
        T.directed_out_laplacian(sparse.csr_matrix(np.array([[0, np.nan], [0, 0]])))
    A = sparse.csr_matrix(np.array([[0, 1], [0, 0]], float))
    bad = sparse.csr_matrix(np.array([[1, -1], [0, 5]], float))
    with pytest.raises(ValueError):
        T.validate_laplacian(bad, A)


def test_invalid_parameters():
    df = frame([[1, 2, 0]])
    with pytest.raises(ValueError):
        T.build_snapshots(df, "year", "binary")
    with pytest.raises(ValueError):
        T.build_snapshots(df, "month", "weighted")
    with pytest.raises(T.DataError):
        T.build_snapshots(df.iloc[0:0], "month", "binary")
    with pytest.raises(T.DataError):
        T.build_snapshots(frame([[1, 1, 0]]), "month", "binary")      # only self-loops


def test_summary_density_consistent():
    snaps, m = T.build_snapshots(make_synthetic_df(), "month", "count")
    summ = T.summarize_snapshots(snaps, len(m))
    n = len(m)
    assert np.allclose(summ["density"], summ["distinct_directed_edges"] / (n * (n - 1)))
    assert (summ["distinct_directed_edges"] <= summ["interactions"]).all()


def test_equal_snapshots_cover_observed_span_and_keep_all_nodes():
    df = frame([[1, 2, 0], [2, 3, 10 * DAY], [3, 1, 20 * DAY]])
    snaps, mapping = T.build_snapshots(df, "equal", "binary", snapshot_count=5)
    assert len(snaps) == 5
    assert all(s.A.shape == (3, 3) for s in snaps)
    assert sum(s.n_interactions for s in snaps) == 3
    assert snaps[0].start_day == 0
    assert snaps[-1].end_day >= 20
    custom, _ = T.build_snapshots(df, "equal", "binary",
                                 snapshot_boundaries_days=[0, 5, 15, 21])
    assert [s.n_interactions for s in custom] == [1, 1, 1]
    with pytest.raises(ValueError):
        T.build_snapshots(df, "equal", "binary", snapshot_boundaries_days=[0, 5, 10])


def make_synthetic_df():
    import tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp()) / "s.txt.gz"
    make_synthetic(d, n_nodes=30, n_rows=2000, days=90)
    return T.read_interactions(d)
