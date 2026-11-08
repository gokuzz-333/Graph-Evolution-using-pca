import numpy as np
import pytest

from src import features as F
from src import temporal_graph as T
from test_temporal_graph import frame, DAY


def build(rows, interval="week", mode="binary"):
    snaps, m = T.build_snapshots(frame(rows), interval, mode)
    return snaps, m, T.node_ids_from_mapping(m)


def test_snapshot_features_values():
    snaps, m, ids = build([[1, 2, 0], [1, 3, 5], [2, 1, 6], [4, 4, 7], [1, 2, 8 * DAY]])
    f = F.snapshot_features(snaps, len(m))
    r = f.iloc[0]
    assert r.edge_count == 3 and r.active_senders == 2 and r.active_receivers == 3
    assert r.max_out_degree == 2 and r.max_in_degree == 1
    assert r.isolated_nodes == 1                      # node 4 only had a self-loop
    assert r.density == pytest.approx(3 / (4 * 3))
    assert r.mean_out_degree == pytest.approx(3 / 4) and r.mean_in_degree == pytest.approx(3 / 4)


def test_node_features_shape_names_and_means():
    snaps, m, ids = build([[1, 2, 0], [1, 3, 5], [2, 1, 8 * DAY]])
    X, names = F.node_feature_matrix(snaps, ids)
    T_ = len(snaps)
    assert X.shape == (3, 2 * T_ + 2) and len(names) == X.shape[1]
    assert names[-2:] == F.SUMMARY_FEATURES
    out_cols = [i for i, nm in enumerate(names[:-2]) if nm.endswith("out_degree")]
    assert np.allclose(X[:, -2], X[:, out_cols].mean(axis=1))
    assert X[m[1], -2] == pytest.approx(2 / T_)       # node 1 sent 2 distinct edges in week 0
    assert ids[m[3]] == 3                              # index <-> original ID preserved


def test_summary_features_use_whole_timeline_not_first_snapshot():
    snaps, m, ids = build([[1, 2, 0], [3, 2, 8 * DAY]])
    X, names = F.node_feature_matrix(snaps, ids)
    X2 = F.summary_feature_matrix(X, names)
    assert X2[m[3], 0] > 0 and X[m[3], 0] == 0       # node 3 silent in snapshot 0 but counted


def test_preprocess_validation():
    with pytest.raises(ValueError):
        F.preprocess_summary_features(np.array([[1.0, np.nan]]))
    with pytest.raises(ValueError):
        F.preprocess_summary_features(np.array([[1.0, -1.0]]))
    with pytest.raises(ValueError):
        F.preprocess_summary_features(np.ones((3, 3)))
    assert np.allclose(F.preprocess_summary_features(np.array([[0.0, 1.0]])), [[0, np.log(2)]])


def test_select_sample_deterministic_and_active_only():
    X2 = np.vstack([np.zeros((5, 2)), np.random.default_rng(0).random((50, 2)) + 0.1])
    a, b = F.select_sample(X2, 10, 7), F.select_sample(X2, 10, 7)
    assert np.array_equal(a, b) and (a >= 5).all() and len(set(a)) == 10 and (np.diff(a) > 0).all()
    assert not np.array_equal(a, F.select_sample(X2, 10, 8))
    assert len(F.select_sample(X2, 500, 1)) == 50
    with pytest.raises(ValueError):
        F.select_sample(X2, 2, 1)


def test_pca_matches_manual_svd_and_properties():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(80, 5)) @ rng.normal(size=(5, 5))
    r = F.run_classical_pca(X, 2)
    Z = (X - X.mean(0)) / X.std(0)
    s = np.linalg.svd(Z, compute_uv=False)
    assert np.allclose(r.explained_variance_ratio, (s**2 / (s**2).sum())[:2])
    assert abs(r.scores.mean()) < 1e-9
    assert np.allclose(np.cov(r.scores.T)[0, 1], 0, atol=1e-9)
    assert r.explained_variance_ratio[0] >= r.explained_variance_ratio[1]


def test_pca_handles_constant_column_and_rejects_bad_input():
    X = np.column_stack([np.zeros(10), np.arange(10.0), np.arange(10.0) ** 2])
    assert np.isfinite(F.run_classical_pca(X, 2).scores).all()
    with pytest.raises(ValueError):
        F.run_classical_pca(np.array([[1.0, np.inf], [2.0, 3.0], [3.0, 4.0]]))
    with pytest.raises(ValueError):
        F.run_classical_pca(np.ones((1, 3)))
    with pytest.raises(ValueError):
        F.run_classical_pca(np.ones((5, 1)), 2)
