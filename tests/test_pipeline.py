import json
import pandas as pd
import pytest

from src.pipeline import RunConfig, run_pipeline, validate_config
from synthetic import make_synthetic


def test_config_validation():
    with pytest.raises(ValueError):
        validate_config(RunConfig(interval="day"))
    with pytest.raises(ValueError):
        validate_config(RunConfig(quantum_samples=2))
    with pytest.raises(ValueError):
        validate_config(RunConfig(subset_nodes=100))
    with pytest.raises(ValueError):
        validate_config(RunConfig(graph_representation="degree-only"))


@pytest.mark.parametrize("mode", ["binary", "count"])
def test_end_to_end(tmp_path, mode):
    data = make_synthetic(tmp_path / "d.txt.gz", gap_days=(40, 70))
    cfg = RunConfig(interval="week", edge_mode=mode, data_file=str(data), output_dir=str(tmp_path / "o"),
                    quantum_samples=20, skip_quantum=False)
    try:
        import qiskit  # noqa
        has_q = True
    except ImportError:
        has_q = False
    meta = run_pipeline(cfg)
    res, fig = tmp_path / "o" / "results", tmp_path / "o" / "figures"
    temporal = res / "01_temporal_graph"
    full_pca = res / "02_feature_pca" / "full_node_temporal"
    matrix_pca = res / "04_graph_matrix_pca"
    assert meta["n_empty_snapshots"] > 0
    for path in [temporal / "snapshot_graph_statistics.csv",
                 temporal / "snapshot_graph_features.csv",
                 temporal / "node_id_to_matrix_index.csv",
                 full_pca / "full_temporal_pca_scores.csv",
                 full_pca / "temporal_node_degree_features.csv",
                 res / "02_feature_pca" / "matched_sample" / "matched_sample_classical_pca_scores.csv"]:
        assert path.exists()
    for method in ["adjacency", "laplacian", "combined"]:
        scores = pd.read_csv(matrix_pca / f"{method}_matrix_pca_scores.csv")
        assert len(scores) == meta["n_nodes"]
        assert scores["node_id"].nunique() == meta["n_nodes"]
    assert (temporal / "global_adjacency_matrix.npz").exists()
    assert (temporal / "global_directed_out_laplacian_matrix.npz").exists()
    assert (matrix_pca / "graph_matrix_pca_metrics.csv").exists()
    assert json.loads((res / "run_metadata.json").read_text())["n_nodes"] == meta["n_nodes"]
    expected = 15 + (1 if has_q else 0)
    assert len(list(fig.glob("*.png"))) == expected
    if has_q:
        cmp = pd.read_csv(res / "05_classical_quantum_comparison" / "matched_classical_vs_quantum_scores.csv")
        ids = pd.read_csv(temporal / "node_id_to_matrix_index.csv").set_index("node_index")["node_id"]
        assert (cmp["node_id"].to_numpy() == ids.loc[cmp["node_index"]].to_numpy()).all()
        assert len(cmp) == 20 and {"classical_PC1", "quantum_QPC2"} <= set(cmp.columns)
        assert meta["comparison_metrics"]["distance_rank_correlation"] == meta["comparison_metrics"]["distance_rank_correlation"]


def test_reproducible(tmp_path):
    data = make_synthetic(tmp_path / "d.txt.gz")
    outs = []
    for k in range(2):
        cfg = RunConfig(data_file=str(data), output_dir=str(tmp_path / f"o{k}"), quantum_samples=15, skip_quantum=True)
        run_pipeline(cfg)
        outs.append(pd.read_csv(tmp_path / f"o{k}" / "results" / "02_feature_pca" / "matched_sample" /
                                "matched_sample_classical_pca_scores.csv"))
    pd.testing.assert_frame_equal(outs[0], outs[1])


def test_missing_file_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        run_pipeline(RunConfig(data_file=str(tmp_path / "x.txt"), output_dir=str(tmp_path)))
