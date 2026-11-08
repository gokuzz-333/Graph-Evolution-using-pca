import numpy as np
from scipy import sparse

from src.graph_embedding import compare_graph_representations
from src.temporal_graph import directed_out_laplacian


def test_all_graph_representations_use_full_nodes_and_centered_pca():
    adjacency = sparse.csr_matrix([[0, 1, 0, 0], [0, 0, 1, 0],
                                   [1, 0, 0, 0], [0, 0, 0, 0]], dtype=float)
    laplacian = directed_out_laplacian(adjacency)
    results, metrics = compare_graph_representations(adjacency, laplacian, 2)
    repeat, _ = compare_graph_representations(adjacency, laplacian, 2)
    assert set(results) == {"adjacency", "laplacian", "combined"}
    for name, result in results.items():
        assert result.scores.shape == (4, 2)
        assert np.allclose(result.scores, repeat[name].scores)
        assert 0 <= result.cumulative_explained_variance <= 1 + 1e-10
    assert metrics.loc[metrics.representation == "combined", "matrix_columns"].item() == 8
    assert not np.allclose(results["adjacency"].scores, results["laplacian"].scores)
