import numpy as np
import pytest

pytest.importorskip("qiskit")
from sklearn.decomposition import KernelPCA

from src import quantum_kpca as Q


@pytest.fixture(scope="module")
def result():
    X = np.random.default_rng(3).random((25, 2)) * 5
    return X, Q.run_quantum_kernel_pca(X, 2)


def test_kernel_properties(result):
    _, r = result
    K = r.kernel
    assert K.shape == (25, 25) and np.allclose(K, K.T)
    assert np.allclose(np.diag(K), 1) and K.min() >= -1e-12 and K.max() <= 1 + 1e-12
    assert np.linalg.eigvalsh(K).min() > -1e-8
    assert r.n_qubits == 2


def test_known_overlap_matches_formula():
    angles = np.array([[0.3, 1.1], [0.3, 1.1], [2.0, 0.5]])
    K = Q.fidelity_kernel(Q.statevectors(angles))
    assert K[0, 1] == pytest.approx(1.0) and K[0, 2] < 1.0


def test_centering_and_eigen(result):
    _, r = result
    assert np.allclose(r.centered_kernel.sum(axis=0), 0, atol=1e-9)
    assert (r.eigenvalues >= 0).all() and np.all(np.diff(r.eigenvalues) <= 1e-12)
    assert r.explained_fraction.sum() == pytest.approx(1.0)
    assert np.allclose(r.scores.T @ r.scores, np.diag(r.eigenvalues[:2]), atol=1e-8)


def test_matches_sklearn_precomputed_kernel_pca(result):
    _, r = result
    kp = KernelPCA(n_components=2, kernel="precomputed").fit(r.kernel)
    assert np.allclose(kp.eigenvalues_, r.eigenvalues[:2], rtol=1e-6)
    sk = kp.transform(r.kernel)
    for j in range(2):
        assert np.allclose(np.abs(sk[:, j]), np.abs(r.scores[:, j]), atol=1e-6)


def test_deterministic(result):
    X, r = result
    assert np.array_equal(Q.run_quantum_kernel_pca(X, 2).scores, r.scores)


def test_input_validation():
    with pytest.raises(ValueError):
        Q.run_quantum_kernel_pca(np.ones((5, 3)))
    with pytest.raises(ValueError):
        Q.run_quantum_kernel_pca(np.random.rand(2, 2))
    with pytest.raises(ValueError):
        Q.run_quantum_kernel_pca(np.array([[1, 2], [1, 3], [1, 4.0]]))      # constant feature
    with pytest.raises(ValueError):
        Q.run_quantum_kernel_pca(np.array([[np.nan, 1], [2, 3], [3, 4.0]]))
    with pytest.raises(Q.KernelError):
        Q.fidelity_kernel(np.ones((3, 4)))                                  # not normalised


def test_missing_qiskit_gives_clear_error(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("qiskit"):
            raise ImportError("blocked")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    with pytest.raises(Q.QiskitNotAvailableError, match="pip install qiskit"):
        Q.run_quantum_kernel_pca(np.random.rand(5, 2))
