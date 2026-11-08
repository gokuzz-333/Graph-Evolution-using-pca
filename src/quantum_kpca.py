"""Quantum-kernel PCA prototype (Qiskit statevector simulation, 2 qubits).

This is NOT the original quantum PCA (density-matrix exponentiation / phase
estimation). Here a quantum feature map ``x -> |psi(x)>`` defines a fidelity kernel
``K_ij = |<psi(x_i)|psi(x_j)>|^2``; the kernel is centred and diagonalised
**classically** (kernel PCA). The circuits are simulated on a classical computer:
no quantum hardware is used and no quantum speed-up is claimed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

INSTALL_HINT = "pip install qiskit qiskit-aer"


class QiskitNotAvailableError(RuntimeError):
    """Qiskit could not be imported."""


class KernelError(ValueError):
    """The kernel failed a numerical sanity check."""


@dataclass
class QuantumKPCAResult:
    scores: np.ndarray               # (m, n_components)
    eigenvalues: np.ndarray          # all eigenvalues of centred kernel, descending, clipped >= 0
    explained_fraction: np.ndarray   # eigenvalue / sum(eigenvalues), all components
    kernel: np.ndarray               # raw fidelity kernel (m, m)
    centered_kernel: np.ndarray
    angles: np.ndarray               # (m, 2) values fed to the feature map, in [0, pi]
    n_qubits: int
    backend: str = "qiskit.quantum_info.Statevector (classical simulation)"


def _load_feature_map(n_qubits: int, reps: int, entanglement: str):
    try:
        import qiskit.circuit.library as lib
        from qiskit.quantum_info import Statevector
    except ImportError as exc:
        raise QiskitNotAvailableError(
            f"Qiskit is not installed. Install it with: {INSTALL_HINT}") from exc
    if hasattr(lib, "zz_feature_map"):          # Qiskit >= 2.1 (class is deprecated)
        fm = lib.zz_feature_map(n_qubits, reps=reps, entanglement=entanglement)
    else:
        fm = lib.ZZFeatureMap(n_qubits, reps=reps, entanglement=entanglement)
    return fm, Statevector


def scale_to_angles(X: np.ndarray) -> np.ndarray:
    """Min-max scale each column to [0, pi] (the feature-map rotation angles)."""
    X = np.asarray(X, dtype=float)
    lo, hi = X.min(axis=0), X.max(axis=0)
    if np.any(hi - lo <= 0):
        raise ValueError("A feature is constant in the sample; cannot scale it to angles.")
    return (X - lo) / (hi - lo) * np.pi


def statevectors(angles: np.ndarray, reps: int = 1, entanglement: str = "linear") -> np.ndarray:
    """Return an (m, 2**q) complex array; row i is |psi(x_i)>."""
    q = angles.shape[1]
    fm, Statevector = _load_feature_map(q, reps, entanglement)
    states = []
    for row in angles:
        bound = fm.assign_parameters(list(row))
        states.append(np.asarray(Statevector(bound).data, dtype=complex))
    return np.vstack(states)


def fidelity_kernel(states: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """``K_ij = |<psi_i|psi_j>|^2`` with numerical sanity checks."""
    norms = np.linalg.norm(states, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-8):
        raise KernelError("Statevectors are not normalised.")
    K = np.abs(states.conj() @ states.T) ** 2
    K = (K + K.T) / 2.0
    if not np.isfinite(K).all():
        raise KernelError("Kernel contains non-finite values.")
    if K.min() < -tol or K.max() > 1 + tol:
        raise KernelError("Kernel entries must lie in [0, 1].")
    if not np.allclose(np.diag(K), 1.0, atol=1e-8):
        raise KernelError("Kernel diagonal must be 1.")
    if np.linalg.eigvalsh(K).min() < -tol:
        raise KernelError("Fidelity kernel should be positive semi-definite.")
    return K


def center_kernel(K: np.ndarray) -> np.ndarray:
    """``K_c = H K H`` with ``H = I - 11^T / m`` (centres data in feature space)."""
    m = K.shape[0]
    H = np.eye(m) - np.ones((m, m)) / m
    Kc = H @ K @ H
    return (Kc + Kc.T) / 2.0


def kernel_pca_from_kernel(K: np.ndarray, n_components: int = 2, tol: float = 1e-8):
    """Eigen-decompose the centred kernel. Returns (scores, eigenvalues, fractions, Kc)."""
    Kc = center_kernel(K)
    if not np.allclose(Kc.sum(axis=0), 0.0, atol=1e-8):
        raise KernelError("Centred kernel rows/columns should sum to zero.")
    w, V = np.linalg.eigh(Kc)
    order = np.argsort(w)[::-1]
    w, V = w[order], V[:, order]
    if w.min() < -tol:
        raise KernelError(f"Centred kernel has a significantly negative eigenvalue ({w.min():.3e}).")
    w = np.clip(w, 0.0, None)
    if w.sum() <= 0:
        raise KernelError("Centred kernel is all zeros; all samples map to the same state.")
    rank = int(np.count_nonzero(w > tol * w[0]))
    if rank < n_components:
        raise KernelError(f"Centred kernel has rank {rank} < {n_components} requested components.")
    V = V[:, :n_components]
    for j in range(n_components):                    # deterministic sign
        if V[np.argmax(np.abs(V[:, j])), j] < 0:
            V[:, j] *= -1
    scores = V * np.sqrt(w[:n_components])
    return scores, w, w / w.sum(), Kc


def run_quantum_kernel_pca(X2: np.ndarray, n_components: int = 2, reps: int = 1,
                           entanglement: str = "linear") -> QuantumKPCAResult:
    """Quantum-kernel PCA on an (m, 2) feature array (m >= 3), 2 qubits."""
    X2 = np.asarray(X2, dtype=float)
    if X2.ndim != 2 or X2.shape[1] != 2:
        raise ValueError(f"Quantum branch uses exactly 2 features / 2 qubits; got {X2.shape}.")
    if X2.shape[0] < 3:
        raise ValueError("Need at least 3 samples for kernel PCA.")
    if not np.isfinite(X2).all():
        raise ValueError("Features contain non-finite values.")
    angles = scale_to_angles(X2)
    states = statevectors(angles, reps, entanglement)
    K = fidelity_kernel(states)
    scores, w, frac, Kc = kernel_pca_from_kernel(K, n_components)
    return QuantumKPCAResult(scores, w, frac, K, Kc, angles, angles.shape[1])
