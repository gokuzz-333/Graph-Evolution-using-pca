"""Beginner-friendly standardisation and PCA using NumPy only.

PCA steps: standardise when requested, centre columns, form a covariance matrix,
eigendecompose it, sort directions by variance, and project rows onto the top PCs.
"""
from __future__ import annotations

import numpy as np


class StandardScaler:
    """Scale each feature to zero mean and unit population standard deviation."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] == 0:
            raise ValueError("Scaler input must be a non-empty 2-D array.")
        if not np.isfinite(X).all():
            raise ValueError("Scaler input must contain only finite values.")
        self.mean_ = np.mean(X, axis=0)
        # ddof=0 matches the previous sklearn StandardScaler preprocessing.
        self.std_ = np.std(X, axis=0, ddof=0)
        self.std_[self.std_ < 1e-9] = 1.0
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise ValueError("Call fit before transform.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.mean_):
            raise ValueError("Transform input must have the same number of columns as fit data.")
        if not np.isfinite(X).all():
            raise ValueError("Transform input must contain only finite values.")
        return (X - self.mean_) / self.std_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


class CustomPCA:
    """PCA with explicit centring, covariance, eigenvectors, and projection.

    When there are fewer features than samples, diagonalise the feature
    covariance matrix Xc.T @ Xc. For wide graph matrices, diagonalise the
    smaller sample Gram matrix Xc @ Xc.T and recover feature directions.
    Both matrices have the same non-zero eigenvalues, so this is the same PCA
    while avoiding an unnecessarily large covariance matrix.
    """

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ratio_ = None
        self.covariance_matrix_ = None
        self.eigen_matrix_type_ = None

    def fit(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[0] < 2 or X.shape[1] < 1:
            raise ValueError("PCA needs a 2-D array with at least 2 rows and 1 column.")
        if not np.isfinite(X).all():
            raise ValueError("PCA input must contain only finite values.")
        if not isinstance(self.n_components, (int, np.integer)) or self.n_components < 1:
            raise ValueError("n_components must be a positive integer.")
        n_samples, n_features = X.shape
        if self.n_components > min(n_samples, n_features):
            raise ValueError("n_components cannot exceed the smaller input dimension.")

        # Centering removes each feature's baseline before measuring variation.
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_

        # Covariance uses n - 1 so the eigenvalues are sample variances.
        if n_features <= n_samples:
            self.covariance_matrix_ = (X_centered.T @ X_centered) / (n_samples - 1)
            self.eigen_matrix_type_ = "feature_covariance"
            eigenvalues, eigenvectors = np.linalg.eigh(self.covariance_matrix_)
            directions = eigenvectors
        else:
            # The dual covariance/Gram matrix is smaller for wide inputs.
            gram_matrix = (X_centered @ X_centered.T) / (n_samples - 1)
            self.covariance_matrix_ = gram_matrix
            self.eigen_matrix_type_ = "sample_gram_matrix"
            eigenvalues, sample_directions = np.linalg.eigh(gram_matrix)
            # Convert sample-space eigenvectors into feature-space directions.
            safe_values = np.maximum(eigenvalues, 0.0)
            denominators = np.sqrt((n_samples - 1) * safe_values)
            directions = np.zeros((n_features, len(eigenvalues)), dtype=np.float64)
            positive = denominators > np.finfo(np.float64).eps
            directions[:, positive] = (
                X_centered.T @ sample_directions[:, positive]
            ) / denominators[positive]

        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        directions = directions[:, order]
        k = self.n_components
        self.explained_variance_ = eigenvalues[:k]
        self.components_ = directions[:, :k].T

        # Eigenvector signs are arbitrary; choose a stable sign for readable loadings.
        for component in self.components_:
            largest_loading = np.argmax(np.abs(component))
            if component[largest_loading] < 0:
                component *= -1

        total_variance = float(eigenvalues.sum())
        if total_variance > 0:
            self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        else:
            self.explained_variance_ratio_ = np.zeros(k, dtype=np.float64)
        self.cumulative_variance_ratio_ = np.cumsum(self.explained_variance_ratio_)
        return self

    def transform(self, X):
        if self.mean_ is None or self.components_ is None:
            raise ValueError("Call fit before transform.")
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.mean_):
            raise ValueError("Transform input must have the same number of columns as fit data.")
        if not np.isfinite(X).all():
            raise ValueError("PCA input must contain only finite values.")
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X):
        return self.fit(X).transform(X)
