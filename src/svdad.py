"""Independent re-implementation of the article's SVD-residual detector (formulas (1)-(6)) in pure NumPy.

Written from the article's formulas, not copied from the authors' reference code.
All statistics (mu, sigma, V_k, threshold) come from the training matrix only.
"""
import numpy as np

HEAVY_TAIL = (0, 1, 7)  # p1, p2, p8 (0-based) get log1p in the 'log' preprocessing variant


def preprocess(X, variant):
    X = np.asarray(X, dtype=float).copy()
    if variant == "log":
        X[:, HEAVY_TAIL] = np.log1p(np.maximum(X[:, HEAVY_TAIL], 0.0))
    elif variant != "raw":
        raise ValueError(variant)
    return X


def fit_standardizer(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)
    sd = np.where(sd == 0, 1.0, sd)
    return mu, sd


def select_k(S, ev=0.90):
    """Formula (4): smallest K with sum_{i<=K} sigma_i^2 / sum sigma_i^2 >= ev."""
    cum = np.cumsum(S ** 2) / np.sum(S ** 2)
    return int(np.searchsorted(cum, ev - 1e-12) + 1)


def residual_norm(Z, V):
    """Formula (5): e_i = || z_i - V V^T z_i ||_2 for orthonormal V (n x k)."""
    return np.linalg.norm(Z - (Z @ V) @ V.T, axis=1)


class SVDResidual:
    """fit(X) -> unsupervised; score(X) -> e_i (higher = more anomalous)."""

    def __init__(self, k=None, ev=0.90, variant="raw"):
        self.k, self.ev, self.variant = k, ev, variant

    def fit(self, X):
        X = preprocess(X, self.variant)
        self.mu_, self.sd_ = fit_standardizer(X)
        Z = (X - self.mu_) / self.sd_
        _, S, Vt = np.linalg.svd(Z, full_matrices=False)
        self.S_ = S
        self.k_ = max(1, min(self.k if self.k else select_k(S, self.ev), Z.shape[1]))
        self.V_ = Vt[: self.k_].T
        self.explained_ = float((S[: self.k_] ** 2).sum() / (S ** 2).sum())
        return self

    def score(self, X):
        Z = (preprocess(X, self.variant) - self.mu_) / self.sd_
        return residual_norm(Z, self.V_)


class ECDF:
    """Maps a score to the fraction of TRAIN scores <= it (threshold at percentile q <=> u > q/100)."""

    def __init__(self, train_scores):
        self.s_ = np.sort(np.asarray(train_scores, dtype=float))

    def __call__(self, x):
        return np.searchsorted(self.s_, np.asarray(x, dtype=float), side="right") / len(self.s_)


def precision_from_rates(pi, tpr, fpr):
    return pi * tpr / (pi * tpr + (1 - pi) * fpr)
