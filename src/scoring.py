"""Stage 31: Q (SPE), Hotelling-type T^2 and pre-specified Q+T^2 combinations.  See 02_method_specification.md.

All statistics are higher = more anomalous.  Everything is fitted on the TRAIN part only.
"""
import numpy as np

HEAVY = (0, 1, 7)            # p1, p2, p8 : log1p in variants 'log' and 'robust'
K_GRID = (1, 2, 3, 4, 5, 6)
RHO_GRID = (0.0, 0.1, 1.0)   # ridge, in units of the mean eigenvalue
W_GRID = (0.25, 0.5, 0.75)
KINDS = ("Q", "T2", "Cmax", "Cmean", "Cfisher", "Cnorm", "Cw")
TINY = 1e-12


class Prep:
    """variant: raw (z-score), log (log1p on p1,p2,p8 + z-score), robust (log1p + median/1.4826MAD, clip +-10, re-centred)."""

    def __init__(self, variant="raw", cols=tuple(range(9))):
        self.variant, self.cols = variant, list(cols)

    def _base(self, X):
        B = np.asarray(X, float)[:, self.cols].copy()
        if self.variant in ("log", "robust"):
            for j, c in enumerate(self.cols):
                if c in HEAVY:
                    B[:, j] = np.log1p(np.maximum(B[:, j], 0.0))
        return B

    def fit(self, X):
        B = self._base(X)
        if self.variant in ("raw", "log"):
            self.c_ = B.mean(0); s = B.std(0); self.s_ = np.where(s == 0, 1.0, s)
        else:
            self.med_ = np.median(B, 0); mad = 1.4826 * np.median(np.abs(B - self.med_), 0); sd = B.std(0)
            self.s_ = np.where(mad > 0, mad, np.where(sd > 0, sd, 1.0))
            self.c_ = np.clip((B - self.med_) / self.s_, -10, 10).mean(0)
        return self

    def transform(self, X):
        B = self._base(X)
        if self.variant in ("raw", "log"):
            return (B - self.c_) / self.s_
        return np.clip((B - self.med_) / self.s_, -10, 10) - self.c_


class Subspace:
    """SVD of the (train) standardised matrix; exposes coordinates so that any (k, rho) is cheap."""

    def __init__(self, variant="raw", cols=tuple(range(9))):
        self.variant, self.cols = variant, cols

    def fit(self, X):
        self.prep_ = Prep(self.variant, self.cols).fit(X)
        Z = self.prep_.transform(X)
        _, S, Vt = np.linalg.svd(Z, full_matrices=False)
        self.Vt_ = Vt; self.lam_ = S ** 2 / len(Z); self.lbar_ = float(self.lam_.mean())
        self.d_ = Z.shape[1]
        return self

    def coords(self, X):
        Z = self.prep_.transform(X)
        return Z @ self.Vt_.T, (Z ** 2).sum(1)

    def QT(self, P, tot, k, rho):
        k = min(k, self.d_ - 1)
        p2 = P[:, :k] ** 2
        Q = np.maximum(tot - p2.sum(1), 0.0)                      # squared norm of the residual (SPE)
        den = np.maximum(self.lam_[:k] + rho * self.lbar_, TINY * self.lbar_)
        T2 = (p2 / den).sum(1)                                     # sum_j (v_j^T z)^2 / (lambda_j + rho * mean lambda)
        return Q, T2


def ecdf(train, x):
    return np.searchsorted(np.sort(train), x, side="right") / len(train)


def statistic(kind, Qtr, Ttr, Qte, Tte, w=0.5):
    """Returns (train statistic, test statistic) for one pre-specified score."""
    if kind == "Q":
        return Qtr, Qte
    if kind == "T2":
        return Ttr, Tte
    uQtr, uTtr = ecdf(Qtr, Qtr), ecdf(Ttr, Ttr)
    uQte, uTte = ecdf(Qtr, Qte), ecdf(Ttr, Tte)
    eps = 1.0 / (len(Qtr) + 1)
    if kind == "Cmax":
        return np.maximum(uQtr, uTtr), np.maximum(uQte, uTte)
    if kind == "Cmean":
        return 0.5 * (uQtr + uTtr), 0.5 * (uQte + uTte)
    if kind == "Cw":
        return w * uQtr + (1 - w) * uTtr, w * uQte + (1 - w) * uTte
    if kind == "Cfisher":
        f = lambda u: -np.log(1 - u + eps)
        return f(uQtr) + f(uTtr), f(uQte) + f(uTte)
    if kind == "Cnorm":
        mq, mt = max(Qtr.mean(), TINY), max(Ttr.mean(), TINY)
        return Qtr / mq + Ttr / mt, Qte / mq + Tte / mt
    raise ValueError(kind)


def grid_for(kind):
    if kind == "Q":
        return [(k, 0.0, None) for k in K_GRID]
    if kind == "Cw":
        return [(k, r, w) for k in K_GRID for r in RHO_GRID for w in W_GRID]
    return [(k, r, None) for k in K_GRID for r in RHO_GRID]
