import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import scoring as sc


def data(seed=0, m=400, d=9):
    r = np.random.default_rng(seed); return r.standard_normal((m, d)) @ r.standard_normal((d, d)) * [1, 5, 1, 1, 2, 1, 1, 30, 1] + 3


def test_q_plus_projection_energy_is_total():
    X = data(); s = sc.Subspace("raw").fit(X); P, tot = s.coords(X)
    for k in (1, 3, 5):
        Q, T = s.QT(P, tot, k, 0.0)
        assert np.allclose(Q + (P[:, :k] ** 2).sum(1), tot)
        assert (Q >= 0).all()


def test_t2_equals_mahalanobis_on_retained_components():
    X = data(1); s = sc.Subspace("raw").fit(X); P, tot = s.coords(X); Z = s.prep_.transform(X)
    k = 4; C = np.cov(Z.T, bias=True); w, V = np.linalg.eigh(C); idx = np.argsort(w)[::-1][:k]
    ref = ((Z @ V[:, idx]) ** 2 / w[idx]).sum(1)
    assert np.allclose(s.QT(P, tot, k, 0.0)[1], ref, rtol=1e-8)


def test_mean_t2_on_train_is_k():
    X = data(2); s = sc.Subspace("raw").fit(X); P, tot = s.coords(X)
    assert np.isclose(s.QT(P, tot, 3, 0.0)[1].mean(), 3.0)


def test_ridge_shrinks_t2_and_handles_degenerate_columns():
    X = data(3); X[:, 2] = 1.0                                    # constant column -> sd=0 handled
    s = sc.Subspace("raw").fit(X); P, tot = s.coords(X)
    a = s.QT(P, tot, 6, 0.0)[1]; b = s.QT(P, tot, 6, 1.0)[1]
    assert np.isfinite(a).all() and (b <= a + 1e-9).all()


def test_statistics_use_train_only_and_orientation():
    r = np.random.default_rng(0); Qtr = r.random(200); Ttr = r.random(200); Qte = np.array([2.0, -1.0]); Tte = np.array([2.0, -1.0])
    for kind in sc.KINDS:
        st, se = sc.statistic(kind, Qtr, Ttr, Qte, Tte)
        assert se[0] >= se[1]                                     # larger raw scores -> larger (more anomalous) statistic
    _, se1 = sc.statistic("Cmean", Qtr, Ttr, Qte, Tte); _, se2 = sc.statistic("Cmean", Qtr, Ttr, Qte[:1], Tte[:1])
    assert se1[0] == se2[0]                                       # a test point does not depend on other test points


def test_robust_prep_degenerate():
    X = data(4); X[:, 0] = np.where(np.arange(len(X)) % 2 == 0, 0.0, 0.0)  # MAD = 0 and sd = 0
    Z = sc.Prep("robust").fit(X).transform(X); assert np.isfinite(Z).all()
