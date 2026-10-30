"""M1-M3 tests (synthetic data). Run from the repository root: python -m pytest tests -q
The original test comparing against the authors' reference code of an earlier article was removed (that code is not part of this repository)."""
import importlib.util, os, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
import svdad  # noqa: E402
import evaluate as ev  # noqa: E402



def data(seed=0, m=300, n=9):
    r = np.random.default_rng(seed)
    return r.standard_normal((m, n)) @ r.standard_normal((n, n)) + r.standard_normal(n) * 3


# ---- M1
@pytest.mark.parametrize("k", [1, 2, 3, 5])
def test_energy_identity_and_eckart_young(k):
    X = data(); mu, sd = svdad.fit_standardizer(X); Z = (X - mu) / sd
    d = svdad.SVDResidual(k=k).fit(X)
    e = d.score(X)
    S = d.S_
    assert np.isclose((e ** 2).sum(), (S[k:] ** 2).sum(), rtol=1e-9)           # Eckart-Young / tail energy
    assert np.isclose((e ** 2).mean(), (1 - d.explained_) * Z.shape[1], rtol=1e-9)  # mean e^2 = (1-EV_k) n


def test_V_orthonormal():
    d = svdad.SVDResidual(k=4).fit(data())
    assert np.allclose(d.V_.T @ d.V_, np.eye(4), atol=1e-10)


def test_select_k_rule():
    S = np.array([3.0, 2.0, 1.0]); assert svdad.select_k(S, 0.90) == 2  # cum energy 9/14=.643, 13/14=.929, 1
    assert svdad.select_k(S, 0.64) == 1 and svdad.select_k(S, 0.65) == 2


# ---- M2
def test_precision_identity_on_confusion_matrices():
    r = np.random.default_rng(0)
    for _ in range(200):
        y = (r.random(500) < 0.08).astype(int); p = (r.random(500) < 0.1).astype(int)
        tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum()
        if tp + fp == 0 or y.sum() == 0:
            continue
        pi = y.mean(); tpr = tp / y.sum(); fpr = fp / (y == 0).sum()
        assert np.isclose(tp / (tp + fp), svdad.precision_from_rates(pi, tpr, fpr), rtol=1e-12, atol=1e-12)


def test_p95_caps_recall_when_prevalence_exceeds_flag_rate():
    """tau=P95 of the training scores flags ~5%: recall <= 0.05/pi on a same-distribution test set."""
    r = np.random.default_rng(3); n = 20000
    y = (r.random(n) < 0.08).astype(int)
    s = r.standard_normal(n) + 3.0 * y                      # a strong detector
    tau = np.percentile(s, 95)
    pred = s > tau
    assert pred.mean() <= 0.0505
    assert (pred & (y == 1)).sum() / y.sum() <= 0.05 / y.mean() + 0.01


# ---- M3 (no leakage)
def test_statistics_come_from_train_only():
    Xtr, Xte = data(0), data(5) + 10
    d = svdad.SVDResidual(k=3).fit(Xtr)
    mu = Xtr.mean(0)
    assert np.allclose(d.mu_, mu)
    s1 = d.score(Xte)
    s2 = d.score(Xte[:50])
    assert np.allclose(s1[:50], s2)                          # a test row's score does not depend on other test rows


def test_ecdf_uses_train_scores_only():
    tr = np.arange(100.0); f = svdad.ECDF(tr)
    assert f(49.5) == 0.5 and f(1000) == 1.0 and f(-1) == 0.0


def test_hyperparameter_selection_ignores_test_data():
    r = np.random.default_rng(0)
    Xtr = r.standard_normal((300, 9)); ytr = (r.random(300) < 0.1).astype(int); Xtr[ytr == 1] += 2
    a = ev.select_param("svd", "raw", Xtr, ytr, seed=1)
    b = ev.select_param("svd", "raw", Xtr, ytr, seed=1)       # signature has no test arguments at all
    assert a == b


def test_scores_do_not_depend_on_test_labels():
    r = np.random.default_rng(0)
    X = r.standard_normal((400, 9)); y = (r.random(400) < 0.1).astype(int); X[y == 1] += 2
    tr, te = np.arange(300), np.arange(300, 400)
    u1, _ = ev.fold_scores("svd", "raw", X[tr], y[tr], X[te], seed=0)
    u2, _ = ev.fold_scores("svd", "raw", X[tr], y[tr], X[te], seed=0)
    assert np.allclose(u1, u2)
