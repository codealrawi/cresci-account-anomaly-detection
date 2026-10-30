"""Leak-free repeated-CV evaluation on Cresci-2017 features (plan 06, M3-M8).

Design
  * dataset: all genuine accounts that have tweets in the corpus + a stratified random draw of
    spambot/fake-follower accounts so that prevalence = 8 % (as in the article); several independent draws.
  * per draw: R repeats x 5-fold stratified CV. mu, sigma, V_k, hyper-parameters (inner 3-fold CV on the
    TRAIN part only) and the ECDF used for thresholds come from the training fold only.
  * every method is converted to u = ECDF_train(score) so "threshold = train percentile q" is the same
    rule for all methods (q = 95 as in the article, and q = 92 = 100(1-pi)).
Run: python -I evaluate.py <features.csv> <out_dir> [n_draws] [n_repeats]
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time, itertools, platform
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import svdad

FEATS = [f"p{i}" for i in range(1, 10)]
METHODS = ("svd", "if", "ocsvm", "lof")
VARIANTS = ("raw", "log")
GRIDS = {
    "svd": [0, 1, 2, 3, 4, 5, 6],                                   # 0 = k by the 90 % rule
    "if": [(100, 128), (100, 256), (100, 512), (300, 128), (300, 256), (300, 512)],
    "ocsvm": [(g, nu) for g in (0.25, 1.0, 4.0) for nu in (0.05, 0.1, 0.2)],
    "lof": [10, 20, 50, 100],
}
DEFAULTS = {"svd": 0, "if": (100, 256), "ocsvm": (1.0, 0.08), "lof": 20}   # article's untuned settings
QS = (95, 92)
SEED0 = 20261021


def make_model(method, variant, param, seed):
    if method == "svd":
        return svdad.SVDResidual(k=param if param else None, variant=variant)
    return _Wrapped(method, variant, param, seed)


class _Wrapped:
    def __init__(self, method, variant, param, seed):
        self.m, self.v, self.p, self.seed = method, variant, param, seed

    def fit(self, X):
        X = svdad.preprocess(X, self.v)
        self.mu_, self.sd_ = svdad.fit_standardizer(X)
        Z = (X - self.mu_) / self.sd_
        n = Z.shape[1]
        if self.m == "if":
            ne, ms = self.p
            self.est_ = IsolationForest(n_estimators=ne, max_samples=min(ms, len(Z)), random_state=self.seed).fit(Z)
        elif self.m == "ocsvm":
            gm, nu = self.p
            self.est_ = OneClassSVM(kernel="rbf", gamma=gm / n, nu=nu).fit(Z)
        elif self.m == "lof":
            self.est_ = LocalOutlierFactor(n_neighbors=min(self.p, len(Z) - 1), novelty=True).fit(Z)
        return self

    def score(self, X):
        Z = (svdad.preprocess(X, self.v) - self.mu_) / self.sd_
        if self.m == "if":
            return -self.est_.score_samples(Z)
        return -self.est_.decision_function(Z)


def select_param(method, variant, Xtr, ytr, seed):
    """Inner 3-fold CV on the TRAIN part only; criterion = mean validation AUC."""
    skf = StratifiedKFold(3, shuffle=True, random_state=seed)
    best, best_auc = None, -1
    for p in GRIDS[method]:
        aucs = []
        for a, b in skf.split(Xtr, ytr):
            s = make_model(method, variant, p, seed).fit(Xtr[a]).score(Xtr[b])
            aucs.append(roc_auc_score(ytr[b], s))
        if np.mean(aucs) > best_auc + 1e-12:
            best, best_auc = p, float(np.mean(aucs))
    return best


def fold_scores(method, variant, Xtr, ytr, Xte, seed, tuned=False):
    """Returns (u_test, info). u = ECDF of the train scores evaluated at test scores."""
    p = select_param(method, variant, Xtr, ytr, seed) if tuned else DEFAULTS[method]
    mdl = make_model(method, variant, p, seed).fit(Xtr)
    u = svdad.ECDF(mdl.score(Xtr))(mdl.score(Xte))
    return u, {"param": p, "k": getattr(mdl, "k_", None)}


def metrics(y, u):
    out = {"auc": roc_auc_score(y, u), "ap": average_precision_score(y, u)}
    fpr, tpr, _ = roc_curve(y, u)
    for f in (0.01, 0.05):
        out[f"tpr@fpr{int(f*100)}"] = float(tpr[fpr <= f].max())
    pi = y.mean()
    for q in QS:
        pred = u > q / 100
        tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        tpr_, fpr_ = tp / y.sum(), fp / (y == 0).sum()
        prec = tp / (tp + fp) if tp + fp else np.nan
        if tp + fp:  # M2: precision must agree with pi, TPR, FPR
            assert abs(prec - svdad.precision_from_rates(pi, tpr_, fpr_)) < 1e-9
        out.update({f"flag_q{q}": float(pred.mean()), f"tpr_q{q}": tpr_, f"fpr_q{q}": fpr_, f"prec_q{q}": prec,
                    f"f1_q{q}": 2 * tp / (2 * tp + fp + (y.sum() - tp)) if tp else 0.0})
    return out


def build_draw(df, draw, prev=0.08):
    rng = np.random.default_rng(SEED0 + draw)
    g = df[df.label == 0]
    npos = int(round(prev / (1 - prev) * len(g)))
    pos = df[df.label == 1]; cnt = pos.group.value_counts()
    raw = cnt / cnt.sum() * npos; take = np.floor(raw).astype(int)
    for gname in (raw - take).sort_values(ascending=False).index[: npos - take.sum()]:
        take[gname] += 1
    parts = [g] + [pos[pos.group == k].sample(int(v), random_state=int(rng.integers(1 << 31))) for k, v in take.items()]
    d = pd.concat(parts).sample(frac=1.0, random_state=int(rng.integers(1 << 31))).reset_index(drop=True)
    return d


def run_unit(df_draw, draw, rep):
    X = df_draw[FEATS].to_numpy(float); y = df_draw.label.to_numpy(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED0 * 7 + draw * 100 + rep)
    cfgs = list(itertools.product(METHODS, VARIANTS, ("default", "tuned")))
    oof = {c: np.full(len(y), np.nan) for c in cfgs}
    chosen = []
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        for c in cfgs:
            m, v, mode = c
            seed = SEED0 + draw * 1000 + rep * 10 + fi
            u, info = fold_scores(m, v, X[tr], y[tr], X[te], seed, tuned=(mode == "tuned"))
            oof[c][te] = u
            chosen.append(dict(draw=draw, rep=rep, fold=fi, method=m, variant=v, mode=mode, param=str(info["param"]), k=info["k"]))
    rows = []
    for c, u in oof.items():
        assert not np.isnan(u).any()
        rows.append(dict(draw=draw, rep=rep, method=c[0], variant=c[1], mode=c[2], **metrics(y, u)))
    return rows, chosen, {c: u for c, u in oof.items()}


def boot_auc(y, scores_by_cfg, base, B=500, seed=0):
    """Paired account-level bootstrap of AUC and of AUC differences vs `base`."""
    rng = np.random.default_rng(seed); n = len(y); res = {c: [] for c in scores_by_cfg}
    for _ in range(B):
        i = rng.integers(0, n, n)
        if y[i].sum() in (0, n):
            continue
        for c, s in scores_by_cfg.items():
            res[c].append(roc_auc_score(y[i], s[i]))
    out = []
    for c in scores_by_cfg:
        a = np.array(res[c]); b = np.array(res[base]); d = b - a if c != base else np.zeros_like(a)
        out.append(dict(config="|".join(c), auc_lo=np.percentile(a, 2.5), auc_hi=np.percentile(a, 97.5),
                        diff_base_minus_cfg_lo=np.percentile(d, 2.5), diff_base_minus_cfg_hi=np.percentile(d, 97.5)))
    return out


def timing(df_draw, reps=10):
    X = df_draw[FEATS].to_numpy(float); n = int(0.8 * len(X)); Xtr = X[:n]; out = []
    for m in METHODS:
        fit_t, sc_t = [], []
        for _ in range(reps):
            t = time.perf_counter(); mdl = make_model(m, "raw", DEFAULTS[m], 0).fit(Xtr); fit_t.append(time.perf_counter() - t)
            t = time.perf_counter(); mdl.score(X[n:]); sc_t.append(time.perf_counter() - t)
        out.append(dict(method=m, n_train=n, fit_median_s=np.median(fit_t), fit_min_s=min(fit_t), fit_max_s=max(fit_t), score_median_s=np.median(sc_t)))
    return out


if __name__ == "__main__":
    feats, out_dir = sys.argv[1], sys.argv[2]
    n_draws = int(sys.argv[3]) if len(sys.argv) > 3 else 5
    n_rep = int(sys.argv[4]) if len(sys.argv) > 4 else 5
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(feats, dtype={"user_id": str})
    draws = [build_draw(df, d) for d in range(n_draws)]
    jobs = [(d, r) for d in range(n_draws) for r in range(n_rep)]
    t0 = time.time()
    res = Parallel(n_jobs=12, verbose=5)(delayed(run_unit)(draws[d], d, r) for d, r in jobs)
    rows = [x for r in res for x in r[0]]; chosen = [x for r in res for x in r[1]]
    pd.DataFrame(rows).to_csv(f"{out_dir}/units_metrics.csv", index=False)
    pd.DataFrame(chosen).to_csv(f"{out_dir}/chosen_params.csv", index=False)
    # repeat-averaged OOF scores per draw -> paired bootstrap and subclass analysis
    boots, sub = [], []
    for d in range(n_draws):
        y = draws[d].label.to_numpy(int); grp = draws[d].group.to_numpy()
        avg = {}
        for c in res[0][2]:
            avg[c] = np.mean([res[d * n_rep + r][2][c] for r in range(n_rep)], axis=0)
        base = ("svd", "raw", "tuned")
        for b in boot_auc(y, {c: avg[c] for c in avg if c[2] == "tuned" and c[1] == "raw"}, base, seed=d):
            boots.append(dict(draw=d, **b))
        for c, u in avg.items():
            for g in np.unique(grp[y == 1]):
                m = (y == 0) | (grp == g)
                sub.append(dict(draw=d, config="|".join(c), subclass=g, n_pos=int((grp == g).sum()),
                                auc_vs_genuine=roc_auc_score(y[m], u[m]), recall_q95=float((u[(grp == g)] > 0.95).mean())))
    pd.DataFrame(boots).to_csv(f"{out_dir}/paired_bootstrap.csv", index=False)
    pd.DataFrame(sub).to_csv(f"{out_dir}/subclass.csv", index=False)
    pd.DataFrame(timing(draws[0])).to_csv(f"{out_dir}/timing.csv", index=False)
    import sklearn, scipy
    cfg = dict(seed0=SEED0, n_draws=n_draws, n_repeats=n_rep, folds=5, inner_folds=3, prevalence=0.08, qs=QS, grids={k: [str(x) for x in v] for k, v in GRIDS.items()},
               defaults={k: str(v) for k, v in DEFAULTS.items()}, n_genuine=int((df.label == 0).sum()), n_pool_pos=int((df.label == 1).sum()),
               draw_size=[len(x) for x in draws], versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__),
               cpu=platform.processor(), elapsed_s=time.time() - t0)
    json.dump(cfg, open(f"{out_dir}/config.json", "w"), indent=1, ensure_ascii=False)
    print("done", cfg["elapsed_s"])
