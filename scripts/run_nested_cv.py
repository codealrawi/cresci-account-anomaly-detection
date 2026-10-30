"""Stage 31 nested-CV runner (T3).  Outer folds / seeds are identical to pipeline/evaluate.py (stage M).
Usage: python -I scripts/run_nested_cv.py <features.csv> <out_dir> <n_draws> <n_repeats> <min_tweets> <with_baselines 0|1>
(original: run_stage31.py; only the path handling was changed for the public layout)
"""
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys, json, time, platform
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import evaluate as ev            # read-only reuse: build_draw, fold_scores, metrics, SEED0, FEATS
import scoring as sc

BASELINES = ("if", "lof", "ocsvm")
VARIANT = "raw"                 # primary (article-faithful z-score); others only in sensitivity


def tune_and_score(kind, subs_inner, inner_idx, ytr, sub_full, Ptr, tottr, Pte, totte):
    """Choose (k, rho, w) by mean inner-validation AUC (train labels only); return (u_test, params)."""
    best, best_auc = None, -1.0
    for (k, r, w) in sc.grid_for(kind):
        aucs = []
        for (sub, (a, b), Pa, ta, Pb, tb) in subs_inner:
            Qa, Ta = sub.QT(Pa, ta, k, r); Qb, Tb = sub.QT(Pb, tb, k, r)
            _, sb = sc.statistic(kind, Qa, Ta, Qb, Tb, w if w is not None else 0.5)
            aucs.append(roc_auc_score(ytr[b], sb))
        m = float(np.mean(aucs))
        if m > best_auc + 1e-12:
            best, best_auc = (k, r, w), m
    k, r, w = best
    Qtr, Ttr = sub_full.QT(Ptr, tottr, k, r); Qte, Tte = sub_full.QT(Pte, totte, k, r)
    st, se = sc.statistic(kind, Qtr, Ttr, Qte, Tte, w if w is not None else 0.5)
    return sc.ecdf(st, se), dict(k=k, rho=r, w=w, inner_auc=best_auc)


def run_unit(D, draw, rep, with_baselines):
    X = D[ev.FEATS].to_numpy(float); y = D.label.to_numpy(int)
    skf = StratifiedKFold(5, shuffle=True, random_state=ev.SEED0 * 7 + draw * 100 + rep)
    kinds = list(sc.KINDS) + (list(BASELINES) if with_baselines else [])
    oof = {k: np.full(len(y), np.nan) for k in kinds}
    frows, chosen = [], []
    for fi, (tr, te) in enumerate(skf.split(X, y)):
        seed = ev.SEED0 + draw * 1000 + rep * 10 + fi
        Xtr, ytr, Xte, yte = X[tr], y[tr], X[te], y[te]
        inner = StratifiedKFold(3, shuffle=True, random_state=seed)
        subs_inner = []
        for a, b in inner.split(Xtr, ytr):
            s = sc.Subspace(VARIANT).fit(Xtr[a]); Pa, ta = s.coords(Xtr[a]); Pb, tb = s.coords(Xtr[b])
            subs_inner.append((s, (a, b), Pa, ta, Pb, tb))
        full = sc.Subspace(VARIANT).fit(Xtr); Ptr, tottr = full.coords(Xtr); Pte, totte = full.coords(Xte)
        for kind in sc.KINDS:
            u, info = tune_and_score(kind, subs_inner, None, ytr, full, Ptr, tottr, Pte, totte)
            oof[kind][te] = u
            chosen.append(dict(draw=draw, rep=rep, fold=fi, method=kind, **info))
        if with_baselines:
            for m in BASELINES:
                u, info = ev.fold_scores(m, VARIANT, Xtr, ytr, Xte, seed, tuned=True)
                oof[m][te] = u
                chosen.append(dict(draw=draw, rep=rep, fold=fi, method=m, k=None, rho=None, w=None, inner_auc=np.nan, param=str(info["param"])))
        for kind in kinds:
            mt = ev.metrics(yte, oof[kind][te])
            frows.append(dict(draw=draw, rep=rep, fold=fi, method=kind, n_test=len(te), n_pos_test=int(yte.sum()), **mt))
    urows = []
    for kind in kinds:
        assert not np.isnan(oof[kind]).any()
        urows.append(dict(draw=draw, rep=rep, method=kind, **ev.metrics(y, oof[kind])))
    return frows, urows, chosen, oof


if __name__ == "__main__":
    features_csv, out_dir = sys.argv[1], sys.argv[2]
    n_draws, n_rep, min_tw, wb = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), bool(int(sys.argv[6]))
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(features_csv, dtype={"user_id": str})
    df = df[df.n_tweets >= min_tw].reset_index(drop=True)
    draws = [ev.build_draw(df, d) for d in range(n_draws)]
    jobs = [(d, r) for d in range(n_draws) for r in range(n_rep)]
    t0 = time.time()
    res = Parallel(n_jobs=int(os.environ.get("SVDAD_N_JOBS", "12")), verbose=5)(delayed(run_unit)(draws[d], d, r, wb) for d, r in jobs)
    pd.DataFrame([x for r in res for x in r[0]]).to_csv(f"{out_dir}/fold_metrics.csv", index=False)
    pd.DataFrame([x for r in res for x in r[1]]).to_csv(f"{out_dir}/unit_metrics.csv", index=False)
    pd.DataFrame([x for r in res for x in r[2]]).to_csv(f"{out_dir}/chosen_params.csv", index=False)
    for d in range(n_draws):                                   # every repeat's OOF scores are kept
        k0 = list(res[0][3].keys())
        np.savez_compressed(f"{out_dir}/oof_draw{d}.npz", y=draws[d].label.to_numpy(int), grp=draws[d].group.to_numpy(str),
                            user_id=draws[d].user_id.to_numpy(str), **{k: np.stack([res[d * n_rep + r][3][k] for r in range(n_rep)]) for k in k0})
    import sklearn, scipy
    json.dump(dict(stage="31", n_draws=n_draws, n_repeats=n_rep, min_tweets=min_tw, baselines=wb, variant=VARIANT, seed0=ev.SEED0,
                   k_grid=sc.K_GRID, rho_grid=sc.RHO_GRID, w_grid=sc.W_GRID, n_accounts=len(df), draw_size=[len(x) for x in draws],
                   versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__),
                   elapsed_s=time.time() - t0), open(f"{out_dir}/run_config.json", "w"), indent=1)
    print("done", time.time() - t0)
