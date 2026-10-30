# -*- coding: utf-8 -*-
"""Stage 43: ONE computation - the 'article A configuration' of the SVD-residual detector on the official Cresci-2017 corpus (variant B features).
Fixed protocol (see 02_protocol_preregistered.md): z-score (train statistics, ddof=0) -> SVD -> k = smallest K with cumulative sigma^2 share >= 0.90 (no tuning, no labels)
-> e = ||z - V_k V_k^T z||_2 -> threshold tau = np.percentile(e_train, 95) -> flag e > tau.  No hyperparameter grid, no other model, no q other than 95.
Evaluation design = Stage 36 design (same 10 draws, 10 repeats, 5 stratified outer folds, same seeds), metrics aggregated A-style (per test fold, then averaged)
and, for comparability with Stage 36 tables, pooled out-of-fold (u = train-ECDF of e).
Reuses src/evaluate.py (build_draw, SEED0) and src/svdad.py (SVDResidual, ECDF).  (original: run_stage43.py; only path handling changed)
Usage: python -B scripts/run_explained_variance_rule.py <features.csv> <out_dir>"""
import os, sys, json, time, platform
sys.dont_write_bytecode = True
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"): os.environ.setdefault(_v, "1")
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, average_precision_score
sys.stdout.reconfigure(encoding="utf-8")
feat_path, out = sys.argv[1:3]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import evaluate as ev            # build_draw, SEED0, FEATS (read-only reuse)
import svdad
FEATS = ev.FEATS; SEED0 = ev.SEED0; Q = 95.0; EV = 0.90; N_DRAWS, N_REPS = 10, 10
df = pd.read_csv(feat_path, dtype={"user_id": str})
os.makedirs(out, exist_ok=True)
fold_rows, unit_rows = [], []
t0 = time.time()
for d in range(N_DRAWS):
    D = ev.build_draw(df, d); X = D[FEATS].to_numpy(float); y = D.label.to_numpy(int)
    u_oof_all = np.zeros((N_REPS, len(y)))
    for r in range(N_REPS):
        skf = StratifiedKFold(5, shuffle=True, random_state=SEED0 * 7 + d * 100 + r)
        u_oof = np.full(len(y), np.nan); fm = []
        for fi, (tr, te) in enumerate(skf.split(X, y)):
            det = svdad.SVDResidual(k=None, ev=EV, variant="raw").fit(X[tr])      # k by the 90 % rule, from TRAIN singular values only
            e_tr = det.score(X[tr]); e_te = det.score(X[te]); yt = y[te]
            tau = float(np.percentile(e_tr, Q))                                    # as in the reference code of article A (np.percentile, linear interpolation)
            pred = e_te > tau
            tp = int((pred & (yt == 1)).sum()); fp = int((pred & (yt == 0)).sum()); fn = int(((~pred) & (yt == 1)).sum()); tn = int(((~pred) & (yt == 0)).sum())
            prec = tp / (tp + fp) if tp + fp else 0.0; tpr = tp / (tp + fn); fpr = fp / (fp + tn)
            f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
            row = dict(draw=d, rep=r, fold=fi, n_train=len(tr), n_test=len(te), n_pos_test=int(yt.sum()), k=det.k_, explained=det.explained_,
                       sv2_share_1=float(det.S_[0] ** 2 / (det.S_ ** 2).sum()), sv2_share_2=float(det.S_[1] ** 2 / (det.S_ ** 2).sum()), sv2_share_3=float(det.S_[2] ** 2 / (det.S_ ** 2).sum()),
                       tau=tau, n_train_above_tau=int((e_tr > tau).sum()), mean_e2_train=float((e_tr ** 2).mean()), identity_rhs=float(len(FEATS) * (1 - det.explained_)),
                       auc=roc_auc_score(yt, e_te), ap=average_precision_score(yt, e_te), tp=tp, fp=fp, fn=fn, tn=tn, flagged_test=tp + fp, flagged_share_test=(tp + fp) / len(te),
                       tpr=tpr, fpr=fpr, precision=prec, f1=f1, mean_e_pos=float(e_te[yt == 1].mean()), sd_e_pos=float(e_te[yt == 1].std(ddof=1)),
                       mean_e_neg=float(e_te[yt == 0].mean()), sd_e_neg=float(e_te[yt == 0].std(ddof=1)))
            fold_rows.append(row); fm.append(row)
            u_oof[te] = svdad.ECDF(e_tr)(e_te)
        u_oof_all[r] = u_oof
        # pooled out-of-fold metrics (Stage 36 convention: u = ECDF of train scores, flag u > 0.95)
        pred = u_oof > Q / 100; tp = int((pred & (y == 1)).sum()); fp = int((pred & (y == 0)).sum())
        unit_rows.append(dict(draw=d, rep=r, auc_foldmean=np.mean([f["auc"] for f in fm]), ap_foldmean=np.mean([f["ap"] for f in fm]),
                              tpr_foldmean=np.mean([f["tpr"] for f in fm]), fpr_foldmean=np.mean([f["fpr"] for f in fm]), precision_foldmean=np.mean([f["precision"] for f in fm]),
                              f1_foldmean=np.mean([f["f1"] for f in fm]), k_mean=np.mean([f["k"] for f in fm]), explained_mean=np.mean([f["explained"] for f in fm]),
                              auc_pooled=roc_auc_score(y, u_oof), ap_pooled=average_precision_score(y, u_oof), tpr_pooled=tp / y.sum(), fpr_pooled=fp / (y == 0).sum(),
                              precision_pooled=tp / (tp + fp) if tp + fp else 0.0, flagged_share_pooled=float(pred.mean())))
    np.savez_compressed(os.path.join(out, f"oof_u_draw{d}.npz"), y=y, grp=D.group.to_numpy(str), user_id=D.user_id.to_numpy(str), u=u_oof_all)
    print("draw", d, "done", round(time.time() - t0, 1), "s", flush=True)
pd.DataFrame(fold_rows).to_csv(os.path.join(out, "fold_level.csv"), index=False)
pd.DataFrame(unit_rows).to_csv(os.path.join(out, "unit_level.csv"), index=False)
import sklearn, scipy
json.dump(dict(stage="43", n_draws=N_DRAWS, n_repeats=N_REPS, folds=5, seed0=SEED0, q=Q, explained_variance_rule=EV, variant="raw (z-score only)", features=feat_path,
               versions=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__, scipy=scipy.__version__),
               elapsed_s=time.time() - t0, n_fold_rows=len(fold_rows), n_unit_rows=len(unit_rows)), open(os.path.join(out, "run_record.json"), "w", encoding="utf-8"), indent=1)
print("finished", len(fold_rows), len(unit_rows))
