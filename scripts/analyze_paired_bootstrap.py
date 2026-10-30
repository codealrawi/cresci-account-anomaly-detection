"""T4 statistics for a run directory.  python -I analyze.py <run_dir> [n_draws]
Hierarchical paired bootstrap: draws resampled with replacement x accounts resampled within draw (paired across methods).
"""
import sys, os, glob, json
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
sys.stdout.reconfigure(encoding="utf-8")
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 40)
run = sys.argv[1]; B1, B2 = 500, 2000
files = sorted(glob.glob(f"{run}/oof_draw*.npz"), key=lambda p: int(p.split("draw")[-1].split(".")[0]))
D = [np.load(f, allow_pickle=True) for f in files]
methods = [k for k in D[0].files if k not in ("y", "grp", "user_id")]
PAIRS = [("T2", "Q"), ("Cmean", "T2"), ("Cmean", "Q"), ("Cmax", "T2"), ("Cfisher", "T2"), ("Cnorm", "T2"), ("Cw", "T2"), ("Cw", "Q")]
for a in ("Q", "T2", "Cmean"):
    for b in ("if", "lof", "ocsvm"):
        if b in methods: PAIRS.append((a, b))

# ---- fold / unit level descriptives
fm = pd.read_csv(f"{run}/fold_metrics.csv"); um = pd.read_csv(f"{run}/unit_metrics.csv")
cols = ["auc", "ap", "tpr@fpr1", "tpr@fpr5"]
fold_tab = fm.groupby("method")[cols].agg(["mean", "std"]); fold_tab.columns = ["_".join(c) for c in fold_tab.columns]
ucols = cols + ["flag_q95", "tpr_q95", "fpr_q95", "prec_q95", "f1_q95", "flag_q92", "tpr_q92", "fpr_q92", "prec_q92", "f1_q92"]
unit_tab = um.groupby("method")[ucols].mean()
draw_sd = um.groupby(["method", "draw"])["auc"].mean().groupby("method").std().rename("auc_sd_between_draws")
order = [m for m in ["Q", "T2", "Cmax", "Cmean", "Cfisher", "Cnorm", "Cw", "if", "lof", "ocsvm"] if m in methods]
fold_tab.loc[order].round(4).to_csv(f"{run}/table_fold_level.csv"); unit_tab.loc[order].round(4).to_csv(f"{run}/table_unit_level.csv")
print("== FOLD-LEVEL (n=%d outer folds per method): mean and SD over folds" % (fm.method == order[0]).sum()); print(fold_tab.loc[order].round(3))
print("\n== UNIT-LEVEL pooled OOF (mean over 10 draws x 10 repeats)"); print(unit_tab.loc[order].join(draw_sd).round(3))
cp = pd.read_csv(f"{run}/chosen_params.csv")
chosen = cp[cp.method.isin(sc_ := ["Q", "T2", "Cmax", "Cmean", "Cfisher", "Cnorm", "Cw"])].groupby("method").agg(k_mean=("k", "mean"), rho_mean=("rho", "mean"), inner_auc=("inner_auc", "mean"))
chosen["k_mode"] = cp[cp.method.isin(sc_)].groupby("method")["k"].agg(lambda s: s.mode().iloc[0]); print("\n== CHOSEN PARAMS"); print(chosen.round(3)); chosen.round(3).to_csv(f"{run}/table_chosen_params.csv")
print("k distribution T2:", cp[cp.method == "T2"].k.value_counts().sort_index().to_dict(), " rho T2:", cp[cp.method == "T2"].rho.value_counts().sort_index().to_dict())

# ---- paired hierarchical bootstrap on repeat-averaged OOF scores
def stat(y, s, kind): return roc_auc_score(y, s) if kind == "auc" else average_precision_score(y, s)
rows = []; point = {}
for kind in ("auc", "ap"):
    boots = {p: np.zeros((len(D), B1)) for p in PAIRS}; pt = {p: np.zeros(len(D)) for p in PAIRS}
    for d, z in enumerate(D):
        y = z["y"]; n = len(y); avg = {m: z[m].mean(0) for m in methods}; rng = np.random.default_rng(1000 + d)
        idx = rng.integers(0, n, (B1, n)); ok = np.array([0 < y[i].sum() < n for i in idx])
        base = {m: np.array([stat(y[i], avg[m][i], kind) if o else np.nan for i, o in zip(idx, ok)]) for m in {x for p in PAIRS for x in p}}
        full = {m: stat(y, avg[m], kind) for m in base}
        for p in PAIRS:
            boots[p][d] = base[p[0]] - base[p[1]]; pt[p][d] = full[p[0]] - full[p[1]]
    rng = np.random.default_rng(7)
    for p in PAIRS:
        reps = np.empty(B2)
        for r in range(B2):
            ds = rng.integers(0, len(D), len(D)); bs = rng.integers(0, B1, len(D))
            v = boots[p][ds, bs]; reps[r] = np.nanmean(v)
        est = pt[p].mean(); lo, hi = np.percentile(reps, [2.5, 97.5])
        pval = min(1.0, 2 * min((reps <= 0).mean(), (reps >= 0).mean()) + 1.0 / B2)
        rows.append(dict(metric=kind, a=p[0], b=p[1], diff_mean_over_draws=est, ci_lo=lo, ci_hi=hi, p_two_sided=pval,
                         draws_a_gt_b=int((pt[p] > 0).sum()), n_draws=len(D)))
res = pd.DataFrame(rows)
# repeat-level / fold-level sign consistency
piv = um.pivot_table(index=["draw", "rep"], columns="method", values="auc")
fpiv = fm.pivot_table(index=["draw", "rep", "fold"], columns="method", values="auc")
for i, r in res.iterrows():
    if r.metric == "auc":
        res.loc[i, "units_a_gt_b"] = float((piv[r.a] > piv[r.b]).mean()); res.loc[i, "folds_a_gt_b"] = float((fpiv[r.a] > fpiv[r.b]).mean())
        res.loc[i, "fold_diff_mean"] = float((fpiv[r.a] - fpiv[r.b]).mean()); res.loc[i, "fold_diff_sd"] = float((fpiv[r.a] - fpiv[r.b]).std())
# Holm over the two pre-registered confirmatory hypotheses (H1: T2>Q ; H2: Cmean>T2) on AUC
h = res[(res.metric == "auc") & (((res.a == "T2") & (res.b == "Q")) | ((res.a == "Cmean") & (res.b == "T2")))].sort_values("p_two_sided")
holm = {}
for j, (i, r) in enumerate(h.iterrows()): holm[i] = min(1.0, r.p_two_sided * (len(h) - j))
res["p_holm_confirmatory"] = pd.Series(holm)
res.round(4).to_csv(f"{run}/table_paired_bootstrap.csv", index=False)
print("\n== PAIRED HIERARCHICAL BOOTSTRAP (a - b)"); print(res.round(3).to_string(index=False))
