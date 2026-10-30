# -*- coding: utf-8 -*-
"""Figure 1 of the article: mean ROC curves of T^2, Q (SPE) and OC-SVM (black-and-white, decimal comma, Russian labels as in the article).
No model is trained and no score is generated: ROC curves are recomputed from the stored out-of-fold scores (oof_draw*.npz) of a run of scripts/run_nested_cv.py.
Adapted from the project's figure scripts (figcommon.py + make_fig01_bw.py): helpers merged into this file, run-directory arguments instead of project-layout paths,
the cross-check now uses table_unit_level.csv from the same run directory (created by scripts/analyze_paired_bootstrap.py).
Usage: python scripts/make_fig01_roc.py <run_dir_with_oof_draw*.npz_and_table_unit_level.csv> <out_dir>"""
import sys, os, json
sys.dont_write_bytecode = True
run_dir, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from scipy.stats import rankdata
sys.stdout.reconfigure(encoding="utf-8")


def load_oof(d):
    return [np.load(os.path.join(d, f"oof_draw{i}.npz"), allow_pickle=True) for i in range(10)]


def auc(y, s):
    r = rankdata(s); n1 = int(y.sum()); n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def roc_points(y, s):
    """Full ROC (every distinct threshold; no drop_intermediate)."""
    o = np.argsort(-s, kind="mergesort"); ys = y[o]; ss = s[o]
    distinct = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    tp = np.cumsum(ys)[distinct]; fp = (distinct + 1) - tp
    return np.r_[0, fp / (len(y) - y.sum())], np.r_[0, tp / y.sum()]


def tpr_on_grid(fpr, tpr, grid):
    """Step function: TPR at FPR=f is the largest TPR among ROC points with FPR <= f."""
    idx = np.searchsorted(fpr, grid, side="right") - 1
    return tpr[idx]


Z = load_oof(run_dir); METHODS = ["T2", "Q", "ocsvm"]; grid = np.linspace(0, 0.20, 4001)
curves = {m: np.zeros((10, 10, len(grid))) for m in METHODS}; aucs = {m: np.zeros((10, 10)) for m in METHODS}; t1 = {m: np.zeros((10, 10)) for m in METHODS}; t2 = {m: np.zeros((10, 10)) for m in METHODS}
for d, z in enumerate(Z):
    y = z["y"].astype(int)
    for r in range(10):
        for m in METHODS:
            s = z[m][r]; fpr, tpr = roc_points(y, s); curves[m][d, r] = tpr_on_grid(fpr, tpr, grid); aucs[m][d, r] = auc(y, s)
            t1[m][d, r] = tpr_on_grid(fpr, tpr, np.array([0.01]))[0]; t2[m][d, r] = tpr_on_grid(fpr, tpr, np.array([0.02]))[0]
unit = pd.read_csv(os.path.join(run_dir, "table_unit_level.csv")).set_index("method"); ver = {}
for m in METHODS:
    ver[m] = dict(auc_unit_mean=float(aucs[m].mean()), auc_table=float(unit.loc[m, "auc"]), tpr1_full=float(t1[m].mean()), tpr2_full=float(t2[m].mean()))
    ver[m]["ok"] = bool(abs(ver[m]["auc_unit_mean"] - ver[m]["auc_table"]) < 6e-5)
assert all(v["ok"] for v in ver.values()), ver
cm = lambda x, _: f"{x:.2f}".replace(".", ","); cm3 = lambda x, _: f"{x:.3f}".replace(".", ",")
plt.rcParams.update({"font.family": ["Times New Roman", "DejaVu Serif"], "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8.5, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.linewidth": 0.8, "svg.fonttype": "none", "pdf.fonttype": 42, "hatch.linewidth": 0.5})
STY = {"T2": dict(color="black", ls="-", lw=1.5), "Q": dict(color="0.35", ls=(0, (6, 2)), lw=1.4), "ocsvm": dict(color="0.15", ls=(0, (1, 1.6)), lw=1.6)}
HATCH = {"T2": "////", "Q": "\\\\\\\\", "ocsvm": "...."}; LAB = {"T2": "T²", "Q": "Q (SPE)", "ocsvm": "OC-SVM"}
fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.9), gridspec_kw=dict(width_ratios=[1.15, 1]))
for ax, xmax, title in ((axes[0], 0.20, "а"), (axes[1], 0.02, "б")):
    sel = grid <= xmax + 1e-12; ax.plot([0, xmax], [0, xmax], color="0.6", lw=0.8, ls="-.", label="случайный классификатор" if ax is axes[0] else None)
    for m in METHODS:
        mc = curves[m].mean(axis=(0, 1)); dm = curves[m].mean(axis=1)
        ax.fill_between(grid[sel], dm.min(0)[sel], dm.max(0)[sel], facecolor="0.85", edgecolor="0.55", hatch=HATCH[m], alpha=0.35, lw=0)
        lab = f"{LAB[m]} (AUC {aucs[m].mean():.3f})".replace(".", ",") if ax is axes[0] else None
        ax.plot(grid[sel], mc[sel], label=lab, drawstyle="steps-post", **STY[m])
    ax.set_xlim(0, xmax); ax.set_xlabel("Доля ложных срабатываний, FPR"); ax.grid(True, color="0.8", lw=0.5); ax.xaxis.set_major_formatter(FuncFormatter(cm3 if xmax <= 0.02 else cm)); ax.yaxis.set_major_formatter(FuncFormatter(cm if xmax > 0.02 else cm3))
    ax.text(0.02, 1.02, title, transform=ax.transAxes, fontsize=9, style="italic", fontweight="normal")
axes[0].set_ylim(0, 0.8); axes[1].set_ylim(0, max(curves[m].mean(axis=1)[:, grid <= 0.02].max() for m in METHODS) * 1.05); axes[0].set_ylabel("Полнота, TPR"); axes[0].legend(loc="upper left", frameon=False)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig01_roc_low_fpr_bw.png"), dpi=600); fig.savefig(os.path.join(OUT, "fig01_roc_low_fpr_bw.pdf"))
tab = {m: {str(f): float(curves[m].mean(axis=(0, 1))[np.argmin(np.abs(grid - f))]) for f in (0.005, 0.01, 0.02, 0.05, 0.10, 0.20)} for m in METHODS}
json.dump(dict(verification=ver, tpr_at_fpr_mean_curve=tab, note="recomputed from stored OOF scores; no training, no new scores"), open(os.path.join(OUT, "fig01_verification.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print(json.dumps(ver))
