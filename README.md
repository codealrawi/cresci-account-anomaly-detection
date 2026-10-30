# Hotelling's T² vs. Q/SPE for anomalous-account detection on Cresci-2017

Code and aggregated results accompanying the manuscript **"Comparison of Hotelling's T² and Q/SPE Statistics for Anomalous Account Detection on the Cresci-2017 Dataset"** (Russian title: «Сравнение статистик Хотеллинга T² и Q/SPE для выявления аномальных аккаунтов на данных Cresci-2017»; submitted to *Doklady TUSUR*, specialty 2.3.5 — publication status and DOI: not yet available).

> **Important.** This repository does **not** contain the Cresci-2017 data. You have to obtain it yourself and comply with its terms of use (see [Data](#data)).

## Scientific task

A static, unsupervised-score detector of anomalous social-media accounts is built from account-level behavioural features: standardisation → SVD/PCA subspace of the training part → anomaly scores. Two classical PCA-monitoring statistics, their combinations and three baselines are compared under a nested cross-validation protocol fixed before the run.

Research questions (as in the manuscript):

1. **RQ1** – how does the residual statistic Q behave (with k chosen by inner validation and by the 90 % explained-variance rule)?
2. **RQ2** – how do Q and T², and their combinations, compare under strict evaluation?
3. **RQ3** – do the conclusions depend on the treatment of line breaks in the text of genuine accounts (variants A and B)?
4. **RQ4** – which limitations do the data, the protocol and the threshold rules impose, and do the results allow conclusions about practical applicability?

### The statistics

With a standardised training matrix `Z = U Σ Vᵀ`, `V_k` the first `k` right singular vectors, `t = V_kᵀ z` the projections, `λ_j = S_j² / m` the eigenvalues of the training covariance and `λ̄` their mean:

* **Q / SPE** – squared norm of the residual of the projection onto the first `k` components: `Q = ‖z‖² − Σ_{j≤k} t_j²`.
* **T²** (regularised, ridge `ρ` in units of the mean eigenvalue): `T²(z) = Σ_{j≤k} t_j² / (λ_j + ρ·λ̄)`.
* Five pre-specified combinations of the two statistics (`Cmax`, `Cmean`, `Cfisher`, `Cnorm`, `Cw`) built from their empirical CDFs on the training part; see `src/scoring.py`.
* Baselines: Isolation Forest, Local Outlier Factor (novelty mode), one-class SVM (scikit-learn).

Hyper-parameters (`k ∈ {1,…,6}`, `ρ ∈ {0; 0.1; 1}`, `w ∈ {0.25; 0.5; 0.75}`, baseline grids) are selected only on the training part by an inner 3-fold CV (mean ROC-AUC). Thresholds are training-ECDF percentiles (q = 95 main, q = 92 additional). The method is **not** claimed to be new.

## Data

**Cresci-2017** (Cresci, Di Pietro, Petrocchi, Spognardi, Tesconi, *The Paradigm-Shift of Social Spambots: Evidence, Theories, and Tools for the Arms Race*, WWW '17 Companion, 2017, pp. 963–972, DOI [10.1145/3041021.3055135](https://doi.org/10.1145/3041021.3055135)).

* The dataset is distributed by its authors for **research purposes only**. The project page (cited in the manuscript from an archived snapshot dated 2025-09-09; its current availability was not re-checked) is `http://mib.projects.iit.cnr.it/dataset.html`. The datasets are provided on request; follow the instructions and conditions stated there (cite the paper above, do not redistribute the data).
* **This repository contains no Cresci-2017 records**: no tweets, no account tables, no per-account scores or features. Per the authors' statement quoted in the manuscript, publication of research code, aggregated metrics, statistical comparisons and descriptions of features/preprocessing is permitted; the original data and its parts are not. This README is not legal advice — you remain responsible for complying with the dataset's terms.
* The manuscript uses the **official full archive** `datasets_full.csv.zip` (877,405,477 bytes, SHA-256 `59A829644039FC09EE05CAEEB46E0D740C0A5132D2FE0D4EE5FC12C1FF84CDA6`). A mirror copy of the data exists in a public bot repository, but its file with the genuine accounts' tweets is truncated (see the manuscript); do not use it to reproduce the results.
* Extract the archive so that you have a directory (below: `DATA_DIR`) containing `genuine_accounts.csv.zip`, `social_spambots_1.csv.zip`, `social_spambots_2.csv.zip`, `social_spambots_3.csv.zip`, `fake_followers.csv.zip`.

### Sample used

Accounts with at least two tweets: 3,394 genuine accounts and 8,000 positives (`social_spambots_1–3` and `fake_followers`; traditional spam-bots are not used) = 11,394 accounts. Each of 10 "draws" takes all genuine accounts plus 295 positives (3,689 accounts, 8.0 % positives); each draw is evaluated with 10 repeats × 5-fold stratified CV (500 outer folds).

### Features (our own definitions; exact formulas in `src/extract_features_v2.py`)

| | feature |
|---|---|
| p1 | posts per day (tweets / observed span in days, ≥ 1 day) |
| p2 | profile favourites per day of account age at crawl |
| p3 | share of retweets |
| p4 | share of tweets with ≥ 1 URL |
| p5 | Shannon entropy (bits) of the posting-hour histogram (UTC) |
| p6 | mean tweet length (first 100 tweets) |
| p7 | share of uppercase letters (first 100 tweets) |
| p8 | variance of inter-post intervals (hours²) |
| p9 | share of repeated word bi/trigram occurrences (first 100 tweets) |

Two preprocessing variants differ only in how the text of the **genuine** accounts' tweets is read: **B** (main) removes CR/LF characters (`escaped_nonl`), **A** keeps them (`escaped`). Positives are read identically in both.

## Repository layout

```
src/        svdad.py (SVD residual, ECDF), scoring.py (Q, T², combinations), evaluate.py (draws, folds, metrics, baselines),
            extract_features_v2.py + official_reader.py (feature extraction from the official archive), extract_features.py (frozen reference version, used by a test)
scripts/    run_nested_cv.py, analyze_paired_bootstrap.py, run_explained_variance_rule.py, make_fig01_roc.py
tests/      pytest tests on synthetic data (no Cresci-2017 content)
results/    aggregated results of the runs reported in the manuscript (see results/README.md)
docs/       licensing notes
```

Note on provenance: file headers and comments in `src/` and `scripts/` mention internal project stages ("Stage 31/35/36/43") and an earlier article whose SVD-residual detector configuration was re-implemented ("article A"). The internal documents they refer to (method specification, data contract, pre-registration notes) are not part of this repository; the code itself is self-contained. Files in `src/` are copies of the frozen code used for the reported runs; the three scripts in `scripts/` differ from their originals only in path handling (see the comment at the top of each).

## Environment

Python 3.12.13 was used for all reported runs. Recorded package versions of the runs (`results/*/run_config.json`): numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1, scipy 1.18.1. `requirements.txt` additionally pins joblib, matplotlib and pytest as installed in the authors' environment (these three were not recorded inside the run records). Other versions were not tested and may change numerical details.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -I -m pytest tests -q -p no:cacheprovider
```

## Reproducing the results

All commands run from the repository root. Replace `DATA_DIR` with the directory described in [Data](#data).

```bash
# 1. Features (variant B = main; variant A = sensitivity analysis). Long: reads ~0.9 GB of zipped CSV.
python -I src/extract_features_v2.py DATA_DIR features_B.csv escaped_nonl pandas features_B_stats.json
python -I src/extract_features_v2.py DATA_DIR features_A.csv escaped pandas features_A_stats.json
#    Expected SHA-256 of the feature tables used in the manuscript:
#    B: CBF250AB153E2D8B458A1F7A2CCE22590A4595E93250DD7E853BDD45DECA8A3D
#    A: 14C9114BE973B4AD27CCB70B3F033D1A035E0D34C58FBA278158B73ADD7A8096

# 2. Main nested-CV experiment (10 draws x 10 repeats; 6.4 min with 12 parallel jobs in the original run)
#    arguments: features, output dir, n_draws, n_repeats, min_tweets, with_baselines
python -I scripts/run_nested_cv.py features_B.csv runs/main_B 10 10 0 1
python -I scripts/analyze_paired_bootstrap.py runs/main_B          # tables + paired hierarchical bootstrap (~2 min)

# 3. Sensitivity analysis (variant A) - the same two commands with features_A.csv
python -I scripts/run_nested_cv.py features_A.csv runs/variant_A 10 10 0 1
python -I scripts/analyze_paired_bootstrap.py runs/variant_A

# 4. Explained-variance (90 %) rule - a single configuration, takes seconds
python -B scripts/run_explained_variance_rule.py features_B.csv runs/rule90

# 5. Figure 1 (needs runs/main_B with its oof_draw*.npz and the table written by analyze_paired_bootstrap.py)
python -B scripts/make_fig01_roc.py runs/main_B figures
```

`SVDAD_N_JOBS` (default 12) sets the number of parallel workers of `run_nested_cv.py`; it does not change the results.

`runs/` and the feature tables are git-ignored: they contain account-level derived data (account identifiers and per-account scores) that are not part of the public release.

### What was verified for this release

Using the authors' locally stored feature table of variant B (not included here): the 90 % rule script reproduced `results/explained_variance_rule/*.csv` exactly; `run_nested_cv.py` on draw 0 / repeat 0 reproduced the stored per-unit metrics exactly; `analyze_paired_bootstrap.py` reproduced all four `table_*.csv` files of the main run exactly, and `make_fig01_roc.py` reproduced the figure's verification numbers. The feature-extraction command was not re-run on the 0.9 GB archive for this release (it is covered by the synthetic tests in `tests/` and by the checksums above).

## Results (variant B, main experiment; from `results/main_variant_B/`)

Mean ROC-AUC over 100 units (draw × repeat); paired differences with 95 % hierarchical-bootstrap intervals over 10 draws:

| Statistic / method | ROC-AUC | PR-AUC |
|---|---|---|
| Q / SPE | 0.562 | 0.097 |
| T² | **0.824** | 0.199 |
| Cmax / Cmean / Cfisher / Cnorm / Cw | 0.761 / 0.721 / 0.749 / 0.796 / 0.798 | 0.150 / 0.128 / 0.139 / 0.168 / 0.166 |
| Isolation Forest / LOF / OC-SVM | 0.570 / 0.490 / 0.790 | 0.093 / 0.091 / 0.177 |

* T² − Q: **+0.262** (95 % CI 0.250–0.274; 10/10 draws; Holm p ≤ 0.001). Pre-registered H2 (Cmean > T²) was **not** supported (−0.103). No combination beat T².
* T² − OC-SVM: +0.0285 (95 % CI 0.021–0.035), a secondary descriptive comparison.
* **Low-FPR behaviour:** TPR at FPR ≤ 2 % does not exceed 0.046 for any method; at FPR 1 % T² reaches 0.014 against 0.031 (Q) and 0.028 (OC-SVM). The mean ROC curves cross, so **the ROC-AUC advantage of T² arises at higher FPR and does not hold at low FPR**.
* 90 % explained-variance rule for Q: ROC-AUC 0.518 (k = 7 in 22 folds, 8 in 478 folds; a single computation).
* T² selected k = 1 and ρ = 0 in all 500 folds, i.e. it is the squared projection on the first principal component.
* Variant A (line breaks kept) changes ROC-AUC by at most 0.00095 (`results/sensitivity_variant_A/`).

## Limitations

* One corpus (Cresci-2017) and internal validation only; the hypothesis about T² arose on the same data; there is no external validation.
* Intervals are conditional on a shared set of genuine accounts and do not cover uncertainty from the corpus or from the feature definitions; the 100 units and 500 folds are dependent.
* The results do not show practical suitability: thresholds give weak precision and recall (see the manuscript); transferability to other corpora, domains or platforms was not examined.
* Features are our own definitions; other feature sets were not tested. Only the removal of CR/LF was tested as a preprocessing sensitivity.
* LOF's threshold is biased (6.7 % flagged instead of 5 %); its threshold metrics are not comparable.
* A separate re-implementation covered Q, T² and Cmean (500/500 folds agreed); the other methods were re-checked from predictions only. This is not an independent replication.

## How to cite

The manuscript has no DOI yet; please check back for the final reference. Please also cite the dataset paper (Cresci et al., 2017, DOI above). Primary methodological references (full list in the manuscript): Hotelling (1931) for T²; Jackson & Mudholkar (1979) for the residual statistic Q/SPE; Liu et al. (2008), Breunig et al. (2000) and Schölkopf et al. (2001) for the baselines.

## License

No license has been selected for the code yet (the decision is pending with the authors), so no license file is included; until one is added, standard copyright applies and you must ask the authors before reuse. See `docs/LICENSE_OPTIONS.md`. The Cresci-2017 data are **not** covered by any license in this repository and are not redistributed; their use is governed by the dataset authors' conditions.
