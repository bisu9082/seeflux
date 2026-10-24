# Exploratory ML analysis plan (fixed before running, 2026-10-16)

Status: exploratory, post hoc relative to the Step 4 design. All results reported, including null ones.
Data are PARMA 4.10 outputs (deterministic model), not measurements. No ML result is a claim about the real
atmosphere; each answers a question about the reference surface or the simplified models.
Seed 20261016. Target for flux models: ln F(>10 MeV). Error metric: |F_pred / F_PARMA - 1| (%): median, P95, max.

## ML1. Is the 7% P95 of Eq. (1) limited by its closed form?
Emulators: Eq. (1) (fixed coefficients), ridge on cubic polynomial features, Gaussian process (ARD Matern 5/2 + white),
gradient-boosted trees (XGBoost), MLP (5-seed ensemble), k-nearest neighbours.
Training: T252 = the same 252 grid nodes used for Eq. (1). Hyperparameters fixed in advance (no tuning on test data).
Tests: (a) 4669 held-out grid nodes, (b) independent set (1995 in-domain points), (c) the four region hold-outs of
Step 4 (W>=100, Rc>12, alt>12.5, alt<9), each model refitted on the same reduced training nodes.
Extra: GP 95% predictive interval coverage on (b) and (c) (does GP uncertainty flag extrapolation?).
Decision rule: an emulator "beats" Eq. (1) only if better on (b) AND not worse on (c).

## ML2. Attribution of each simplified model's error to its causes
Data: audit nodes at cruise (9 to 12.5 km). Target y = ln(model / PARMA) for IEC, JESD89 mid, AN4589 ABB, CoolCAD.
Method: XGBoost R^2 on a 50/50 split for nested feature sets {alt,|lat|}, +W, +Rc (gain from Rc at fixed |lat| = longitude /
geomagnetic share; gain from W = missing solar term), plus mean |SHAP| shares on (alt, |lat|, W, Rc).

## ML3. A readable rule for when IEC 62396-1 is non-conservative (ratio < 1)
Train on audit cruise nodes, test on independent cruise points (9 to 12.5 km).
Feature sets: A = what the standard user has (alt, |lat|, W); B = (alt, Rc, W).
Models: decision tree depth 2 and 3, logistic regression. Metrics: balanced accuracy, recall of the non-conservative class.

## ML4. Sparse symbolic regression: a simpler or better closed form?
Library of 20 linear-in-parameter terms in h, Rc, W. Forward stepwise selection on T252, BIC along the path.
Compare the BIC-best and each k-term model (k = 3..10) with Eq. (1) on held-out grid and independent set.

## Use in the paper
Default: supplementary material only. A result enters the main text only if it changes or sharpens a stated conclusion.

## Amendment after internal review (2026-10-16)
ML2 originally split the grid nodes at random into halves. Neighbouring nodes of a deterministic grid then fall on both
sides, which inflates R^2 (reviewer comment). ML2 was rerun with a split by location: all altitudes and W levels of a
grid cell are on the same side. Both results are kept in data/step_ml (ml2_attribution_r2_randomsplit.csv is the first run).
