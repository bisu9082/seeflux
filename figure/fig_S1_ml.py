# Fig. S1 (supplementary, exploratory ML; plan in data/step_ml/ml_plan.md).
# (a) P95 error of emulators trained on the same 252 nodes as Eq. (1): independent set and four region hold-outs.
# (b) Held-out R^2 of ln(model/PARMA) explained by nested input sets (XGBoost), cruise audit nodes.
import sys, pathlib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from figstyle import DOUBLE, OKABE, panel_label

E = pd.read_csv(ROOT / "data/step_ml/ml1_emulators.csv")
N = pd.read_csv(ROOT / "data/step_ml/ml4_nested.csv")
A = pd.read_csv(ROOT / "data/step_ml/ml2_attribution_r2.csv")

TESTS = [("independent (1995)", "in-domain: independent (1995)", "Independent set\n(in domain)"),
         ("extrapolate W>=100 (train W<=75)", "W>=100 (train W<=75)", "$W \\geq 100$"),
         ("extrapolate Rc>12 (train Rc<=12)", "Rc>12 (train Rc<=12)", "$R_c > 12$ GV"),
         ("extrapolate alt>12.5 (train alt<=12)", "alt>12.5 (train alt<=12)", "alt > 12.5 km"),
         ("extrapolate alt<9 (train alt>=9)", "alt<9 (train alt>=9)", "alt < 9 km")]
MODELS = [("Eq. (1)", "#000000", "o"), ("Linear, 10 terms", OKABE["green"], "D"),
          ("Gaussian process", OKABE["blue"], "s"), ("MLP ensemble", OKABE["purple"], "^"),
          ("XGBoost", OKABE["orange"], "v"), ("kNN (k=4, distance)", "#999999", "x")]


def p95(model, t_ml1, t_nest):
    if model == "Linear, 10 terms":
        return float(N[(N.test == t_nest) & (N.model == "nested k=10")].p95_pct.iloc[0])
    return float(E[(E.model == model) & (E.test == t_ml1)].p95_pct.iloc[0])


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.9), constrained_layout=True, gridspec_kw={"width_ratios": [1.35, 1]})
x = np.arange(len(TESTS))
off = np.linspace(-0.3, 0.3, len(MODELS))
for (m, c, mk), dx in zip(MODELS, off):
    y = [p95(m, a, b) for a, b, _ in TESTS]
    ax1.scatter(x + dx, y, color=c, marker=mk, s=16, label=m, zorder=3, linewidths=0.9)
ax1.set_yscale("log"); ax1.set_ylim(0.8, 400)
ax1.set_yticks([1, 2, 5, 10, 20, 50, 100, 200]); ax1.set_yticklabels(["1", "2", "5", "10", "20", "50", "100", "200"])
ax1.axvline(0.5, color="#bbbbbb", lw=0.6)
for xi in x[1:-1]:
    ax1.axvline(xi + 0.5, color="#eeeeee", lw=0.5, zorder=0)
ax1.text(2.5, 330, "region hold-out (extrapolation)", ha="center", va="top", fontsize=6.5, color="#555555")
ax1.set_xticks(x); ax1.set_xticklabels([t[2] for t in TESTS], fontsize=6.5)
ax1.set_ylabel("P95 |error| relative to PARMA (%)")
ax1.legend(fontsize=6, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.2, columnspacing=0.8)
panel_label(ax1, "(a)")

SETS = [("R2 alt,|lat|", "alt, |λ|"), ("R2 alt,|lat|,W", "+ W"), ("R2 alt,|lat|,Rc", "+ $R_c$"), ("R2 alt,|lat|,W,Rc", "+ W + $R_c$")]
COL = ["#c6dbef", OKABE["orange"], OKABE["sky"], OKABE["blue"]]
mods = A.model.tolist()
y = np.arange(len(mods))
h = 0.2
for i, ((col, lab), c) in enumerate(zip(SETS, COL)):
    ax2.barh(y + (i - 1.5) * h, A[col].clip(lower=0), height=h, color=c, label=lab)
ax2.set_yticks(y); ax2.set_yticklabels(mods, fontsize=7); ax2.invert_yaxis()
ax2.set_xlim(0, 1.0); ax2.set_xlabel("Held-out $R^2$ of ln(model / PARMA)")
ax2.legend(fontsize=6, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 1.0), frameon=False, handletextpad=0.3, columnspacing=0.8)
panel_label(ax2, "(b)")

for ext in ("pdf", "png"):
    fig.savefig(HERE / f"fig_S1_ml.{ext}", dpi=400 if ext == "png" else None)
print("saved")
