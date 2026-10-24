# Fig. 4: performance and limits of the surrogate (Eq. 1).
# (a) surrogate vs direct PARMA on the independent set (all 1995 in-domain points, 6-15 km, 2009-2026 dates).
# (b) 95th-percentile absolute error: held-out grid nodes by group (interpolation) and four region hold-out
#     tests (extrapolation), from data/step4/formula_heldout_by_group.csv and formula_fit.json.
import sys, json, pathlib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from figstyle import DOUBLE, DATA_CMAP, BAND, panel_label

V = pd.read_csv(ROOT / "data/step4/validation_direct.csv"); V = V[V.in_domain]
G = pd.read_csv(ROOT / "data/step4/formula_heldout_by_group.csv")
fit = json.load(open(ROOT / "data/step4/formula_fit.json"))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 3.0), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.25]})

# (a) parity plot
lo, hi = 150, 9000
x = np.array([lo, hi])
ax1.fill_between(x, x / 1.1, x * 1.1, color=BAND, lw=0, zorder=0)
ax1.plot(x, x, color="#444444", lw=0.7, zorder=1)
sc = ax1.scatter(V.PARMA, V.FORMULA, c=V.W, cmap=DATA_CMAP, s=7, edgecolors="white", linewidths=0.2, rasterized=True, zorder=2)
ax1.set_xscale("log"); ax1.set_yscale("log"); ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi); ax1.set_aspect("equal")
TK = [200, 500, 1000, 2000, 5000]
for setter in (ax1.set_xticks, ax1.set_yticks):
    setter(TK)
ax1.set_xticklabels([str(t) for t in TK]); ax1.set_yticklabels([str(t) for t in TK]); ax1.minorticks_off()
ax1.set_xlabel("PARMA, flux > 10 MeV (cm$^{-2}$ h$^{-1}$)")
ax1.set_ylabel("Surrogate, Eq. (1) (cm$^{-2}$ h$^{-1}$)")
e = np.abs(V.FORMULA / V.PARMA - 1) * 100
ax1.text(0.04, 0.96, "n = %d\nmedian |error| %.1f%%\nP95 |error| %.1f%%" % (len(V), np.median(e), np.percentile(e, 95)),
         transform=ax1.transAxes, va="top", fontsize=7)
ax1.text(0.96, 0.06, "band: $\\pm$10%", transform=ax1.transAxes, ha="right", fontsize=6.5, color="#555555")
cb = fig.colorbar(sc, ax=ax1, pad=0.02, aspect=28, shrink=0.85)
cb.set_label("W-index"); cb.outline.set_visible(False); cb.ax.tick_params(length=2)
panel_label(ax1, "(a)")

# (b) interpolation (held-out groups) vs extrapolation (region hold-out)
def nice(gname):
    k, v = gname.split("=", 1)
    v = v.replace("(", "").replace("]", "").replace(", ", "–").replace(".0", "")
    return {"W": "W = %s" % v.replace(".0", ""), "alt_km": "%s km" % v.replace("5.9", "6").replace("15.1", "15"),
            "Rc_GV": "R$_c$ %s GV" % v.replace("-0.1", "0").replace("18.1", "18")}[k]
rows = [(nice(g), p95, "interp") for g, p95 in zip(G.group, G.p95_abs_pct)]
EXT = {"W: train W<=75, test W>=100": "W ≥ 100 (trained W ≤ 75)",
       "Rc: train Rc<=12, test Rc>12": "R$_c$ > 12 GV (trained ≤ 12)",
       "alt: train alt<=12, test alt>12.5": "> 12.5 km (trained ≤ 12)",
       "alt: train alt>=9, test alt<9": "< 9 km (trained ≥ 9)"}
rows += [(EXT[k], v["p95_abs_pct"], "extrap") for k, v in fit["extrapolation"].items()]
y = np.arange(len(rows))[::-1]
for yi, (lab, p95, kind) in zip(y, rows):
    col = "#2171b5" if kind == "interp" else "#D55E00"
    ax2.plot([0, p95], [yi, yi], color=col, lw=1.0, alpha=0.6)
    ax2.plot([p95], [yi], marker="o", ms=4.5, color=col, mec="white", mew=0.5)
    ax2.text(p95 + 0.4, yi, "%.1f" % p95, va="center", fontsize=6.5, color="#333333")
n_int = sum(1 for r in rows if r[2] == "interp")
ax2.axhline(y[n_int - 1] - 0.5, color="#bbbbbb", lw=0.6)
ax2.set_yticks(y); ax2.set_yticklabels([r[0] for r in rows], fontsize=6.5)
ax2.tick_params(axis="y", which="both", length=0); ax2.spines["left"].set_visible(False); ax2.minorticks_off()
ax2.set_xlim(0, 26); ax2.set_xlabel("P95 |error| relative to PARMA (%)")
ax2.text(25.8, y[0], "held-out nodes\n(interpolation)", ha="right", va="top", fontsize=6.5, color="#2171b5")
ax2.text(25.8, (y[-1] + y[-2]) / 2, "region hold-out\n(extrapolation)", ha="right", va="center", fontsize=6.5, color="#D55E00")
panel_label(ax2, "(b)", dy=-0.20)

out = HERE / "fig_04_surrogate_performance"
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
print("independent: n=%d median %.2f%% p95 %.2f%% max %.2f%%" % (len(V), np.median(e), np.percentile(e, 95), e.max()))
for r in rows: print("%-40s %.2f %s" % r)
