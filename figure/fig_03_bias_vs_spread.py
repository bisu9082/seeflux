# Fig. 3: reference-dependent bias versus reference-independent (structural) spread, cruise 9-12.5 km,
# independent direct-PARMA set. (a) bias-spread map: x = median model/PARMA with the reference band
# k = 0.88-1.37 applied (horizontal bar), y = p95/p05 spread of model/PARMA (invariant to any constant
# rescaling of the reference). (b) distributions after removing each model's own median bias.
import sys, json, pathlib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from figstyle import DOUBLE, MODEL, BAND, panel_label

U = pd.read_csv(ROOT / "data/step4/uncertainty_band.csv")
U = U[U.dataset == "independent"].set_index("model")
band = json.load(open(ROOT / "data/step4/reference_band.json"))
V = pd.read_csv(ROOT / "data/step4/validation_direct.csv")
V = V[V.alt_km.between(9.0, 12.5)]
ORDER = ["FORMULA", "S2_JESD_mid", "S1_IEC", "S4_AN_ABB", "S4b_AN_CoolCAD"]
rng = np.random.default_rng(11)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 3.0), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.1]})

# (a) bias-spread map
ax1.axvspan(0.3, 1.0, color="#fbeee6", lw=0, zorder=0)
ax1.text(0.32, 0.97, "non-conservative side", fontsize=6.5, color="#a0522d", va="bottom")
ax1.text(2.35, 0.97, "conservative side", fontsize=6.5, color="#555555", va="bottom", ha="right")
ax1.axvline(1.0, color="#444444", lw=0.7)
# label anchor (x, y, ha, va) chosen per model to avoid overlaps
LBL = {"S1_IEC": ("hi+", 0.0, "left", "center"), "S4_AN_ABB": ("x", 0.10, "center", "bottom"),
       "S4b_AN_CoolCAD": ("x", -0.10, "center", "top"), "S2_JESD_mid": ("hi+", 0.0, "left", "center"),
       "FORMULA": ("x", 0.10, "center", "bottom")}
for m in ORDER:
    col, _, lab = MODEL[m]
    x, lo, hi, y = U.loc[m, "median_ratio"], U.loc[m, "bias_band_lo"], U.loc[m, "bias_band_hi"], U.loc[m, "spread_p95_over_p05"]
    ax1.plot([lo, hi], [y, y], color=col, lw=3.0, alpha=0.35, solid_capstyle="butt", zorder=2)
    ax1.plot([x], [y], marker="o", ms=6, mfc=col, mec="white", mew=0.8, zorder=3)
    where, dy, ha, va = LBL[m]
    lx = hi * 1.04 if where == "hi+" else x
    ax1.text(lx, y + dy, lab, color=col, fontsize=7, ha=ha, va=va)
ax1.set_xscale("log")
ax1.set_xlim(0.3, 2.4); ax1.set_ylim(0.95, 3.1)
ax1.set_xticks([0.4, 0.5, 0.7, 1, 1.4, 2]); ax1.set_xticklabels(["0.4", "0.5", "0.7", "1", "1.4", "2"]); ax1.minorticks_off()
ax1.set_xlabel("Bias: median model / reference")
ax1.set_ylabel("Structural spread: P95 / P5 of model / PARMA")
r2 = lambda v: f"{np.floor(v * 100 + 0.5) / 100:.2f}"  # round half up, as in the text
ax1.text(0.98, 0.42, "dot: PARMA as reference\nbar: reference band\nk = %s–%s" % (r2(band["K_LO"]), r2(band["K_HI"])),
         transform=ax1.transAxes, ha="right", va="center", fontsize=6.5, color="#555555")
panel_label(ax1, "(a)")

# (b) residual distributions after removing each model's median bias
ax2.axvspan(1 / 1.3, 1.3, color=BAND, lw=0, zorder=0)
for x in (0.5, 2.0):
    ax2.axvline(x, color="#9a9a9a", lw=0.6, ls=(0, (3, 2)), zorder=0)
ax2.axvline(1.0, color="#444444", lw=0.7, zorder=0)
for i, m in enumerate(ORDER):
    s = V[V.in_domain] if m == "FORMULA" else V
    r = (s[m] / s.PARMA).values; rn = r / np.median(r)
    col, _, lab = MODEL[m]; y = len(ORDER) - 1 - i
    ax2.scatter(rn, y + rng.uniform(-0.22, 0.22, len(rn)), s=3, color=col, alpha=0.35, linewidths=0, rasterized=True, zorder=1)
    q05, q25, q50, q75, q95 = np.percentile(rn, [5, 25, 50, 75, 95])
    ax2.plot([q05, q95], [y, y], color="#222222", lw=0.8, zorder=3)
    ax2.add_patch(plt.Rectangle((q25, y - 0.17), q75 - q25, 0.34, fc="white", ec="#222222", lw=0.8, zorder=3))
    ax2.plot([1, 1], [y - 0.17, y + 0.17], color=col, lw=2.0, zorder=4)
    ax2.text(3.3, y, "%d%%" % round(U.loc[m, "frac_outside_30pct_after_bias_removal"] * 100), ha="right", va="center", fontsize=7)
ax2.text(3.3, len(ORDER) - 0.35, "outside $\\pm$30%\nafter bias removal", ha="right", va="bottom", fontsize=6.5, color="#555555")
ax2.set_xscale("log"); ax2.set_xlim(0.35, 3.4)
ax2.set_xticks([0.5, 0.77, 1, 1.3, 2]); ax2.set_xticklabels(["0.5", "0.77", "1", "1.3", "2"]); ax2.minorticks_off()
ax2.set_yticks(range(len(ORDER))); ax2.set_yticklabels([MODEL[m][2] for m in ORDER][::-1])
ax2.set_ylim(-0.6, len(ORDER) - 0.1); ax2.tick_params(axis="y", length=0); ax2.spines["left"].set_visible(False)
ax2.set_xlabel("Model / PARMA, divided by the model's median")
panel_label(ax2, "(b)")

out = HERE / "fig_03_bias_vs_spread"
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
print(U[["median_ratio", "bias_band_lo", "bias_band_hi", "spread_p95_over_p05", "frac_outside_30pct_after_bias_removal"]].round(3).to_string())
