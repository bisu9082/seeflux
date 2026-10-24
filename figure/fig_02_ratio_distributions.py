# Fig. 2: how far each simplified model departs from the PARMA reference at cruise (9-12.5 km).
# (a) distribution of model/PARMA on the independent direct-PARMA set (n = 802 cruise points, 2009-2026 dates);
# (b) median and 5-95% range of model/PARMA versus solar activity (W-index) on the geographic grid audit.
import sys, pathlib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from figstyle import DOUBLE, MODEL, BAND, panel_label

V = pd.read_csv(ROOT / "data/step4/validation_direct.csv")
V = V[V.alt_km.between(9.0, 12.5)]
A = pd.read_csv(ROOT / "data/step4/audit_nodes.csv", usecols=["W", "alt_km", "PARMA", "S1_IEC", "S2_JESD_mid", "S4_AN_ABB", "FORMULA"])
A = A[A.alt_km.between(9.0, 12.5)]
ORDER = ["FORMULA", "S2_JESD_mid", "S1_IEC", "S4_AN_ABB", "S4b_AN_CoolCAD"]
rng = np.random.default_rng(7)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 3.0), constrained_layout=True, gridspec_kw={"width_ratios": [1.15, 1]})

# (a) distributions, log ratio axis
ax1.axvspan(1 / 1.3, 1.3, color=BAND, lw=0, zorder=0)
for x in (0.5, 2.0):
    ax1.axvline(x, color="#9a9a9a", lw=0.6, ls=(0, (3, 2)), zorder=0)
ax1.axvline(1.0, color="#444444", lw=0.7, zorder=0)
stats = []
for i, m in enumerate(ORDER):
    s = V[V.in_domain] if m == "FORMULA" else V
    r = (s[m] / s.PARMA).values
    col, _, lab = MODEL[m]
    y = len(ORDER) - 1 - i
    ax1.scatter(r, y + rng.uniform(-0.22, 0.22, len(r)), s=3, color=col, alpha=0.35, linewidths=0, rasterized=True, zorder=1)
    q05, q25, q50, q75, q95 = np.percentile(r, [5, 25, 50, 75, 95])
    ax1.plot([q05, q95], [y, y], color="#222222", lw=0.8, zorder=3)
    ax1.add_patch(plt.Rectangle((q25, y - 0.17), q75 - q25, 0.34, fc="white", ec="#222222", lw=0.8, zorder=3))
    ax1.plot([q50, q50], [y - 0.17, y + 0.17], color=col, lw=2.0, zorder=4)
    out30 = ((r > 1.3) | (r < 1 / 1.3)).mean() * 100; below1 = (r < 1).mean() * 100
    stats.append((m, len(r), q50, out30, below1))
    ax1.text(3.3, y, "%d%%" % round(out30), ha="right", va="center", fontsize=7, color="#222222")
    ax1.text(4.9, y, "%d%%" % round(below1), ha="right", va="center", fontsize=7, color="#222222")
ax1.text(3.3, len(ORDER) - 0.35, "outside\n$\\pm$30%", ha="right", va="bottom", fontsize=6.5, color="#555555")
ax1.text(4.9, len(ORDER) - 0.35, "below\n1", ha="right", va="bottom", fontsize=6.5, color="#555555")
ax1.set_xscale("log")
ax1.set_xlim(0.2, 5.0)
ax1.set_xticks([0.25, 0.5, 0.77, 1, 1.3, 2]); ax1.set_xticklabels(["0.25", "0.5", "0.77", "1", "1.3", "2"])
ax1.minorticks_off()
ax1.set_yticks(range(len(ORDER))); ax1.set_yticklabels([MODEL[m][2] for m in ORDER][::-1])
ax1.set_ylim(-0.6, len(ORDER) - 0.1)
ax1.tick_params(axis="y", length=0)
ax1.spines["left"].set_visible(False)
ax1.set_xlabel("Model / PARMA, flux > 10 MeV")
ax1.text(0.5 * (1 / 1.3 + 1.3) ** 0 * 1.0, -0.55, "", fontsize=6)
panel_label(ax1, "(a)")

# (b) solar-phase dependence: median and 5-95% band per W level
for m in ["S1_IEC", "S2_JESD_mid", "S4_AN_ABB", "FORMULA"]:
    col, ls, lab = MODEL[m]
    g = (A[m] / A.PARMA).groupby(A.W)
    med, lo, hi = g.median(), g.quantile(0.05), g.quantile(0.95)
    ax2.fill_between(med.index, lo, hi, color=col, alpha=0.08, lw=0)
    ax2.plot(lo.index, lo.values, color=col, lw=0.5, alpha=0.6); ax2.plot(hi.index, hi.values, color=col, lw=0.5, alpha=0.6)
    ax2.plot(med.index, med.values, color=col, ls=ls, marker="o", ms=3, lw=1.4, label=lab)
ax2.axhline(1.0, color="#444444", lw=0.7)
ax2.axhspan(1 / 1.3, 1.3, color=BAND, lw=0, zorder=0)
ax2.set_xlabel("Solar activity, W-index (PARMA)")
ax2.set_ylabel("Model / PARMA (median, 5–95%)")
ax2.set_xlim(-5, 155); ax2.set_ylim(0.4, 3.0)
ax2.set_xticks([0, 25, 50, 75, 100, 125, 150])
ax2.text(2, 0.43, "solar\nminimum", fontsize=6.5, color="#555555", va="bottom")
ax2.text(148, 0.43, "solar\nmaximum", fontsize=6.5, color="#555555", va="bottom", ha="right")
ax2.legend(loc="upper left", ncol=2, handlelength=2.2, columnspacing=1.0)
panel_label(ax2, "(b)")

out = HERE / "fig_02_ratio_distributions"
fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
for m, n, q50, o, b in stats:
    print("%-16s n=%d median %.3f outside30 %.1f%% below1 %.1f%%" % (m, n, q50, o, b))
for m in ["S1_IEC", "S2_JESD_mid"]:
    med = (A[m] / A.PARMA).groupby(A.W).median()
    print(m, "median at W=0: %.3f, W=150: %.3f" % (med.loc[0], med.loc[150]))
