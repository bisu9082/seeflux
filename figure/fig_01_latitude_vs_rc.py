# Fig. 1: geographic latitude does not determine the flux; cutoff rigidity does.
# Data: data/step4/audit_nodes.csv (PARMA on the 5 x 10 deg geographic grid), 12 km, W-index 100.
# Design (v2): muted sequential colour for data, two high-contrast accent colours for standard models,
# black for the surrogate, in-panel annotation of the key number. Two colour variants for author choice.
import sys, pathlib, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
HERE = pathlib.Path(__file__).resolve().parent; ROOT = HERE.parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(ROOT / "code"))
from figstyle import DOUBLE, panel_label
import standards as S
from seeflux_formula import flux_gt10

A = pd.read_csv(ROOT / "data/step4/audit_nodes.csv")
d = A[np.isclose(A.alt_km, 12.0) & (A.W == 100)].copy()
d["alat"] = d.lat.abs()
g = d.groupby("alat").PARMA.agg(["min", "max"])

VARIANTS = {
    # A: blue sequential data + vermillion/amber accents (blue-orange contrast, colour-vision safe)
    "A": dict(cmap=LinearSegmentedColormap.from_list("blues", ["#a6c8e4", "#5a9fd0", "#2171b5", "#08306b"]),
              iec="#D55E00", an="#E69F00", sur="#111111", band="#e9eef4"),
    # B: viridis data + magenta/black accents
    "B": dict(cmap=plt.get_cmap("viridis"), iec="#CC3377", an="#555555", sur="#111111", band="#efefef"),
}


def draw(v, tag):
    c = VARIANTS[v]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(DOUBLE, 2.9), constrained_layout=True)

    # (a) flux vs geographic latitude
    ax1.fill_between(g.index, g["min"], g["max"], color=c["band"], lw=0, zorder=0)
    sc = ax1.scatter(d.alat, d.PARMA, c=d.Rc, cmap=c["cmap"], s=9, edgecolors="white", linewidths=0.25,
                     zorder=2, rasterized=True)
    lat = np.linspace(0, 85, 300)
    ax1.plot(lat, S.s1_iec_boeing(12.0, lat), color=c["iec"], lw=1.6, zorder=3)
    ax1.plot(lat, S.s4_an4589_abb(12.0, lat), color=c["an"], lw=1.6, ls=(0, (4, 2)), zorder=3)
    ax1.text(78, S.s1_iec_boeing(12.0, 78.0) + 250, "IEC 62396-1", color=c["iec"], ha="right", va="bottom", fontsize=7)
    ax1.text(78, S.s4_an4589_abb(12.0, 78.0) + 250, "AN4589", color=c["an"], ha="right", va="bottom", fontsize=7)
    # annotate spread at 30 deg
    lo, hi = g.loc[30, "min"], g.loc[30, "max"]
    ax1.annotate("", xy=(32.0, hi), xytext=(32.0, lo), arrowprops=dict(arrowstyle="|-|", lw=0.8, color="#222222",
                                                                        shrinkA=0, shrinkB=0, mutation_scale=2.5))
    ax1.annotate("%.1f$\\times$ spread\nat 30$^{\\circ}$" % (hi / lo), xy=(32.0, hi), xytext=(14.0, 3900),
                 ha="center", va="bottom", fontsize=7, fontweight="bold", color="#222222",
                 arrowprops=dict(arrowstyle="-", lw=0.6, color="#555555", shrinkA=1, shrinkB=1))
    ax1.set_xlabel("Geographic latitude |$\\lambda$| (deg)")
    ax1.set_ylabel("Neutron flux > 10 MeV (cm$^{-2}$ h$^{-1}$)")
    ax1.set_xlim(0, 85); ax1.set_ylim(0, 8500)
    cb = fig.colorbar(sc, ax=ax1, pad=0.015, aspect=28, shrink=0.9)
    cb.set_label("Cutoff rigidity $R_c$ (GV)"); cb.outline.set_visible(False); cb.ax.tick_params(length=2)
    ax1.legend(handles=[Line2D([], [], marker="o", ls="", ms=3.5, mfc="#6baed6" if v == "A" else "#3b8a8c", mec="white",
                               mew=0.3, label="PARMA, 1260 locations"),
                        plt.Rectangle((0, 0), 1, 1, fc=c["band"], ec="none", label="Range across longitude")],
               loc="upper left", handlelength=1.2)
    panel_label(ax1, "(a)")

    # (b) the same points against Rc, coloured by latitude: all latitudes fall on one curve
    sc2 = ax2.scatter(d.Rc, d.PARMA, c=d.alat, cmap=c["cmap"], s=9, edgecolors="white", linewidths=0.25,
                      zorder=2, rasterized=True)
    rc = np.linspace(0, 17.6, 300)
    ax2.plot(rc, flux_gt10(12.0, rc, 100.0), color=c["sur"], lw=1.3, zorder=3)
    ax2.text(9.0, flux_gt10(12.0, 9.0, 100.0) + 300, "Surrogate, Eq. (1)", fontsize=7, ha="left", va="bottom")
    ax2.set_xlabel("Cutoff rigidity $R_c$ (GV)")
    ax2.set_ylabel("Neutron flux > 10 MeV (cm$^{-2}$ h$^{-1}$)")
    ax2.set_xlim(0, 18); ax2.set_ylim(0, 5000)
    cb2 = fig.colorbar(sc2, ax=ax2, pad=0.015, aspect=28, shrink=0.9)
    cb2.set_label("Geographic latitude |$\\lambda$| (deg)"); cb2.outline.set_visible(False); cb2.ax.tick_params(length=2)
    panel_label(ax2, "(b)")

    out = HERE / f"fig_01_latitude_vs_rc{tag}"
    fig.savefig(out.with_suffix(".pdf")); fig.savefig(out.with_suffix(".png"))
    plt.close(fig)


draw("A", "")  # author choice 2026-10-15: variant A (paper theme); variant B kept in figure/alt/ for record
print("max/min at |lat|=30: %.2f, 45: %.2f; Rc span at 30: %.1f-%.1f GV" %
      (g.loc[30, "max"] / g.loc[30, "min"], g.loc[45, "max"] / g.loc[45, "min"],
       d[d.alat == 30].Rc.min(), d[d.alat == 30].Rc.max()))
r = d.PARMA / flux_gt10(12.0, d.Rc, 100.0)
print("PARMA/surrogate at these nodes: %.3f-%.3f" % (r.min(), r.max()))
import json
json.dump({"spread_30deg": float(g.loc[30, "max"] / g.loc[30, "min"]), "spread_45deg": float(g.loc[45, "max"] / g.loc[45, "min"]),
           "Rc_range_30deg": [float(d[d.alat == 30].Rc.min()), float(d[d.alat == 30].Rc.max())],
           "PARMA_over_surrogate_range": [float(r.min()), float(r.max())],
           "PARMA_over_surrogate_max_Rc_below2": float(r[d.Rc < 2].max())},
          open(ROOT / "data/step4/fig1_stats.json", "w"), indent=1)
